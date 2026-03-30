"""OpenAI Agents SDK adapter — translates SDK tracing spans into universal events.

This adapter hooks into the OpenAI Agents SDK's ``TracingProcessor`` interface
to capture agent executions, LLM generations, tool calls, handoffs, and
guardrail checks as OrchEval events.

**capture_state semantics:** Unlike the LangGraph adapter, which captures the
full graph state dict, the OpenAI Agents SDK does not expose typed input/output
state on agent spans.  When ``capture_state=True``, this adapter captures
*agent metadata* (name, tools, handoffs, output_type) rather than message
history or graph state.  The resulting ``state_diff`` reflects configuration
changes between entry and exit, not data flow.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from orcheval.adapters.base import BaseAdapter
from orcheval.events import (
    AgentMessage,
    ErrorEvent,
    LLMCall,
    NodeEntry,
    NodeExit,
    RoutingDecision,
    ToolCall,
)
from orcheval.sanitize import compute_state_diff, sanitize_outputs, sanitize_state

import warnings


def _ensure_openai_agents() -> None:
    """Raise a helpful ImportError if the openai-agents SDK is not installed."""
    try:
        import agents  # noqa: F401
    except ImportError:
        raise ImportError(
            "openai-agents is required for the OpenAI Agents adapter. "
            "Install it with: pip install orcheval[openai_agents]"
        ) from None


def _parse_iso(s: str | None) -> datetime | None:
    """Parse an ISO 8601 timestamp string, handling the ``Z`` suffix.

    Python 3.10's ``datetime.fromisoformat`` does not accept the ``Z`` UTC
    suffix (support was added in 3.11).  This helper normalises ``Z`` to
    ``+00:00`` before parsing.
    """
    if s is None:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


class OpenAIAgentsAdapter(BaseAdapter):
    """Adapter for the OpenAI Agents SDK tracing system.

    Usage::

        from orcheval import Tracer
        from agents.tracing import add_trace_processor

        tracer = Tracer(adapter="openai_agents")
        add_trace_processor(tracer.handler)

        result = await Runner.run(agent, input)
        trace = tracer.collect()
    """

    def __init__(
        self,
        trace_id: str,
        *,
        infer_routing: bool = False,
        capture_state: bool = False,
    ) -> None:
        _ensure_openai_agents()
        warnings.warn(
            "OpenAIAgentsAdapter is experimental and has not been validated "
            "against real workloads. Use ManualAdapter as a fallback if you "
            "encounter issues.",
            UserWarning,
            stacklevel=2,
        )
        super().__init__(trace_id)
        self._infer_routing = infer_routing
        self._capture_state = capture_state
        self._processor = _create_tracing_processor(self)

    def get_callback_handler(self) -> Any:
        return self._processor


def _create_tracing_processor(adapter: OpenAIAgentsAdapter) -> Any:
    """Create and return a ``TracingProcessor`` wired to *adapter*.

    SDK types are imported inside this function so that the module-level
    import guard (``_ensure_openai_agents``) has already run.
    """
    from agents.tracing import (
        AgentSpanData,
        FunctionSpanData,
        GenerationSpanData,
        GuardrailSpanData,
        HandoffSpanData,
        TracingProcessor,
    )

    class _Processor(TracingProcessor):  # type: ignore[misc]
        """Internal tracing processor.  Not part of the public API."""

        def __init__(self, adapter: OpenAIAgentsAdapter) -> None:
            self._adapter = adapter
            self._lock = threading.RLock()

            # --- Span tracking ---
            # Maps SDK span_id -> OrchEval-generated span_id (uuid4 hex).
            # Needed so child events can reference the correct parent_span_id.
            self._sdk_to_orcheval_span: dict[str, str] = {}

            # Maps SDK agent span_id -> agent name
            self._span_to_node: dict[str, str] = {}

            # For *any* span, maps SDK span_id -> enclosing agent name
            self._parent_agent_for_span: dict[str, str] = {}
            # For *any* span, maps SDK span_id -> enclosing agent's SDK span_id
            self._parent_agent_span_for_span: dict[str, str] = {}

            # --- Timing ---
            self._agent_entry_times: dict[str, datetime] = {}

            # --- State capture ---
            self._entry_states: dict[str, dict[str, Any]] = {}

            # --- Routing inference ---
            self._last_exited_agent: str | None = None
            self._last_exit_outputs: dict[str, Any] = {}
            # Agent names that are targets of explicit handoffs; used to
            # suppress duplicate inferred RoutingDecisions.
            self._handoff_targets: set[str] = set()

        def _make_span_id(self) -> str:
            return uuid.uuid4().hex

        # ------------------------------------------------------------------
        # Parent-context resolution helpers
        # ------------------------------------------------------------------

        def _resolve_parent_context(self, sdk_span_id: str, sdk_parent_id: str | None) -> None:
            """Populate ``_parent_agent_for_span`` for a non-agent span.

            Walks up through ``sdk_parent_id`` to find the nearest enclosing
            agent span, recording the result so ``on_span_end`` can set
            ``node_name`` and ``parent_span_id`` on child events.
            """
            if sdk_parent_id is None:
                return

            # Direct parent is an agent span?
            if sdk_parent_id in self._span_to_node:
                self._parent_agent_for_span[sdk_span_id] = self._span_to_node[sdk_parent_id]
                self._parent_agent_span_for_span[sdk_span_id] = sdk_parent_id
                return

            # Parent is itself a child — inherit its agent context.
            if sdk_parent_id in self._parent_agent_for_span:
                self._parent_agent_for_span[sdk_span_id] = (
                    self._parent_agent_for_span[sdk_parent_id]
                )
                self._parent_agent_span_for_span[sdk_span_id] = (
                    self._parent_agent_span_for_span[sdk_parent_id]
                )

        def _get_agent_context(
            self, sdk_span_id: str
        ) -> tuple[str | None, str | None]:
            """Return ``(agent_name, orcheval_parent_span_id)`` for *sdk_span_id*."""
            agent_name = self._parent_agent_for_span.get(sdk_span_id)
            parent_sdk = self._parent_agent_span_for_span.get(sdk_span_id)
            orcheval_parent = (
                self._sdk_to_orcheval_span.get(parent_sdk) if parent_sdk else None
            )
            return agent_name, orcheval_parent

        # ------------------------------------------------------------------
        # TracingProcessor interface
        # ------------------------------------------------------------------

        def on_trace_start(self, trace: Any) -> None:  # noqa: ARG002
            pass

        def on_trace_end(self, trace: Any) -> None:  # noqa: ARG002
            pass

        def shutdown(self) -> None:
            pass

        def force_flush(self) -> None:
            pass

        # ------------------------------------------------------------------
        # Span start
        # ------------------------------------------------------------------

        def on_span_start(self, span: Any) -> None:
            span_data = span.span_data
            sdk_span_id: str = span.span_id
            sdk_parent_id: str | None = getattr(span, "parent_id", None)

            with self._lock:
                if isinstance(span_data, AgentSpanData):
                    self._handle_agent_span_start(span_data, sdk_span_id, sdk_parent_id)
                else:
                    self._resolve_parent_context(sdk_span_id, sdk_parent_id)

        def _handle_agent_span_start(
            self,
            span_data: Any,
            sdk_span_id: str,
            sdk_parent_id: str | None,
        ) -> None:
            agent_name: str = getattr(span_data, "name", None) or "unknown_agent"
            orcheval_span_id = self._make_span_id()

            # Register mappings
            self._sdk_to_orcheval_span[sdk_span_id] = orcheval_span_id
            self._span_to_node[sdk_span_id] = agent_name
            # An agent span is its own agent context
            self._parent_agent_for_span[sdk_span_id] = agent_name
            self._parent_agent_span_for_span[sdk_span_id] = sdk_span_id

            now = datetime.now(timezone.utc)
            self._agent_entry_times[sdk_span_id] = now

            # Parent span (for nested agents / agents-as-tools)
            parent_orcheval = (
                self._sdk_to_orcheval_span.get(sdk_parent_id)
                if sdk_parent_id
                else None
            )

            # --- Inferred routing ---
            if (
                self._adapter._infer_routing
                and self._last_exited_agent is not None
            ):
                if agent_name in self._handoff_targets:
                    # An explicit handoff already emitted a RoutingDecision
                    # for this target — suppress the inferred duplicate.
                    self._handoff_targets.discard(agent_name)
                else:
                    # node_name == source_node by design: the decision belongs to the node making the routing choice
                    routing_event = RoutingDecision(
                        trace_id=self._adapter.trace_id,
                        span_id=self._make_span_id(),
                        timestamp=now,
                        node_name=self._last_exited_agent,
                        source_node=self._last_exited_agent,
                        target_node=agent_name,
                        decision_context=self._last_exit_outputs,
                        metadata={"inferred": True},
                    )
                    self._adapter._emit(routing_event)
                self._last_exited_agent = None
                self._last_exit_outputs = {}

            # --- State capture ---
            input_state: dict[str, Any] = {}
            if self._adapter._capture_state:
                input_state = {
                    "agent_name": agent_name,
                    "tools": getattr(span_data, "tools", None),
                    "handoffs": getattr(span_data, "handoffs", None),
                }
                self._entry_states[sdk_span_id] = input_state

            # --- Emit NodeEntry ---
            event = NodeEntry(
                trace_id=self._adapter.trace_id,
                span_id=orcheval_span_id,
                parent_span_id=parent_orcheval,
                timestamp=now,
                node_name=agent_name,
                input_state=input_state,
            )
            self._adapter._emit(event)

        # ------------------------------------------------------------------
        # Span end
        # ------------------------------------------------------------------

        def on_span_end(self, span: Any) -> None:
            span_data = span.span_data
            sdk_span_id: str = span.span_id

            # Compute duration from ISO timestamps
            started = _parse_iso(getattr(span, "started_at", None))
            ended = _parse_iso(getattr(span, "ended_at", None))
            duration_ms: float | None = None
            if started is not None and ended is not None:
                duration_ms = (ended - started).total_seconds() * 1000

            with self._lock:
                # --- Error check ---
                # For agent spans, parent_span_id resolves to the agent's own span
                # (the error is attributed to the agent that encountered it).
                error = getattr(span, "error", None)
                if error is not None:
                    agent_name_err, parent_err = self._get_agent_context(sdk_span_id)
                    err_msg = (
                        error.get("message", str(error))
                        if isinstance(error, dict)
                        else str(error)
                    )
                    err_event = ErrorEvent(
                        trace_id=self._adapter.trace_id,
                        span_id=self._make_span_id(),
                        parent_span_id=parent_err,
                        node_name=agent_name_err,
                        error_type="SpanError",
                        error_message=err_msg,
                    )
                    self._adapter._emit(err_event)

                # --- Dispatch by span data type ---
                if isinstance(span_data, AgentSpanData):
                    self._handle_agent_span_end(span_data, sdk_span_id, duration_ms)
                elif isinstance(span_data, GenerationSpanData):
                    self._handle_generation_span_end(span_data, sdk_span_id, duration_ms)
                elif isinstance(span_data, FunctionSpanData):
                    self._handle_function_span_end(span_data, sdk_span_id, duration_ms)
                elif isinstance(span_data, HandoffSpanData):
                    self._handle_handoff_span_end(span_data, sdk_span_id)
                elif isinstance(span_data, GuardrailSpanData):
                    self._handle_guardrail_span_end(span_data, sdk_span_id, duration_ms)

                # --- Cleanup non-agent spans ---
                if not isinstance(span_data, AgentSpanData):
                    self._parent_agent_for_span.pop(sdk_span_id, None)
                    self._parent_agent_span_for_span.pop(sdk_span_id, None)

        # --- Agent span end -> NodeExit ---

        def _handle_agent_span_end(
            self,
            span_data: Any,
            sdk_span_id: str,
            duration_ms: float | None,
        ) -> None:
            orcheval_span_id = self._sdk_to_orcheval_span.get(sdk_span_id, self._make_span_id())
            agent_name = self._span_to_node.get(sdk_span_id, "unknown_agent")

            # State capture
            output_state: dict[str, Any] = {}
            state_diff: dict[str, Any] = {}
            entry_state = self._entry_states.pop(sdk_span_id, {})
            if self._adapter._capture_state:
                output_state = {
                    "agent_name": getattr(span_data, "name", None),
                    "output_type": getattr(span_data, "output_type", None),
                    "tools": getattr(span_data, "tools", None),
                    "handoffs": getattr(span_data, "handoffs", None),
                }
                state_diff = compute_state_diff(entry_state, output_state)

            event = NodeExit(
                trace_id=self._adapter.trace_id,
                span_id=orcheval_span_id,
                timestamp=datetime.now(timezone.utc),
                node_name=agent_name,
                duration_ms=duration_ms,
                output_state=output_state,
                state_diff=state_diff,
            )
            self._adapter._emit(event)

            self._last_exited_agent = agent_name
            self._last_exit_outputs = sanitize_outputs(
                {"output_type": getattr(span_data, "output_type", None)}
            )

            # Cleanup
            self._sdk_to_orcheval_span.pop(sdk_span_id, None)
            self._span_to_node.pop(sdk_span_id, None)
            self._parent_agent_for_span.pop(sdk_span_id, None)
            self._parent_agent_span_for_span.pop(sdk_span_id, None)
            self._agent_entry_times.pop(sdk_span_id, None)

        # --- Generation span end -> LLMCall ---

        def _handle_generation_span_end(
            self,
            span_data: Any,
            sdk_span_id: str,
            duration_ms: float | None,
        ) -> None:
            agent_name, parent_orcheval = self._get_agent_context(sdk_span_id)

            model: str | None = getattr(span_data, "model", None)

            # Input messages — Sequence[Mapping[str, Any]] | None
            raw_input = getattr(span_data, "input", None)
            input_messages: list[dict[str, Any]] = []
            system_message: str | None = None
            prompt_summary: str | None = None

            if raw_input and isinstance(raw_input, (list, tuple)):
                for msg in raw_input:
                    if isinstance(msg, dict):
                        input_messages.append(dict(msg))
                        if msg.get("role") == "system" and system_message is None:
                            content = msg.get("content", "")
                            system_message = str(content)[:500] if content else None
                # prompt_summary from first non-system message
                first_user = next(
                    (
                        m.get("content", "")
                        for m in input_messages
                        if m.get("role") != "system"
                    ),
                    "",
                )
                prompt_summary = str(first_user)[:200] if first_user else None

            # Output — Sequence[Mapping[str, Any]] | None
            raw_output = getattr(span_data, "output", None)
            output_message: dict[str, Any] | None = None
            response_summary: str | None = None

            if raw_output and isinstance(raw_output, (list, tuple)) and raw_output:
                first_out = raw_output[0]
                if isinstance(first_out, dict):
                    output_message = dict(first_out)
                    content = first_out.get("content", "")
                    response_summary = str(content)[:200] if content else None
            elif isinstance(raw_output, dict):
                output_message = dict(raw_output)
                content = raw_output.get("content", "")
                response_summary = str(content)[:200] if content else None

            # Token usage — dict[str, Any] | None
            usage = getattr(span_data, "usage", None)
            input_tokens: int | None = None
            output_tokens: int | None = None
            if isinstance(usage, dict):
                in_toks = usage.get("input_tokens")
                out_toks = usage.get("output_tokens")
                input_tokens = in_toks
                output_tokens = out_toks

            event = LLMCall(
                trace_id=self._adapter.trace_id,
                span_id=self._make_span_id(),
                parent_span_id=parent_orcheval,
                timestamp=datetime.now(timezone.utc),
                node_name=agent_name,
                model=model,
                input_messages=input_messages,
                output_message=output_message,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=duration_ms,
                prompt_summary=prompt_summary,
                response_summary=response_summary,
                system_message=system_message,
            )
            self._adapter._emit(event)

        # --- Function span end -> ToolCall ---

        def _handle_function_span_end(
            self,
            span_data: Any,
            sdk_span_id: str,
            duration_ms: float | None,
        ) -> None:
            agent_name, parent_orcheval = self._get_agent_context(sdk_span_id)

            tool_name: str = getattr(span_data, "name", None) or "unknown_tool"
            raw_input = getattr(span_data, "input", None)
            raw_output = getattr(span_data, "output", None)

            tool_input: dict[str, Any] = {}
            if isinstance(raw_input, dict):
                tool_input = raw_input
            elif raw_input is not None:
                tool_input = {"raw": str(raw_input)}

            tool_output: str | None = None
            if raw_output is not None:
                tool_output = str(raw_output)[:500]

            event = ToolCall(
                trace_id=self._adapter.trace_id,
                span_id=self._make_span_id(),
                parent_span_id=parent_orcheval,
                timestamp=datetime.now(timezone.utc),
                node_name=agent_name,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=tool_output,
                duration_ms=duration_ms,
            )
            self._adapter._emit(event)

        # --- Handoff span end -> RoutingDecision + AgentMessage ---

        def _handle_handoff_span_end(
            self,
            span_data: Any,
            sdk_span_id: str,
        ) -> None:
            from_agent: str = getattr(span_data, "from_agent", None) or "unknown"
            to_agent: str = getattr(span_data, "to_agent", None) or "unknown"
            now = datetime.now(timezone.utc)

            # Always emit — handoffs are explicit routing, not inferred.
            routing_event = RoutingDecision(
                trace_id=self._adapter.trace_id,
                span_id=self._make_span_id(),
                timestamp=now,
                node_name=from_agent,
                source_node=from_agent,
                target_node=to_agent,
                decision_context={"handoff": True},
                metadata={"inferred": False, "mechanism": "handoff"},
            )
            self._adapter._emit(routing_event)

            msg_event = AgentMessage(
                trace_id=self._adapter.trace_id,
                span_id=self._make_span_id(),
                timestamp=now,
                node_name=from_agent,
                sender=from_agent,
                receiver=to_agent,
                content_summary=f"Handoff from {from_agent} to {to_agent}",
            )
            self._adapter._emit(msg_event)

            # Dedup guard: record the handoff target so the subsequent
            # AgentSpan start for *to_agent* does not also emit an
            # inferred RoutingDecision for the same transition.
            self._handoff_targets.add(to_agent)

        # --- Guardrail span end -> ToolCall ---

        def _handle_guardrail_span_end(
            self,
            span_data: Any,
            sdk_span_id: str,
            duration_ms: float | None,
        ) -> None:
            agent_name, parent_orcheval = self._get_agent_context(sdk_span_id)

            name: str = getattr(span_data, "name", None) or "unknown"
            triggered: bool = getattr(span_data, "triggered", False)

            event = ToolCall(
                trace_id=self._adapter.trace_id,
                span_id=self._make_span_id(),
                parent_span_id=parent_orcheval,
                timestamp=datetime.now(timezone.utc),
                node_name=agent_name,
                tool_name=f"guardrail:{name}",
                tool_input={},
                tool_output=f"triggered={triggered}",
                duration_ms=duration_ms,
                metadata={"guardrail": True, "triggered": triggered},
            )
            self._adapter._emit(event)

    return _Processor(adapter)
