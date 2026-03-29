"""LangGraph adapter — translates LangChain callbacks into universal events."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from orcheval.adapters.base import BaseAdapter
from orcheval.events import (
    ErrorEvent,
    LLMCall,
    NodeEntry,
    NodeExit,
    RoutingDecision,
    ToolCall,
)
from orcheval.sanitize import compute_state_diff, sanitize_outputs, sanitize_state


def _normalize_tool_calls(raw: list[Any]) -> list[dict[str, Any]]:
    """Normalize LangChain tool call formats into ``[{"name": ..., "args": ...}]``."""
    result: list[dict[str, Any]] = []
    for tc in raw:
        if isinstance(tc, dict):
            name = tc.get("name") or tc.get("function", {}).get("name", "")
            args = tc.get("args") or tc.get("function", {}).get("arguments", "")
        else:
            # Dataclass-style (e.g. LangChain ToolCall objects)
            name = getattr(tc, "name", "")
            args = getattr(tc, "args", "")
        result.append({"name": name, "args": args})
    return result


def _ensure_langchain() -> None:
    """Raise a helpful ImportError if langchain_core is not installed."""
    try:
        import langchain_core  # noqa: F401
    except ImportError:
        raise ImportError(
            "langchain_core is required for the LangGraph adapter. "
            "Install it with: pip install orcheval[langgraph]"
        ) from None


class LangGraphAdapter(BaseAdapter):
    """Adapter for LangGraph / LangChain callback-based tracing.

    Usage::

        from orcheval import Tracer

        tracer = Tracer(adapter="langgraph")
        result = graph.invoke(input, config={"callbacks": [tracer.handler]})
        trace = tracer.collect()
    """

    def __init__(
        self,
        trace_id: str,
        *,
        infer_routing: bool = False,
        capture_state: bool = False,
    ) -> None:
        _ensure_langchain()
        super().__init__(trace_id)
        self._infer_routing = infer_routing
        self._capture_state = capture_state
        self._handler = _create_callback_handler(self)

    def get_callback_handler(self) -> Any:
        return self._handler


def _extract_node_name(metadata: dict[str, Any] | None) -> str | None:
    """Extract the LangGraph node name from callback metadata."""
    if metadata is None:
        return None
    return metadata.get("langgraph_node")


def _create_callback_handler(adapter: LangGraphAdapter) -> Any:
    """Create and return a LangChain BaseCallbackHandler wired to the adapter.

    This function is called at adapter init time, after langchain_core
    availability has been verified.
    """
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult  # noqa: TC002

    class _Handler(BaseCallbackHandler):
        """Internal callback handler. Not part of the public API."""

        def __init__(self, adapter: LangGraphAdapter) -> None:
            super().__init__()
            self._adapter = adapter
            self._lock = threading.RLock()

            # Span tracking
            self._run_to_span: dict[str, str] = {}  # run_id -> span_id
            self._run_to_node: dict[str, str] = {}  # run_id -> node_name
            self._span_stack: list[str] = []  # stack of span_ids for nesting

            # Nested chain deduplication: track outermost active span per node
            # and suppress inner spans from emitting NodeEntry/NodeExit.
            self._active_node_spans: dict[str, str] = {}  # node_name -> outermost span_id
            self._suppressed_spans: set[str] = set()  # suppressed inner span_ids

            # Pending start data for pairing start/end callbacks
            self._pending_llm: dict[str, dict[str, Any]] = {}
            self._pending_tool: dict[str, dict[str, Any]] = {}

            # Node entry timestamps for duration calculation
            self._node_entry_times: dict[str, datetime] = {}

            # Sanitized entry states for state diff computation (span_id -> state)
            self._entry_states: dict[str, dict[str, Any]] = {}

            # Last exited node for routing inference
            self._last_exited_node: str | None = None

            # Last node outputs for populating decision_context
            self._last_exit_outputs: dict[str, Any] = {}

        @property
        def _current_parent_span(self) -> str | None:
            for span_id in reversed(self._span_stack):
                if span_id not in self._suppressed_spans:
                    return span_id
            return None

        @property
        def _current_node_name(self) -> str | None:
            for span_id in reversed(self._span_stack):
                if span_id not in self._suppressed_spans:
                    return self._run_to_node.get(span_id)
            return None

        def _make_span_id(self) -> str:
            return uuid.uuid4().hex

        # --- Chain callbacks (graph node boundaries) ---

        def on_chain_start(
            self,
            serialized: dict[str, Any],
            inputs: dict[str, Any] | Any,
            *,
            run_id: Any,
            parent_run_id: Any | None = None,
            tags: list[str] | None = None,
            metadata: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            run_id_str = str(run_id)
            node_name = _extract_node_name(metadata)
            span_id = self._make_span_id()

            with self._lock:
                self._run_to_span[run_id_str] = span_id

                if node_name is not None:
                    self._run_to_node[span_id] = node_name

                    if node_name in self._active_node_spans:
                        # Nested chain for already-active node — suppress
                        self._span_stack.append(span_id)
                        self._suppressed_spans.add(span_id)
                        return

                    # First entry for this node — register as outermost
                    self._active_node_spans[node_name] = span_id
                    parent = self._current_parent_span
                    self._span_stack.append(span_id)

                    now = datetime.now(timezone.utc)
                    self._node_entry_times[span_id] = now

                    # Emit inferred routing decision before the NodeEntry
                    if (
                        self._adapter._infer_routing
                        and self._last_exited_node is not None
                    ):
                        # node_name == source_node by design: the decision belongs to the node making the routing choice
                        routing_event = RoutingDecision(
                            trace_id=self._adapter.trace_id,
                            span_id=self._make_span_id(),
                            timestamp=now,
                            node_name=self._last_exited_node,
                            source_node=self._last_exited_node,
                            target_node=node_name,
                            decision_context=self._last_exit_outputs,
                            metadata={"inferred": True},
                        )
                        self._adapter._emit(routing_event)
                        self._last_exit_outputs = {}

                    # Capture input state when opt-in is enabled
                    input_state: dict[str, Any] = {}
                    if self._adapter._capture_state:
                        input_state = sanitize_state(inputs) if isinstance(inputs, dict) else {}
                        self._entry_states[span_id] = input_state

                    event = NodeEntry(
                        trace_id=self._adapter.trace_id,
                        span_id=span_id,
                        parent_span_id=parent,
                        timestamp=now,
                        node_name=node_name,
                        input_state=input_state,
                    )
                    self._adapter._emit(event)
                else:
                    self._span_stack.append(span_id)

        def on_chain_end(
            self,
            outputs: dict[str, Any] | Any,
            *,
            run_id: Any,
            parent_run_id: Any | None = None,
            **kwargs: Any,
        ) -> None:
            run_id_str = str(run_id)

            with self._lock:
                span_id = self._run_to_span.pop(run_id_str, None)
                if span_id is None:
                    return

                node_name = self._run_to_node.pop(span_id, None)

                # Pop from span stack (search from end for correct stack behaviour)
                try:
                    idx = len(self._span_stack) - 1 - self._span_stack[::-1].index(span_id)
                    self._span_stack.pop(idx)
                except ValueError:
                    pass

                if node_name is not None:
                    # Suppressed inner span — clean up and skip NodeExit
                    if span_id in self._suppressed_spans:
                        self._suppressed_spans.discard(span_id)
                        return

                    # Outermost span — clean up active tracking and emit NodeExit
                    if self._active_node_spans.get(node_name) == span_id:
                        del self._active_node_spans[node_name]

                    now = datetime.now(timezone.utc)
                    duration_ms = None
                    entry_time = self._node_entry_times.pop(span_id, None)
                    if entry_time is not None:
                        duration_ms = (now - entry_time).total_seconds() * 1000

                    # Capture output state and compute diff when opt-in is enabled
                    output_state: dict[str, Any] = {}
                    state_diff: dict[str, Any] = {}
                    entry_state = self._entry_states.pop(span_id, {})
                    if self._adapter._capture_state:
                        output_state = (
                            sanitize_state(outputs) if isinstance(outputs, dict) else {}
                        )
                        state_diff = compute_state_diff(entry_state, output_state)

                    event = NodeExit(
                        trace_id=self._adapter.trace_id,
                        span_id=span_id,
                        timestamp=now,
                        node_name=node_name,
                        duration_ms=duration_ms,
                        output_state=output_state,
                        state_diff=state_diff,
                    )
                    self._adapter._emit(event)
                    self._last_exited_node = node_name
                    self._last_exit_outputs = sanitize_outputs(outputs)

        def on_chain_error(
            self,
            error: BaseException,
            *,
            run_id: Any,
            parent_run_id: Any | None = None,
            **kwargs: Any,
        ) -> None:
            run_id_str = str(run_id)

            with self._lock:
                span_id = self._run_to_span.get(run_id_str, self._make_span_id())
                node_name = self._current_node_name

                err_event = ErrorEvent(
                    trace_id=self._adapter.trace_id,
                    span_id=span_id,
                    node_name=node_name,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
                self._adapter._emit(err_event)

            # Delegate to on_chain_end for cleanup
            self.on_chain_end({}, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

        # --- LLM callbacks ---

        def on_llm_start(
            self,
            serialized: dict[str, Any],
            prompts: list[str],
            *,
            run_id: Any,
            parent_run_id: Any | None = None,
            tags: list[str] | None = None,
            metadata: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            run_id_str = str(run_id)

            model_name = None
            if serialized:
                model_name = (
                    serialized.get("kwargs", {}).get("model_name")
                    or serialized.get("kwargs", {}).get("model")
                )

            # Also check metadata for model name (langchain convention)
            if model_name is None and metadata:
                model_name = metadata.get("ls_model_name")

            with self._lock:
                self._pending_llm[run_id_str] = {
                    "start_time": datetime.now(timezone.utc),
                    "model": model_name,
                    "prompts": prompts,
                }

        def on_chat_model_start(
            self,
            serialized: dict[str, Any],
            messages: list[list[Any]],
            *,
            run_id: Any,
            parent_run_id: Any | None = None,
            tags: list[str] | None = None,
            metadata: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            """Handle ChatModel start — preserves role structure and extracts system message.

            Mutually exclusive with ``on_llm_start`` per run_id — LangChain
            dispatches one or the other, never both.
            """
            run_id_str = str(run_id)

            model_name = None
            if serialized:
                model_name = (
                    serialized.get("kwargs", {}).get("model_name")
                    or serialized.get("kwargs", {}).get("model")
                )
            if model_name is None and metadata:
                model_name = metadata.get("ls_model_name")

            with self._lock:
                pending: dict[str, Any] = {
                    "start_time": datetime.now(timezone.utc),
                    "model": model_name,
                }

                try:
                    # messages is list[list[BaseMessage]] — use first batch
                    chat_msgs = messages[0] if messages else []

                    # Extract system message first (stored in dedicated field)
                    sys_msg = next(
                        (
                            m
                            for m in chat_msgs
                            if getattr(m, "type", None) == "system"
                        ),
                        None,
                    )
                    if sys_msg is not None:
                        pending["system_message"] = getattr(sys_msg, "content", "")

                    # Build chat_messages, excluding system (already extracted)
                    # and capturing tool_calls from AI messages
                    chat_messages: list[dict[str, Any]] = []
                    for m in chat_msgs:
                        if getattr(m, "type", None) == "system":
                            continue
                        msg_dict: dict[str, Any] = {
                            "role": getattr(m, "type", "unknown"),
                            "content": getattr(m, "content", ""),
                        }
                        if getattr(m, "type", None) == "ai":
                            tc = getattr(m, "tool_calls", None) or []
                            if not tc:
                                tc = getattr(m, "additional_kwargs", {}).get(
                                    "tool_calls", []
                                )
                            if tc:
                                msg_dict["tool_calls"] = _normalize_tool_calls(tc)
                        chat_messages.append(msg_dict)
                    pending["chat_messages"] = chat_messages
                except Exception:
                    # If structured parsing fails, store minimal pending entry
                    # so on_llm_end can still emit an event
                    pass

                self._pending_llm[run_id_str] = pending

        def on_llm_end(
            self,
            response: LLMResult,
            *,
            run_id: Any,
            parent_run_id: Any | None = None,
            **kwargs: Any,
        ) -> None:
            run_id_str = str(run_id)

            with self._lock:
                pending = self._pending_llm.pop(run_id_str, {})
                start_time = pending.get("start_time")
                model = pending.get("model")

                now = datetime.now(timezone.utc)
                duration_ms = None
                if start_time is not None:
                    duration_ms = (now - start_time).total_seconds() * 1000

                # Extract token usage
                input_tokens = None
                output_tokens = None
                if response.llm_output:
                    usage = response.llm_output.get("token_usage", {})
                    prompt_toks = usage.get("prompt_tokens")
                    completion_toks = usage.get("completion_tokens")
                    input_tokens = prompt_toks
                    output_tokens = completion_toks

                # Extract response text
                output_message: dict[str, Any] | None = None
                response_summary: str | None = None
                if response.generations and response.generations[0]:
                    gen = response.generations[0][0]
                    response_summary = gen.text[:200] if gen.text else None
                    output_message = {"content": gen.text}
                    if hasattr(gen, "message") and gen.message is not None:
                        msg = gen.message
                        output_message = {
                            "role": getattr(msg, "type", "ai"),
                            "content": getattr(msg, "content", gen.text),
                        }
                        tc = getattr(msg, "tool_calls", None) or []
                        if not tc:
                            tc = getattr(msg, "additional_kwargs", {}).get(
                                "tool_calls", []
                            )
                        if tc:
                            output_message["tool_calls"] = _normalize_tool_calls(tc)

                # Build input messages: use structured chat messages if available
                # (from on_chat_model_start), else fall back to flat prompts
                # (from on_llm_start).
                chat_messages = pending.get("chat_messages")
                if chat_messages is not None:
                    input_messages = chat_messages
                    # Build prompt_summary from first user message content
                    first_content = next(
                        (
                            m.get("content", "")
                            for m in chat_messages
                            if m.get("role") != "system"
                        ),
                        "",
                    )
                    prompt_summary = first_content[:200] if first_content else None
                else:
                    prompts = pending.get("prompts", [])
                    input_messages = [{"role": "user", "content": p} for p in prompts]
                    prompt_summary = prompts[0][:200] if prompts else None

                # Extract system message (only available via on_chat_model_start)
                system_message = pending.get("system_message")

                span_id = self._make_span_id()
                event = LLMCall(
                    trace_id=self._adapter.trace_id,
                    span_id=span_id,
                    parent_span_id=self._current_parent_span,
                    timestamp=now,
                    node_name=self._current_node_name,
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

        def on_llm_error(
            self,
            error: BaseException,
            *,
            run_id: Any,
            parent_run_id: Any | None = None,
            **kwargs: Any,
        ) -> None:
            with self._lock:
                self._pending_llm.pop(str(run_id), None)

                event = ErrorEvent(
                    trace_id=self._adapter.trace_id,
                    span_id=self._make_span_id(),
                    parent_span_id=self._current_parent_span,
                    node_name=self._current_node_name,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
                self._adapter._emit(event)

        # --- Tool callbacks ---

        def on_tool_start(
            self,
            serialized: dict[str, Any],
            input_str: str,
            *,
            run_id: Any,
            parent_run_id: Any | None = None,
            tags: list[str] | None = None,
            metadata: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            run_id_str = str(run_id)
            tool_name = serialized.get("name", "unknown") if serialized else "unknown"

            with self._lock:
                self._pending_tool[run_id_str] = {
                    "start_time": datetime.now(timezone.utc),
                    "tool_name": tool_name,
                    "input_str": input_str,
                }

        def on_tool_end(
            self,
            output: str,
            *,
            run_id: Any,
            parent_run_id: Any | None = None,
            **kwargs: Any,
        ) -> None:
            run_id_str = str(run_id)

            with self._lock:
                pending = self._pending_tool.pop(run_id_str, {})
                start_time = pending.get("start_time")
                tool_name = pending.get("tool_name", "unknown")
                input_str = pending.get("input_str", "")

                now = datetime.now(timezone.utc)
                duration_ms = None
                if start_time is not None:
                    duration_ms = (now - start_time).total_seconds() * 1000

                # Parse input string as tool_input dict
                tool_input: dict[str, Any] = {"raw": input_str}

                span_id = self._make_span_id()
                event = ToolCall(
                    trace_id=self._adapter.trace_id,
                    span_id=span_id,
                    parent_span_id=self._current_parent_span,
                    timestamp=now,
                    node_name=self._current_node_name,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_output=str(output)[:500] if output else None,
                    duration_ms=duration_ms,
                )
                self._adapter._emit(event)

        def on_tool_error(
            self,
            error: BaseException,
            *,
            run_id: Any,
            parent_run_id: Any | None = None,
            **kwargs: Any,
        ) -> None:
            with self._lock:
                self._pending_tool.pop(str(run_id), None)

                event = ErrorEvent(
                    trace_id=self._adapter.trace_id,
                    span_id=self._make_span_id(),
                    parent_span_id=self._current_parent_span,
                    node_name=self._current_node_name,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
                self._adapter._emit(event)

    return _Handler(adapter)
