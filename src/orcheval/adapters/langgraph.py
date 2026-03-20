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

    def __init__(self, trace_id: str, *, infer_routing: bool = False) -> None:
        _ensure_langchain()
        super().__init__(trace_id)
        self._infer_routing = infer_routing
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
            self._lock = threading.Lock()

            # Span tracking
            self._run_to_span: dict[str, str] = {}  # run_id -> span_id
            self._run_to_node: dict[str, str] = {}  # run_id -> node_name
            self._span_stack: list[str] = []  # stack of span_ids for nesting

            # Pending start data for pairing start/end callbacks
            self._pending_llm: dict[str, dict[str, Any]] = {}
            self._pending_tool: dict[str, dict[str, Any]] = {}

            # Node entry timestamps for duration calculation
            self._node_entry_times: dict[str, datetime] = {}

            # Last exited node for routing inference
            self._last_exited_node: str | None = None

        @property
        def _current_parent_span(self) -> str | None:
            return self._span_stack[-1] if self._span_stack else None

        @property
        def _current_node_name(self) -> str | None:
            return self._run_to_node.get(self._span_stack[-1]) if self._span_stack else None

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
                    parent = self._current_parent_span
                    self._span_stack.append(span_id)

                    now = datetime.now(timezone.utc)
                    self._node_entry_times[span_id] = now

                    # Emit inferred routing decision before the NodeEntry
                    if (
                        self._adapter._infer_routing
                        and self._last_exited_node is not None
                        and self._last_exited_node != node_name
                    ):
                        routing_event = RoutingDecision(
                            trace_id=self._adapter.trace_id,
                            span_id=self._make_span_id(),
                            timestamp=now,
                            node_name=self._last_exited_node,
                            source_node=self._last_exited_node,
                            target_node=node_name,
                            metadata={"inferred": True},
                        )
                        self._adapter._emit(routing_event)

                    event = NodeEntry(
                        trace_id=self._adapter.trace_id,
                        span_id=span_id,
                        parent_span_id=parent,
                        timestamp=now,
                        node_name=node_name,
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

                # Pop from span stack
                if span_id in self._span_stack:
                    self._span_stack.remove(span_id)

                if node_name is not None:
                    now = datetime.now(timezone.utc)
                    duration_ms = None
                    entry_time = self._node_entry_times.pop(span_id, None)
                    if entry_time is not None:
                        duration_ms = (now - entry_time).total_seconds() * 1000

                    event = NodeExit(
                        trace_id=self._adapter.trace_id,
                        span_id=span_id,
                        timestamp=now,
                        node_name=node_name,
                        duration_ms=duration_ms,
                    )
                    self._adapter._emit(event)
                    self._last_exited_node = node_name

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
                prompts = pending.get("prompts", [])

                now = datetime.now(timezone.utc)
                duration_ms = None
                if start_time is not None:
                    duration_ms = (now - start_time).total_seconds() * 1000

                # Extract token usage
                input_tokens = None
                output_tokens = None
                if response.llm_output:
                    usage = response.llm_output.get("token_usage", {})
                    input_tokens = usage.get("prompt_tokens")
                    output_tokens = usage.get("completion_tokens")

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

                # Build input messages
                input_messages = [{"role": "user", "content": p} for p in prompts]

                prompt_summary = prompts[0][:200] if prompts else None

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
