"""Manual adapter for emitting events directly."""

from __future__ import annotations

from typing import Any, Literal

from orcheval.adapters.base import BaseAdapter
from orcheval.events import (
    AgentMessage,
    ErrorEvent,
    Event,
    LLMCall,
    NodeEntry,
    NodeExit,
    PassBoundary,
    RoutingDecision,
    ToolCall,
)


class ManualAdapter(BaseAdapter):
    """Adapter for manually emitting orchestration events.

    Use this as an escape hatch when your framework doesn't have a supported
    adapter, or for testing.

    Usage::

        adapter = ManualAdapter(trace_id="my-trace")
        adapter.node_entry("agent_1")
        adapter.llm_call(node_name="agent_1", model="gpt-4o",
                         input_tokens=100, output_tokens=50, cost=0.003)
        adapter.node_exit("agent_1", duration_ms=1500.0)
        events = adapter.get_events()
    """

    def get_callback_handler(self) -> ManualAdapter:
        """Returns self — the manual adapter is its own handler."""
        return self

    def emit(self, event_type: type[Event], **kwargs: Any) -> Event:
        """Create and record an event of the given type.

        The trace_id is automatically set. All other fields are passed through.
        """
        event = event_type(trace_id=self._trace_id, **kwargs)
        self._emit(event)
        return event

    # --- Convenience methods ---

    def node_entry(self, node_name: str, **kwargs: Any) -> NodeEntry:
        event = NodeEntry(trace_id=self._trace_id, node_name=node_name, **kwargs)
        self._emit(event)
        return event

    def node_exit(self, node_name: str, **kwargs: Any) -> NodeExit:
        event = NodeExit(trace_id=self._trace_id, node_name=node_name, **kwargs)
        self._emit(event)
        return event

    def llm_call(self, node_name: str | None = None, **kwargs: Any) -> LLMCall:
        event = LLMCall(trace_id=self._trace_id, node_name=node_name, **kwargs)
        self._emit(event)
        return event

    def tool_call(self, tool_name: str, **kwargs: Any) -> ToolCall:
        event = ToolCall(trace_id=self._trace_id, tool_name=tool_name, **kwargs)
        self._emit(event)
        return event

    def error(self, error_type: str, error_message: str, **kwargs: Any) -> ErrorEvent:
        event = ErrorEvent(
            trace_id=self._trace_id,
            error_type=error_type,
            error_message=error_message,
            **kwargs,
        )
        self._emit(event)
        return event

    def routing_decision(
        self, source_node: str, target_node: str, **kwargs: Any
    ) -> RoutingDecision:
        event = RoutingDecision(
            trace_id=self._trace_id,
            source_node=source_node,
            target_node=target_node,
            **kwargs,
        )
        self._emit(event)
        return event

    def agent_message(self, sender: str, receiver: str, **kwargs: Any) -> AgentMessage:
        event = AgentMessage(
            trace_id=self._trace_id,
            sender=sender,
            receiver=receiver,
            **kwargs,
        )
        self._emit(event)
        return event

    def pass_boundary(
        self, pass_number: int, direction: Literal["enter", "exit"], **kwargs: Any
    ) -> PassBoundary:
        event = PassBoundary(
            trace_id=self._trace_id,
            pass_number=pass_number,
            direction=direction,
            **kwargs,
        )
        self._emit(event)
        return event
