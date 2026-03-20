"""orcheval — Evaluations for multi-agent and swarm-based orchestrations."""

from __future__ import annotations

import uuid
from typing import Any

from orcheval.adapters.base import BaseAdapter
from orcheval.adapters.manual import ManualAdapter
from orcheval.events import (
    AgentMessage,
    AnyEvent,
    ErrorEvent,
    Event,
    LLMCall,
    NodeEntry,
    NodeExit,
    PassBoundary,
    RoutingDecision,
    ToolCall,
)
from orcheval.report import FullReport, report
from orcheval.trace import NodeInvocation, Trace

__version__ = "0.1.0"


class Tracer:
    """Unified entry point for tracing orchestration runs.

    Usage::

        from orcheval import Tracer

        tracer = Tracer(adapter="langgraph")
        result = graph.invoke(input, config={"callbacks": [tracer.handler]})
        trace = tracer.collect()

        # Or with manual adapter (default):
        tracer = Tracer()
        tracer.adapter.node_entry("my_node")
        tracer.adapter.llm_call(model="gpt-4o", input_tokens=100)
        tracer.adapter.node_exit("my_node", duration_ms=1500.0)
        trace = tracer.collect()
    """

    def __init__(
        self,
        adapter: str | BaseAdapter = "manual",
        trace_id: str | None = None,
        *,
        infer_routing: bool = False,
    ) -> None:
        self._trace_id = trace_id or uuid.uuid4().hex
        self._infer_routing = infer_routing

        if isinstance(adapter, str):
            self._adapter = self._resolve_adapter(adapter)
        elif isinstance(adapter, BaseAdapter):
            self._adapter = adapter
        else:
            raise TypeError(
                f"adapter must be a string or BaseAdapter instance, got {type(adapter).__name__}"
            )

    def _resolve_adapter(self, name: str) -> BaseAdapter:
        if name == "langgraph":
            from orcheval.adapters.langgraph import LangGraphAdapter

            return LangGraphAdapter(self._trace_id, infer_routing=self._infer_routing)
        elif name == "manual":
            return ManualAdapter(self._trace_id)
        else:
            raise ValueError(
                f"Unknown adapter: {name!r}. Use 'langgraph', 'manual', "
                f"or pass a BaseAdapter instance."
            )

    @property
    def adapter(self) -> BaseAdapter:
        """The underlying adapter instance."""
        return self._adapter

    @property
    def handler(self) -> Any:
        """Framework-specific callback handler, ready to pass to the framework."""
        return self._adapter.get_callback_handler()

    @property
    def trace_id(self) -> str:
        """The trace ID for this tracing session."""
        return self._trace_id

    def collect(self) -> Trace:
        """Collect all recorded events into a Trace."""
        return Trace(events=self._adapter.get_events(), trace_id=self._trace_id)

    def reset(self) -> None:
        """Clear collected events for reuse."""
        self._adapter.reset()


__all__ = [
    # Core
    "Tracer",
    "Trace",
    "NodeInvocation",
    # Events
    "Event",
    "AnyEvent",
    "NodeEntry",
    "NodeExit",
    "LLMCall",
    "ToolCall",
    "RoutingDecision",
    "AgentMessage",
    "ErrorEvent",
    "PassBoundary",
    # Adapters
    "BaseAdapter",
    "ManualAdapter",
    # Report
    "FullReport",
    "report",
]
