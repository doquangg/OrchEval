"""orcheval — Evaluations for multi-agent and swarm-based orchestrations."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from orcheval._io import DEFAULT_OUTPUT_DIR
from orcheval.adapters.base import BaseAdapter
from orcheval.adapters.manual import ManualAdapter
from orcheval.collection import (
    CollectionSummary,
    ExecutionShape,
    NodeStats,
    PercentileStats,
    TraceCollection,
    TraceOutlier,
    TrendPoint,
    TrendResult,
)
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
from orcheval.report import FullReport, RunComparison, compare_runs, report
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

        # Auto-save all artifacts to orcheval_outputs/:
        tracer = Tracer(adapter="langgraph", save_artifacts=True)
        result = graph.invoke(input, config={"callbacks": [tracer.handler]})
        trace = tracer.collect()
        # Writes orcheval_outputs/trace.json, report.json, trace.html
    """

    def __init__(
        self,
        adapter: str | BaseAdapter = "manual",
        trace_id: str | None = None,
        *,
        infer_routing: bool = False,
        capture_state: bool = False,
        save_artifacts: bool = False,
    ) -> None:
        self._trace_id = trace_id or uuid.uuid4().hex
        self._infer_routing = infer_routing
        self._capture_state = capture_state
        self._save_artifacts = save_artifacts

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

            return LangGraphAdapter(
                self._trace_id,
                infer_routing=self._infer_routing,
                capture_state=self._capture_state,
            )
        elif name == "openai_agents":
            from orcheval.adapters.openai_agents import OpenAIAgentsAdapter

            return OpenAIAgentsAdapter(
                self._trace_id,
                infer_routing=self._infer_routing,
                capture_state=self._capture_state,
            )
        elif name == "manual":
            return ManualAdapter(self._trace_id)
        else:
            raise ValueError(
                f"Unknown adapter: {name!r}. Use 'langgraph', 'openai_agents', "
                f"'manual', or pass a BaseAdapter instance."
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
        """Collect all recorded events into a Trace.

        If *save_artifacts* was set on the tracer, also writes
        ``orcheval_outputs/trace.json``, ``orcheval_outputs/report.json``,
        and ``orcheval_outputs/trace.html``.
        """
        trace = Trace(events=self._adapter.get_events(), trace_id=self._trace_id)
        if self._save_artifacts:
            full = report(trace)
            trace.to_json("trace.json")
            full.to_json("report.json")
            trace.to_html("trace.html", reports=full)
        return trace

    def reset(self) -> None:
        """Clear collected events for reuse."""
        self._adapter.reset()


def html_from_files(
    trace_path: str | Path,
    report_path: str | Path | None = None,
    output_path: str = "trace.html",
) -> str:
    """Generate HTML visualization from saved JSON files.

    Args:
        trace_path: Path to a trace JSON file (from trace.to_json()).
        report_path: Optional path to a report JSON file. If not provided,
            the report is recomputed from the trace.
        output_path: Where to write the HTML file.

    Returns:
        The HTML string.
    """
    trace = Trace.from_json_file(trace_path)
    reports = None
    if report_path is not None:
        reports = FullReport.from_json_file(report_path)
    return trace.to_html(output_path, reports=reports)


__all__ = [
    # Core
    "Tracer",
    "Trace",
    "NodeInvocation",
    "DEFAULT_OUTPUT_DIR",
    "html_from_files",
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
    "RunComparison",
    "compare_runs",
    # Collection
    "TraceCollection",
    "CollectionSummary",
    "NodeStats",
    "PercentileStats",
    "ExecutionShape",
    "TraceOutlier",
    "TrendPoint",
    "TrendResult",
]
