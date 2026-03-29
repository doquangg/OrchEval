"""Report generation for orchestration traces.

Provides structured analysis of collected trace data: cost breakdowns,
routing audits, convergence tracking, timeline views, and retry analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from orcheval._io import write_output
from orcheval.report.comparison import (
    ConvergenceDiff,
    CostDelta,
    DurationDelta,
    ErrorDiff,
    InvocationDelta,
    PatternDiff,
    RoutingDiff,
    RunComparison,
    compare_runs,
)
from orcheval.report.convergence import (
    ConvergenceReport,
    MetricConvergence,
    PassSummary,
    convergence_report,
)
from orcheval.report.cost import CostReport, ModelUsage, NodeCostSummary, cost_report
from orcheval.report.llm_patterns import (
    LLMPattern,
    LLMPatternsReport,
    NodeLLMSummary,
    llm_patterns_report,
)
from orcheval.report.retries import ErrorCluster, RetryReport, RetrySequence, retry_report
from orcheval.report.routing import RoutingEdge, RoutingFlag, RoutingReport, routing_report
from orcheval.report.timeline import TimelineEvent, TimelineReport, TimelineSpan, timeline_report

if TYPE_CHECKING:
    from orcheval.trace import Trace


class FullReport(BaseModel):
    """Aggregation of all report modules."""

    model_config = {"frozen": True}

    cost: CostReport
    routing: RoutingReport
    convergence: ConvergenceReport
    timeline: TimelineReport
    retries: RetryReport
    llm_patterns: LLMPatternsReport

    def to_json(self, path: str | None = None) -> str:
        """Serialize the report to a JSON string.

        If *path* is given, also writes the JSON to that file, routing bare
        filenames through the default output directory (``orcheval_outputs/``).
        """
        result = self.model_dump_json()
        if path is not None:
            write_output(result, path)
        return result

    @classmethod
    def from_json_file(cls, path: str | Path) -> FullReport:
        """Load a report from a JSON file."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def report(trace: Trace) -> FullReport:
    """Generate a complete report from a trace.

    Runs all analysis modules and returns a unified result.
    For targeted analysis, use the individual module functions directly::

        from orcheval.report import cost_report
        result = cost_report(trace)
    """
    return FullReport(
        cost=cost_report(trace),
        routing=routing_report(trace),
        convergence=convergence_report(trace),
        timeline=timeline_report(trace),
        retries=retry_report(trace),
        llm_patterns=llm_patterns_report(trace),
    )


__all__ = [
    # Unified
    "FullReport",
    "report",
    # Comparison
    "RunComparison",
    "CostDelta",
    "DurationDelta",
    "RoutingDiff",
    "InvocationDelta",
    "ErrorDiff",
    "ConvergenceDiff",
    "PatternDiff",
    "compare_runs",
    # Cost
    "CostReport",
    "NodeCostSummary",
    "ModelUsage",
    "cost_report",
    # Timeline
    "TimelineReport",
    "TimelineSpan",
    "TimelineEvent",
    "timeline_report",
    # Routing
    "RoutingReport",
    "RoutingEdge",
    "RoutingFlag",
    "routing_report",
    # Convergence
    "ConvergenceReport",
    "MetricConvergence",
    "PassSummary",
    "convergence_report",
    # Retries
    "RetryReport",
    "ErrorCluster",
    "RetrySequence",
    "retry_report",
    # LLM Patterns
    "LLMPatternsReport",
    "LLMPattern",
    "NodeLLMSummary",
    "llm_patterns_report",
]
