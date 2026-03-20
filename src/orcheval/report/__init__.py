"""Report generation for orchestration traces.

Provides structured analysis of collected trace data: cost breakdowns,
routing audits, convergence tracking, timeline views, and retry analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from orcheval.report.convergence import ConvergenceReport, PassSummary, convergence_report
from orcheval.report.cost import CostReport, ModelUsage, NodeCostSummary, cost_report
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
    )


__all__ = [
    # Unified
    "FullReport",
    "report",
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
    "PassSummary",
    "convergence_report",
    # Retries
    "RetryReport",
    "ErrorCluster",
    "RetrySequence",
    "retry_report",
]
