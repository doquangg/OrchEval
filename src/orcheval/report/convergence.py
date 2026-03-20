"""Multi-pass convergence analysis from PassBoundary events."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from orcheval.events import PassBoundary

if TYPE_CHECKING:
    from orcheval.trace import Trace


class PassSummary(BaseModel):
    """Summary of a single processing pass."""

    model_config = {"frozen": True}

    pass_number: int
    metrics_enter: dict[str, Any] = Field(default_factory=dict)
    metrics_exit: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float | None = None
    metric_deltas: dict[str, float] = Field(default_factory=dict)


class ConvergenceReport(BaseModel):
    """Multi-pass convergence analysis."""

    model_config = {"frozen": True}

    passes: list[PassSummary] = Field(default_factory=list)
    total_passes: int = 0
    is_converging: bool | None = None
    metric_trends: dict[str, list[float]] = Field(default_factory=dict)
    final_metrics: dict[str, Any] = Field(default_factory=dict)


def _numeric_deltas(
    enter: dict[str, Any], exit_: dict[str, Any]
) -> dict[str, float]:
    """Compute exit - enter for keys that are numeric in both snapshots."""
    deltas: dict[str, float] = {}
    for key in enter:
        if key in exit_:
            e_val, x_val = enter[key], exit_[key]
            if isinstance(e_val, (int, float)) and isinstance(x_val, (int, float)):
                deltas[key] = float(x_val) - float(e_val)
    return deltas


def _check_converging(metric_trends: dict[str, list[float]]) -> bool | None:
    """Determine if metrics are converging across passes.

    Returns True if absolute deltas between consecutive exit values
    are monotonically decreasing for all tracked metrics. False if any
    diverge. None if insufficient data (<2 passes).
    """
    if not metric_trends:
        return None

    # Need at least 3 exit values (3 passes) to check if deltas decrease
    # With 2 passes we have only 1 delta — not enough to detect a trend
    usable = {k: v for k, v in metric_trends.items() if len(v) >= 3}
    if not usable:
        # If we have exactly 2 passes, we can't determine convergence
        any_has_two = any(len(v) >= 2 for v in metric_trends.values())
        return None if not any_has_two else None

    all_converging = True
    for values in usable.values():
        abs_deltas = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
        # Check if deltas are non-increasing (allows equal consecutive deltas)
        for i in range(1, len(abs_deltas)):
            if abs_deltas[i] > abs_deltas[i - 1]:
                all_converging = False
                break
        if not all_converging:
            break

    return all_converging


def convergence_report(trace: Trace) -> ConvergenceReport:
    """Analyze multi-pass convergence from PassBoundary events."""
    boundaries = trace.get_events_by_type(PassBoundary)
    if not boundaries:
        return ConvergenceReport()

    # Group by pass_number
    pass_events: dict[int, dict[str, PassBoundary]] = defaultdict(dict)
    for b in boundaries:
        pass_events[b.pass_number][b.direction] = b

    passes: list[PassSummary] = []
    for pass_num in sorted(pass_events):
        directions = pass_events[pass_num]
        enter_event = directions.get("enter")
        exit_event = directions.get("exit")

        metrics_enter = enter_event.metrics_snapshot if enter_event else {}
        metrics_exit = exit_event.metrics_snapshot if exit_event else {}

        # Duration from enter to exit timestamps
        duration: float | None = None
        if enter_event and exit_event:
            delta = exit_event.timestamp - enter_event.timestamp
            duration = delta.total_seconds() * 1000

        deltas = _numeric_deltas(metrics_enter, metrics_exit)

        passes.append(PassSummary(
            pass_number=pass_num,
            metrics_enter=metrics_enter,
            metrics_exit=metrics_exit,
            duration_ms=duration,
            metric_deltas=deltas,
        ))

    # Build metric trends from exit snapshots across passes
    metric_trends: dict[str, list[float]] = defaultdict(list)
    for p in passes:
        for key, val in p.metrics_exit.items():
            if isinstance(val, (int, float)):
                metric_trends[key].append(float(val))

    # Final metrics from last exit
    final_metrics: dict[str, Any] = {}
    for p in reversed(passes):
        if p.metrics_exit:
            final_metrics = p.metrics_exit
            break

    return ConvergenceReport(
        passes=passes,
        total_passes=len(passes),
        is_converging=_check_converging(dict(metric_trends)),
        metric_trends=dict(metric_trends),
        final_metrics=final_metrics,
    )
