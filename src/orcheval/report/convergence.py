"""Multi-pass convergence analysis from PassBoundary events."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from orcheval.events import PassBoundary

if TYPE_CHECKING:
    from orcheval.trace import Trace


MetricStatus = Literal["converging", "diverging", "oscillating", "plateaued"]

# A metric is "plateaued" when its last N deltas are all below this threshold
_PLATEAU_EPSILON = 1e-9
# Minimum direction changes to classify as oscillating
_MIN_OSCILLATION_CHANGES = 3


class MetricConvergence(BaseModel):
    """Per-metric convergence classification."""

    model_config = {"frozen": True}

    metric_name: str
    status: MetricStatus
    values: list[float] = Field(default_factory=list)
    abs_deltas: list[float] = Field(default_factory=list)


class PassSummary(BaseModel):
    """Summary of a single processing pass."""

    model_config = {"frozen": True}

    pass_number: int
    metrics_enter: dict[str, Any] = Field(default_factory=dict)
    metrics_exit: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float | None = None
    metric_deltas: dict[str, float] = Field(default_factory=dict)


class ConvergenceReport(BaseModel):
    """Multi-pass convergence analysis.

    .. note::

       Neither the LangGraph nor OpenAI Agents adapter emits ``PassBoundary``
       events automatically.  This report will be empty unless pass boundaries
       are recorded manually via ``ManualAdapter.pass_boundary()``.
    """

    model_config = {"frozen": True}

    passes: list[PassSummary] = Field(default_factory=list)
    total_passes: int = 0
    is_converging: bool | None = None
    per_metric: list[MetricConvergence] = Field(default_factory=list)
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


def _classify_metric(values: list[float]) -> MetricStatus:
    """Classify a single metric's convergence behaviour.

    - **plateaued**: last two abs-deltas are both below epsilon.
    - **oscillating**: the sign of consecutive deltas changes ≥ 3 times
      AND there is no net improvement (abs-delta not shrinking).
    - **converging**: abs-deltas are monotonically non-increasing.
    - **diverging**: none of the above (abs-deltas increasing).
    """
    abs_deltas = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]

    # --- plateau check: last two deltas effectively zero ---
    if len(abs_deltas) >= 2 and all(d <= _PLATEAU_EPSILON for d in abs_deltas[-2:]):
        return "plateaued"

    # --- oscillation check: direction flips ≥ 3 times ---
    raw_deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
    direction_changes = 0
    for i in range(1, len(raw_deltas)):
        if raw_deltas[i] * raw_deltas[i - 1] < 0:
            direction_changes += 1
    if direction_changes >= _MIN_OSCILLATION_CHANGES:
        return "oscillating"

    # --- converging check: abs-deltas monotonically non-increasing ---
    is_non_increasing = all(
        abs_deltas[i] <= abs_deltas[i - 1] for i in range(1, len(abs_deltas))
    )
    if is_non_increasing:
        return "converging"

    return "diverging"


def _check_converging(
    metric_trends: dict[str, list[float]],
) -> tuple[bool | None, list[MetricConvergence]]:
    """Determine per-metric and overall convergence.

    Returns a tuple of (is_converging, per_metric_list).
    is_converging is True when ALL metrics are converging or plateaued,
    False if any are diverging or oscillating, None if insufficient data.
    """
    if not metric_trends:
        return None, []

    # Need at least 3 exit values (3 passes) to classify
    usable = {k: v for k, v in metric_trends.items() if len(v) >= 3}
    if not usable:
        return None, []

    per_metric: list[MetricConvergence] = []
    for name, values in sorted(usable.items()):
        abs_deltas = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
        status = _classify_metric(values)
        per_metric.append(MetricConvergence(
            metric_name=name,
            status=status,
            values=values,
            abs_deltas=abs_deltas,
        ))

    all_ok = all(m.status in ("converging", "plateaued") for m in per_metric)
    return all_ok, per_metric


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

    is_converging, per_metric = _check_converging(dict(metric_trends))

    return ConvergenceReport(
        passes=passes,
        total_passes=len(passes),
        is_converging=is_converging,
        per_metric=per_metric,
        metric_trends=dict(metric_trends),
        final_metrics=final_metrics,
    )
