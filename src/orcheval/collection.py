"""Cross-run trace aggregation and comparison.

Holds multiple traces and computes aggregate statistics, outlier detection,
execution shape clustering, and trend analysis.
"""

from __future__ import annotations

import logging
import statistics as stats_mod
from collections import defaultdict
from datetime import datetime  # noqa: TC003
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from orcheval.events import ErrorEvent
from orcheval.trace import Trace

_log = logging.getLogger(__name__)


class PercentileStats(BaseModel):
    """Descriptive statistics for a set of numeric values."""

    model_config = {"frozen": True}

    mean: float
    median: float
    p95: float
    min: float
    max: float
    count: int


class NodeStats(BaseModel):
    """Aggregate statistics for a single node across traces."""

    model_config = {"frozen": True}

    node_name: str
    duration: PercentileStats | None = None
    cost: PercentileStats | None = None
    invocation_count: PercentileStats | None = None
    error_rate: float | None = None


class ExecutionShape(BaseModel):
    """A unique execution path shared by one or more traces."""

    model_config = {"frozen": True}

    node_sequence: list[str]
    trace_count: int
    trace_ids: list[str] = Field(default_factory=list)
    fraction: float


class TraceOutlier(BaseModel):
    """A trace flagged as an outlier for a specific metric.

    ``threshold_multiplier`` is the detection threshold used to find outliers,
    not the actual ratio of value to median. The actual ratio is in ``reason``.
    """

    model_config = {"frozen": True}

    trace_id: str
    metric: str
    value: float
    median: float
    threshold_multiplier: float
    reason: str


class TrendPoint(BaseModel):
    """A single data point in a trend analysis."""

    model_config = {"frozen": True}

    trace_id: str
    timestamp: datetime
    value: float


class TrendResult(BaseModel):
    """Result of trend analysis for a metric across traces."""

    model_config = {"frozen": True}

    metric: str
    direction: Literal["increasing", "decreasing", "stable", "insufficient_data"] = Field(
        description="Trend direction based on first-half vs second-half mean comparison."
    )
    points: list[TrendPoint] = Field(default_factory=list)
    change_pct: float | None = Field(
        default=None,
        description="Endpoint-to-endpoint percentage change (first trace value to last). "
        "May differ from direction, which uses half-split means.",
    )


class CollectionSummary(BaseModel):
    """Aggregate summary across all traces in a collection."""

    model_config = {"frozen": True}

    trace_count: int
    total_cost: PercentileStats | None = None
    total_duration: PercentileStats | None = None
    total_tokens: PercentileStats | None = None
    unique_node_names: list[str] = Field(default_factory=list)
    execution_shapes: list[ExecutionShape] = Field(default_factory=list)
    node_stats: list[NodeStats] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STABLE_THRESHOLD = 0.05  # 5% change = stable

_SUPPORTED_METRICS = frozenset({"cost", "duration", "errors", "tokens"})


def _percentile_stats(values: list[float]) -> PercentileStats | None:
    """Compute descriptive statistics from a list of floats."""
    if not values:
        return None
    s = sorted(values)
    p95_idx = int(len(s) * 0.95)
    if p95_idx >= len(s):
        p95_idx = len(s) - 1
    return PercentileStats(
        mean=stats_mod.mean(s),
        median=stats_mod.median(s),
        p95=s[p95_idx],
        min=s[0],
        max=s[-1],
        count=len(s),
    )


def _extract_metric(trace: Trace, metric: str) -> float | None:
    """Extract a single metric value from a trace."""
    if metric == "cost":
        return trace.total_cost()
    elif metric == "duration":
        return trace.total_duration()
    elif metric == "tokens":
        return float(trace.total_tokens()["total"]) if trace.total_tokens()["total"] > 0 else None
    elif metric == "errors":
        return float(len(trace.get_events_by_type(ErrorEvent)))
    else:
        raise ValueError(
            f"Unknown metric: {metric!r}. "
            f"Supported metrics: {', '.join(sorted(_SUPPORTED_METRICS))}"
        )


# ---------------------------------------------------------------------------
# TraceCollection
# ---------------------------------------------------------------------------


class TraceCollection:
    """Holds multiple traces and computes aggregate statistics.

    Usage::

        collection = TraceCollection(traces=[t1, t2, t3])
        collection = TraceCollection.from_json_dir("./traces/")

        summary = collection.summary()
        outliers = collection.find_outliers(metric="cost", threshold=3.0)
        trend = collection.trend(metric="duration")
    """

    __slots__ = ("_traces",)

    def __init__(self, traces: list[Trace]) -> None:
        self._traces = list(traces)

    @classmethod
    def from_json_dir(cls, path: str | Path) -> TraceCollection:
        """Load all ``.json`` files from a directory (non-recursive).

        Files that fail to parse as traces are silently skipped.
        Expected format: one trace per file as produced by ``trace.to_json()``.
        """
        directory = Path(path)
        traces: list[Trace] = []
        for fp in sorted(directory.glob("*.json")):
            try:
                text = fp.read_text(encoding="utf-8")
                traces.append(Trace.from_json(text))
            except Exception as e:  # noqa: BLE001
                _log.warning("Skipping %s: %s", fp.name, e)
                continue
        return cls(traces)

    @classmethod
    def from_traces(cls, *traces: Trace) -> TraceCollection:
        """Construct from explicit Trace objects."""
        return cls(list(traces))

    @property
    def traces(self) -> list[Trace]:
        """Return a copy of the traces list."""
        return list(self._traces)

    def __len__(self) -> int:
        return len(self._traces)

    def __bool__(self) -> bool:
        return len(self._traces) > 0

    # --- Analysis methods ---

    def summary(self) -> CollectionSummary:
        """Compute aggregate statistics across all traces."""
        if not self._traces:
            return CollectionSummary(trace_count=0)

        costs = [v for t in self._traces if (v := t.total_cost()) is not None]
        durations = [v for t in self._traces if (v := t.total_duration()) is not None]
        tokens = [
            float(t.total_tokens()["total"])
            for t in self._traces
            if t.total_tokens()["total"] > 0
        ]

        all_nodes: set[str] = set()
        for t in self._traces:
            all_nodes.update(t.node_sequence())

        shapes = self.execution_shapes()
        node_stats_list = [self.node_stats(n) for n in sorted(all_nodes)]

        return CollectionSummary(
            trace_count=len(self._traces),
            total_cost=_percentile_stats(costs),
            total_duration=_percentile_stats(durations),
            total_tokens=_percentile_stats(tokens),
            unique_node_names=sorted(all_nodes),
            execution_shapes=shapes,
            node_stats=node_stats_list,
        )

    def node_stats(self, node_name: str) -> NodeStats:
        """Compute per-node statistics across all traces."""
        durations: list[float] = []
        costs: list[float] = []
        inv_counts: list[float] = []
        traces_with_errors = 0

        for t in self._traces:
            dur = t.node_durations().get(node_name)
            if dur is not None:
                durations.append(dur)

            # Sum LLM costs for this node
            node_cost = sum(
                c.cost
                for c in t.get_llm_calls()
                if c.node_name == node_name and c.cost is not None
            )
            if node_cost > 0:
                costs.append(node_cost)

            count = sum(
                1 for inv in t.node_invocations() if inv.node_name == node_name
            )
            inv_counts.append(float(count))

            node_events = t.get_events_by_node(node_name)
            if any(isinstance(e, ErrorEvent) for e in node_events):
                traces_with_errors += 1

        # Only compute error_rate if the node appeared in at least one trace
        traces_with_node = sum(1 for c in inv_counts if c > 0)
        error_rate = (
            traces_with_errors / traces_with_node
            if traces_with_node > 0
            else None
        )

        return NodeStats(
            node_name=node_name,
            duration=_percentile_stats(durations),
            cost=_percentile_stats(costs),
            invocation_count=_percentile_stats(inv_counts),
            error_rate=error_rate,
        )

    def find_outliers(
        self,
        metric: str = "cost",
        threshold: float = 3.0,
    ) -> list[TraceOutlier]:
        """Find traces where a metric exceeds ``threshold`` times the median.

        Checks both high outliers (``value > threshold * median``) and low
        outliers (``value < median / threshold``). Low outliers (suspiciously
        cheap/fast) may indicate failures.

        Supported metrics: ``"cost"``, ``"duration"``, ``"errors"``, ``"tokens"``.
        """
        if metric not in _SUPPORTED_METRICS:
            raise ValueError(
                f"Unknown metric: {metric!r}. "
                f"Supported metrics: {', '.join(sorted(_SUPPORTED_METRICS))}"
            )

        # Collect (trace_id, value) pairs
        data: list[tuple[str, float]] = []
        for t in self._traces:
            val = _extract_metric(t, metric)
            if val is not None:
                data.append((t.trace_id, val))

        if not data:
            return []

        values = [v for _, v in data]
        median = stats_mod.median(values)

        if median == 0:
            # Can't detect outliers relative to zero median
            return []

        outliers: list[TraceOutlier] = []
        for trace_id, value in data:
            ratio = value / median
            if value > threshold * median:
                outliers.append(TraceOutlier(
                    trace_id=trace_id,
                    metric=metric,
                    value=value,
                    median=median,
                    threshold_multiplier=threshold,
                    reason=f"{metric} {value:.4g} is {ratio:.1f}x the median {median:.4g}",
                ))
            elif value < median / threshold:
                outliers.append(TraceOutlier(
                    trace_id=trace_id,
                    metric=metric,
                    value=value,
                    median=median,
                    threshold_multiplier=threshold,
                    reason=f"{metric} {value:.4g} is {ratio:.2f}x the median {median:.4g}",
                ))

        return outliers

    def execution_shapes(self) -> list[ExecutionShape]:
        """Cluster traces by their node execution sequence."""
        groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for t in self._traces:
            key = tuple(t.node_sequence())
            groups[key].append(t.trace_id)

        total = len(self._traces) if self._traces else 1
        shapes = [
            ExecutionShape(
                node_sequence=list(seq),
                trace_count=len(ids),
                trace_ids=ids,
                fraction=len(ids) / total,
            )
            for seq, ids in groups.items()
        ]
        # Sort by descending trace count
        shapes.sort(key=lambda s: s.trace_count, reverse=True)
        return shapes

    def trend(self, metric: str = "cost") -> TrendResult:
        """Detect trends in a metric across chronologically ordered traces.

        Sorts traces by their first event's timestamp, then compares
        the first-half mean to the second-half mean. Direction is
        classified as:

        - ``"stable"``: < 5% change between halves
        - ``"increasing"``: second half >= 5% higher
        - ``"decreasing"``: second half >= 5% lower
        - ``"insufficient_data"``: fewer than 3 data points
        """
        if metric not in _SUPPORTED_METRICS:
            raise ValueError(
                f"Unknown metric: {metric!r}. "
                f"Supported metrics: {', '.join(sorted(_SUPPORTED_METRICS))}"
            )

        # Collect (timestamp, trace_id, value)
        raw: list[tuple[datetime, str, float]] = []
        for t in self._traces:
            if not t.events:
                continue
            val = _extract_metric(t, metric)
            if val is not None:
                raw.append((t.events[0].timestamp, t.trace_id, val))

        # Sort by timestamp
        raw.sort(key=lambda x: x[0])

        points = [
            TrendPoint(trace_id=tid, timestamp=ts, value=v)
            for ts, tid, v in raw
        ]

        if len(points) < 3:
            return TrendResult(
                metric=metric,
                direction="insufficient_data",
                points=points,
                change_pct=None,
            )

        # Split into halves: middle trace goes to second half for odd counts
        mid = len(points) // 2
        first_half = [p.value for p in points[:mid]]
        second_half = [p.value for p in points[mid:]]

        first_mean = stats_mod.mean(first_half)
        second_mean = stats_mod.mean(second_half)

        # Compute change percentage
        if first_mean == 0:
            pct_change = 0.0 if second_mean == 0 else float("inf")
        else:
            pct_change = (second_mean - first_mean) / abs(first_mean)

        if abs(pct_change) < _STABLE_THRESHOLD:
            direction: Literal["increasing", "decreasing", "stable"] = "stable"
        elif pct_change > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        # First-to-last percentage change
        first_val = points[0].value
        last_val = points[-1].value
        if first_val == 0:
            total_change_pct = None
        else:
            total_change_pct = round(((last_val - first_val) / abs(first_val)) * 100, 1)

        return TrendResult(
            metric=metric,
            direction=direction,
            points=points,
            change_pct=total_change_pct,
        )
