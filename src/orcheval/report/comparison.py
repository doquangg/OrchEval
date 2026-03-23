"""Run-to-run comparison across all analysis dimensions."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from orcheval.events import ErrorEvent

if TYPE_CHECKING:
    from orcheval.report.convergence import ConvergenceReport
    from orcheval.report.cost import CostReport
    from orcheval.report.llm_patterns import LLMPatternsReport
    from orcheval.report.routing import RoutingReport
    from orcheval.trace import Trace


# ---------------------------------------------------------------------------
# Leaf delta models
# ---------------------------------------------------------------------------


class CostDelta(BaseModel):
    """Cost change for a single node, model, or the total."""

    model_config = {"frozen": True}

    name: str
    baseline: float | None = None
    experiment: float | None = None
    delta: float | None = None
    pct_change: float | None = None
    baseline_tokens: int = 0
    experiment_tokens: int = 0


class DurationDelta(BaseModel):
    """Duration change for a single node or the total trace."""

    model_config = {"frozen": True}

    node_name: str
    baseline_ms: float | None = None
    experiment_ms: float | None = None
    delta_ms: float | None = None
    pct_change: float | None = None
    flagged: bool = False


class RoutingDiff(BaseModel):
    """A routing edge that differs between runs."""

    model_config = {"frozen": True}

    source_node: str
    target_node: str
    status: Literal["added", "removed", "changed"]
    baseline_count: int = 0
    experiment_count: int = 0


class InvocationDelta(BaseModel):
    """Node invocation count change."""

    model_config = {"frozen": True}

    node_name: str
    baseline_count: int = 0
    experiment_count: int = 0
    delta: int = 0


class ErrorDiff(BaseModel):
    """An error that appeared, disappeared, or changed in frequency."""

    model_config = {"frozen": True}

    error_type: str
    node_name: str | None = None
    status: Literal["new", "resolved", "count_changed"]
    baseline_count: int = 0
    experiment_count: int = 0
    message_sample: str | None = None


class ConvergenceDiff(BaseModel):
    """Differences in convergence behaviour."""

    model_config = {"frozen": True}

    baseline_is_converging: bool | None = None
    experiment_is_converging: bool | None = None
    baseline_total_passes: int = 0
    experiment_total_passes: int = 0
    metric_diffs: dict[str, dict[str, Any]] = Field(default_factory=dict)


class PatternDiff(BaseModel):
    """An LLM pattern that appeared or disappeared between runs."""

    model_config = {"frozen": True}

    pattern_type: str
    node_name: str
    status: Literal["new", "resolved"]
    severity: str
    description: str


# ---------------------------------------------------------------------------
# Grouping models
# ---------------------------------------------------------------------------


class CostComparison(BaseModel):
    """Cost dimension comparison."""

    model_config = {"frozen": True}

    total_delta: CostDelta | None = None
    node_deltas: list[CostDelta] = Field(default_factory=list)
    model_deltas: list[CostDelta] = Field(default_factory=list)


class DurationComparison(BaseModel):
    """Duration dimension comparison."""

    model_config = {"frozen": True}

    total_delta: DurationDelta | None = None
    node_deltas: list[DurationDelta] = Field(default_factory=list)


class RoutingComparison(BaseModel):
    """Routing dimension comparison."""

    model_config = {"frozen": True}

    edges_added: list[RoutingDiff] = Field(default_factory=list)
    edges_removed: list[RoutingDiff] = Field(default_factory=list)
    edges_changed: list[RoutingDiff] = Field(default_factory=list)


class InvocationComparison(BaseModel):
    """Node invocation count comparison."""

    model_config = {"frozen": True}

    changes: list[InvocationDelta] = Field(default_factory=list)


class ErrorComparison(BaseModel):
    """Error dimension comparison."""

    model_config = {"frozen": True}

    new_errors: list[ErrorDiff] = Field(default_factory=list)
    resolved_errors: list[ErrorDiff] = Field(default_factory=list)
    count_changes: list[ErrorDiff] = Field(default_factory=list)


class PatternComparison(BaseModel):
    """LLM pattern dimension comparison."""

    model_config = {"frozen": True}

    new_patterns: list[PatternDiff] = Field(default_factory=list)
    resolved_patterns: list[PatternDiff] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------


class RunComparison(BaseModel):
    """Complete comparison between a baseline and experiment trace."""

    model_config = {"frozen": True}

    cost: CostComparison
    duration: DurationComparison
    routing: RoutingComparison
    invocations: InvocationComparison
    errors: ErrorComparison
    convergence: ConvergenceDiff | None = None
    llm_patterns: PatternComparison
    summary: str


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _pct_change(baseline: float | None, experiment: float | None) -> float | None:
    """Return percentage change, or None if not computable."""
    if baseline is None or experiment is None or baseline == 0:
        return None
    return ((experiment - baseline) / abs(baseline)) * 100


def _delta(baseline: float | None, experiment: float | None) -> float | None:
    if baseline is None or experiment is None:
        return None
    return experiment - baseline


def _cost_delta(
    name: str,
    b_cost: float | None,
    e_cost: float | None,
    b_tokens: int = 0,
    e_tokens: int = 0,
) -> CostDelta:
    return CostDelta(
        name=name,
        baseline=b_cost,
        experiment=e_cost,
        delta=_delta(b_cost, e_cost),
        pct_change=_pct_change(b_cost, e_cost),
        baseline_tokens=b_tokens,
        experiment_tokens=e_tokens,
    )


def _duration_delta(
    node_name: str,
    b_ms: float | None,
    e_ms: float | None,
    threshold: float,
) -> DurationDelta:
    pct = _pct_change(b_ms, e_ms)
    return DurationDelta(
        node_name=node_name,
        baseline_ms=b_ms,
        experiment_ms=e_ms,
        delta_ms=_delta(b_ms, e_ms),
        pct_change=pct,
        flagged=pct is not None and abs(pct) > threshold * 100,
    )


# ---------------------------------------------------------------------------
# Per-dimension comparison helpers
# ---------------------------------------------------------------------------


def _compare_costs(
    b_cost: CostReport,
    e_cost: CostReport,
) -> CostComparison:
    # Total
    b_total = b_cost.total_cost
    e_total = e_cost.total_cost
    b_tokens = b_cost.total_tokens
    e_tokens = e_cost.total_tokens
    total = None
    if b_total is not None or e_total is not None:
        total = _cost_delta(
            "__total__",
            b_total,
            e_total,
            b_tokens.get("total", 0),
            e_tokens.get("total", 0),
        )

    # Per-node
    b_nodes = {n.node_name: n for n in b_cost.nodes}
    e_nodes = {n.node_name: n for n in e_cost.nodes}
    all_node_names = sorted(set(b_nodes) | set(e_nodes))
    node_deltas: list[CostDelta] = []
    for name in all_node_names:
        bn = b_nodes.get(name)
        en = e_nodes.get(name)
        node_deltas.append(_cost_delta(
            name,
            bn.total_cost if bn else None,
            en.total_cost if en else None,
            bn.total_tokens if bn else 0,
            en.total_tokens if en else 0,
        ))

    # Per-model
    b_models = {m.model: m for m in b_cost.models}
    e_models = {m.model: m for m in e_cost.models}
    all_model_names = sorted(set(b_models) | set(e_models))
    model_deltas: list[CostDelta] = []
    for name in all_model_names:
        bm = b_models.get(name)
        em = e_models.get(name)
        model_deltas.append(_cost_delta(
            name,
            bm.total_cost if bm else None,
            em.total_cost if em else None,
            bm.total_tokens if bm else 0,
            em.total_tokens if em else 0,
        ))

    return CostComparison(
        total_delta=total,
        node_deltas=node_deltas,
        model_deltas=model_deltas,
    )


def _compare_durations(
    b_trace: Trace,
    e_trace: Trace,
    threshold: float,
) -> DurationComparison:
    b_total = b_trace.total_duration()
    e_total = e_trace.total_duration()
    total = None
    if b_total is not None or e_total is not None:
        total = _duration_delta("__total__", b_total, e_total, threshold)

    b_nodes = b_trace.node_durations()
    e_nodes = e_trace.node_durations()
    all_names = sorted(set(b_nodes) | set(e_nodes))
    node_deltas = [
        _duration_delta(
            name,
            b_nodes.get(name),
            e_nodes.get(name),
            threshold,
        )
        for name in all_names
    ]

    return DurationComparison(total_delta=total, node_deltas=node_deltas)


def _compare_routing(
    b_routing: RoutingReport,
    e_routing: RoutingReport,
) -> RoutingComparison:
    b_edges: dict[tuple[str, str], int] = {}
    for edge in b_routing.decisions:
        b_edges[(edge.source_node, edge.target_node)] = edge.count

    e_edges: dict[tuple[str, str], int] = {}
    for edge in e_routing.decisions:
        e_edges[(edge.source_node, edge.target_node)] = edge.count

    all_keys = set(b_edges) | set(e_edges)

    added: list[RoutingDiff] = []
    removed: list[RoutingDiff] = []
    changed: list[RoutingDiff] = []

    for src, tgt in sorted(all_keys):
        b_count = b_edges.get((src, tgt))
        e_count = e_edges.get((src, tgt))
        if b_count is None:
            added.append(RoutingDiff(
                source_node=src, target_node=tgt, status="added",
                baseline_count=0, experiment_count=e_count or 0,
            ))
        elif e_count is None:
            removed.append(RoutingDiff(
                source_node=src, target_node=tgt, status="removed",
                baseline_count=b_count, experiment_count=0,
            ))
        elif b_count != e_count:
            changed.append(RoutingDiff(
                source_node=src, target_node=tgt, status="changed",
                baseline_count=b_count, experiment_count=e_count,
            ))

    return RoutingComparison(
        edges_added=added, edges_removed=removed, edges_changed=changed,
    )


def _compare_invocations(b_trace: Trace, e_trace: Trace) -> InvocationComparison:
    b_counts: Counter[str] = Counter(inv.node_name for inv in b_trace.node_invocations())
    e_counts: Counter[str] = Counter(inv.node_name for inv in e_trace.node_invocations())
    all_names = sorted(set(b_counts) | set(e_counts))

    changes = [
        InvocationDelta(
            node_name=name,
            baseline_count=b_counts.get(name, 0),
            experiment_count=e_counts.get(name, 0),
            delta=e_counts.get(name, 0) - b_counts.get(name, 0),
        )
        for name in all_names
        if b_counts.get(name, 0) != e_counts.get(name, 0)
    ]

    return InvocationComparison(changes=changes)


def _compare_errors(b_trace: Trace, e_trace: Trace) -> ErrorComparison:
    def _build_error_map(
        trace: Trace,
    ) -> tuple[dict[tuple[str, str | None], int], dict[tuple[str, str | None], str]]:
        counts: dict[tuple[str, str | None], int] = {}
        messages: dict[tuple[str, str | None], str] = {}
        for err in trace.get_events_by_type(ErrorEvent):
            key = (err.error_type, err.node_name)
            counts[key] = counts.get(key, 0) + 1
            if key not in messages:
                messages[key] = err.error_message
        return counts, messages

    b_counts, b_msgs = _build_error_map(b_trace)
    e_counts, e_msgs = _build_error_map(e_trace)

    all_keys = set(b_counts) | set(e_counts)

    new_errors: list[ErrorDiff] = []
    resolved_errors: list[ErrorDiff] = []
    count_changes: list[ErrorDiff] = []

    for error_type, node_name in sorted(all_keys, key=lambda k: (k[0], k[1] or "")):
        bc = b_counts.get((error_type, node_name), 0)
        ec = e_counts.get((error_type, node_name), 0)
        msg = e_msgs.get((error_type, node_name)) or b_msgs.get((error_type, node_name))

        if bc == 0:
            new_errors.append(ErrorDiff(
                error_type=error_type, node_name=node_name, status="new",
                baseline_count=0, experiment_count=ec, message_sample=msg,
            ))
        elif ec == 0:
            resolved_errors.append(ErrorDiff(
                error_type=error_type, node_name=node_name, status="resolved",
                baseline_count=bc, experiment_count=0, message_sample=msg,
            ))
        elif bc != ec:
            count_changes.append(ErrorDiff(
                error_type=error_type, node_name=node_name, status="count_changed",
                baseline_count=bc, experiment_count=ec, message_sample=msg,
            ))

    return ErrorComparison(
        new_errors=new_errors,
        resolved_errors=resolved_errors,
        count_changes=count_changes,
    )


def _compare_convergence(
    b_conv: ConvergenceReport,
    e_conv: ConvergenceReport,
) -> ConvergenceDiff | None:
    if b_conv.total_passes == 0 and e_conv.total_passes == 0:
        return None

    # Build per-metric diff
    b_status = {m.metric_name: m.status for m in b_conv.per_metric}
    e_status = {m.metric_name: m.status for m in e_conv.per_metric}
    all_metrics = sorted(set(b_conv.final_metrics) | set(e_conv.final_metrics))

    metric_diffs: dict[str, dict[str, Any]] = {}
    for name in all_metrics:
        metric_diffs[name] = {
            "baseline_final": b_conv.final_metrics.get(name),
            "experiment_final": e_conv.final_metrics.get(name),
            "baseline_status": b_status.get(name),
            "experiment_status": e_status.get(name),
        }

    return ConvergenceDiff(
        baseline_is_converging=b_conv.is_converging,
        experiment_is_converging=e_conv.is_converging,
        baseline_total_passes=b_conv.total_passes,
        experiment_total_passes=e_conv.total_passes,
        metric_diffs=metric_diffs,
    )


def _compare_patterns(
    b_pat: LLMPatternsReport,
    e_pat: LLMPatternsReport,
) -> PatternComparison:
    b_map = {(p.pattern_type, p.node_name): p for p in b_pat.patterns}
    e_map = {(p.pattern_type, p.node_name): p for p in e_pat.patterns}

    new_patterns = [
        PatternDiff(
            pattern_type=p.pattern_type, node_name=p.node_name,
            status="new", severity=p.severity, description=p.description,
        )
        for key, p in sorted(e_map.items())
        if key not in b_map
    ]

    resolved_patterns = [
        PatternDiff(
            pattern_type=p.pattern_type, node_name=p.node_name,
            status="resolved", severity=p.severity, description=p.description,
        )
        for key, p in sorted(b_map.items())
        if key not in e_map
    ]

    return PatternComparison(
        new_patterns=new_patterns, resolved_patterns=resolved_patterns,
    )


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

_DIRECTION_WORD = {True: "increased", False: "decreased"}


def _build_summary(
    cost: CostComparison,
    duration: DurationComparison,
    routing: RoutingComparison,
    invocations: InvocationComparison,
    errors: ErrorComparison,
    convergence: ConvergenceDiff | None,
    llm_patterns: PatternComparison,
) -> str:
    items: list[tuple[int, str]] = []

    # Priority 0: new errors
    for e in errors.new_errors:
        node = f" in '{e.node_name}'" if e.node_name else ""
        items.append((0, f"New error: {e.error_type}{node} ({e.experiment_count} occurrences)."))

    # Priority 0: count-changed errors (experiment significantly higher)
    for e in errors.count_changes:
        if e.experiment_count > e.baseline_count:
            node = f" in '{e.node_name}'" if e.node_name else ""
            items.append((
                0,
                f"{e.error_type}{node} increased from "
                f"{e.baseline_count} to {e.experiment_count} occurrences.",
            ))

    # Priority 1: total cost change
    td = cost.total_delta
    if td and td.pct_change is not None and td.delta:
        word = _DIRECTION_WORD[td.pct_change > 0]
        items.append((
            1,
            f"Total cost {word} by {abs(td.pct_change):.1f}% "
            f"(${td.baseline:.4f} -> ${td.experiment:.4f}).",
        ))

    # Priority 2: total duration change
    dd = duration.total_delta
    if dd and dd.pct_change is not None and dd.delta_ms:
        word = _DIRECTION_WORD[dd.pct_change > 0]
        items.append((
            2,
            f"Total duration {word} by {abs(dd.pct_change):.1f}% "
            f"({dd.baseline_ms:.0f}ms -> {dd.experiment_ms:.0f}ms).",
        ))

    # Priority 2: flagged node duration changes
    for d in duration.node_deltas:
        if d.flagged and d.pct_change is not None and d.delta_ms:
            word = _DIRECTION_WORD[d.pct_change > 0]
            items.append((2, f"Duration of '{d.node_name}' {word} by {abs(d.pct_change):.1f}%."))

    # Priority 3: new warning-severity LLM patterns
    for p in llm_patterns.new_patterns:
        if p.severity == "warning":
            items.append((3, f"New LLM pattern (warning): {p.description}"))

    # Priority 4: routing edges added/removed
    for r in routing.edges_added:
        items.append((4, f"Routing edge {r.source_node}->{r.target_node} added."))
    for r in routing.edges_removed:
        items.append((4, f"Routing edge {r.source_node}->{r.target_node} removed."))

    # Priority 5: convergence status change
    if (
        convergence is not None
        and convergence.baseline_is_converging != convergence.experiment_is_converging
    ):
        b_str = str(convergence.baseline_is_converging)
        e_str = str(convergence.experiment_is_converging)
        items.append((5, f"Convergence status changed from {b_str} to {e_str}."))

    # Priority 6: resolved errors
    for e in errors.resolved_errors:
        node = f" in '{e.node_name}'" if e.node_name else ""
        items.append((6, f"Resolved: {e.error_type}{node} no longer occurs."))

    # Priority 7: invocation count changes
    for inv in invocations.changes:
        items.append((
            7,
            f"'{inv.node_name}' invocations changed "
            f"from {inv.baseline_count} to {inv.experiment_count}.",
        ))

    # Priority 8: resolved patterns, info-severity new patterns
    for p in llm_patterns.resolved_patterns:
        items.append((8, f"Resolved LLM pattern: {p.description}"))
    for p in llm_patterns.new_patterns:
        if p.severity != "warning":
            items.append((8, f"New LLM pattern (info): {p.description}"))

    # Priority 8: count-changed errors (experiment lower)
    for e in errors.count_changes:
        if e.experiment_count < e.baseline_count:
            node = f" in '{e.node_name}'" if e.node_name else ""
            items.append((
                8,
                f"{e.error_type}{node} decreased from "
                f"{e.baseline_count} to {e.experiment_count} occurrences.",
            ))

    # Sort by priority, then alphabetically for determinism
    items.sort(key=lambda x: (x[0], x[1]))
    sentences = [s for _, s in items]
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_runs(
    baseline: Trace,
    experiment: Trace,
    *,
    baseline_report: Any = None,
    experiment_report: Any = None,
    duration_flag_threshold: float = 0.20,
) -> RunComparison:
    """Compare two traces and return a structured diff.

    Args:
        baseline: The reference trace (before the change).
        experiment: The trace to evaluate (after the change).
        baseline_report: Pre-computed ``FullReport`` for *baseline*.
        experiment_report: Pre-computed ``FullReport`` for *experiment*.
        duration_flag_threshold: Fraction (0-1) above which a node
            duration change is flagged (default 0.20 = 20%).

    Returns:
        A ``RunComparison`` with deltas across all analysis dimensions
        and a natural-language summary.
    """
    # Lazy import to avoid circular dependency within the report package
    from orcheval.report import report as _report

    b_report = baseline_report or _report(baseline)
    e_report = experiment_report or _report(experiment)

    cost = _compare_costs(b_report.cost, e_report.cost)
    dur = _compare_durations(baseline, experiment, duration_flag_threshold)
    rout = _compare_routing(b_report.routing, e_report.routing)
    inv = _compare_invocations(baseline, experiment)
    errs = _compare_errors(baseline, experiment)
    conv = _compare_convergence(b_report.convergence, e_report.convergence)
    pats = _compare_patterns(b_report.llm_patterns, e_report.llm_patterns)

    summary = _build_summary(cost, dur, rout, inv, errs, conv, pats)

    return RunComparison(
        cost=cost,
        duration=dur,
        routing=rout,
        invocations=inv,
        errors=errs,
        convergence=conv,
        llm_patterns=pats,
        summary=summary,
    )


__all__ = [
    # Leaf models
    "CostDelta",
    "DurationDelta",
    "RoutingDiff",
    "InvocationDelta",
    "ErrorDiff",
    "ConvergenceDiff",
    "PatternDiff",
    # Grouping models
    "CostComparison",
    "DurationComparison",
    "RoutingComparison",
    "InvocationComparison",
    "ErrorComparison",
    "PatternComparison",
    # Root
    "RunComparison",
    # Function
    "compare_runs",
]
