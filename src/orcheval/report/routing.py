"""Routing decision audit with suspicious pattern detection."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from orcheval.events import RoutingDecision

if TYPE_CHECKING:
    from orcheval.trace import Trace

RoutingFlagType = Literal[
    "invariant_routing",
    "context_divergence",
    "dominant_path",
    "oscillation",
]

MAX_SAMPLE_CONTEXTS = 3
MIN_OSCILLATION_ALTERNATIONS = 3


class RoutingEdge(BaseModel):
    """An observed source -> target routing edge with frequency data."""

    model_config = {"frozen": True}

    source_node: str
    target_node: str
    count: int
    fraction: float
    sample_contexts: list[dict[str, Any]] = Field(default_factory=list)


class RoutingFlag(BaseModel):
    """A suspicious routing pattern detected in the trace."""

    model_config = {"frozen": True}

    flag_type: RoutingFlagType
    source_node: str
    target_node: str | None = None
    description: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class RoutingReport(BaseModel):
    """Full routing decision audit."""

    model_config = {"frozen": True}

    decisions: list[RoutingEdge] = Field(default_factory=list)
    flags: list[RoutingFlag] = Field(default_factory=list)
    total_decisions: int = 0
    unique_sources: int = 0
    unique_targets: int = 0


def _serialize_context(ctx: dict[str, Any]) -> str:
    """Stable string representation of a decision context for comparison."""
    return json.dumps(ctx, sort_keys=True, default=str)


def _detect_invariant_routing(
    source: str,
    targets: list[str],
) -> RoutingFlag | None:
    """Flag if a source always routes to the same target (>=2 decisions)."""
    if len(targets) < 2:
        return None
    unique = set(targets)
    if len(unique) == 1:
        target = next(iter(unique))
        return RoutingFlag(
            flag_type="invariant_routing",
            source_node=source,
            target_node=target,
            description=(
                f"Node '{source}' always routes to '{target}' "
                f"across {len(targets)} decisions"
            ),
            evidence={"decision_count": len(targets)},
        )
    return None


def _detect_context_divergence(
    source: str,
    decisions: list[RoutingDecision],
) -> list[RoutingFlag]:
    """Flag when identical decision_context maps to different targets."""
    ctx_to_targets: dict[str, set[str]] = defaultdict(set)
    ctx_to_raw: dict[str, dict[str, Any]] = {}
    for d in decisions:
        key = _serialize_context(d.decision_context)
        ctx_to_targets[key].add(d.target_node)
        if key not in ctx_to_raw:
            ctx_to_raw[key] = d.decision_context

    flags = []
    for key, targets in ctx_to_targets.items():
        if len(targets) > 1:
            flags.append(RoutingFlag(
                flag_type="context_divergence",
                source_node=source,
                description=(
                    f"Node '{source}' routes to different targets "
                    f"{sorted(targets)} with identical decision context"
                ),
                evidence={
                    "context": ctx_to_raw[key],
                    "targets": sorted(targets),
                },
            ))
    return flags


def _detect_dominant_path(
    source: str,
    target_counts: dict[str, int],
    total: int,
    threshold: float,
) -> list[RoutingFlag]:
    """Flag when a single target gets >= threshold fraction (with >1 distinct target)."""
    if len(target_counts) <= 1:
        return []
    flags = []
    for target, count in target_counts.items():
        fraction = count / total
        if fraction >= threshold:
            flags.append(RoutingFlag(
                flag_type="dominant_path",
                source_node=source,
                target_node=target,
                description=(
                    f"Node '{source}' routes to '{target}' in "
                    f"{fraction:.0%} of decisions ({count}/{total})"
                ),
                evidence={
                    "count": count,
                    "total": total,
                    "fraction": fraction,
                    "threshold": threshold,
                },
            ))
    return flags


def _detect_oscillation(
    source: str,
    target_sequence: list[str],
) -> list[RoutingFlag]:
    """Flag when a source alternates between two targets repeatedly."""
    if len(target_sequence) < MIN_OSCILLATION_ALTERNATIONS * 2:
        return []

    # Count alternations: consecutive pairs where target changes
    alternations = 0
    for i in range(1, len(target_sequence)):
        if target_sequence[i] != target_sequence[i - 1]:
            alternations += 1

    # Check if it's predominantly a two-target oscillation
    unique_targets = set(target_sequence)
    if len(unique_targets) == 2 and alternations >= MIN_OSCILLATION_ALTERNATIONS:
        pair = sorted(unique_targets)
        return [RoutingFlag(
            flag_type="oscillation",
            source_node=source,
            description=(
                f"Node '{source}' oscillates between '{pair[0]}' and "
                f"'{pair[1]}' with {alternations} alternations across "
                f"{len(target_sequence)} decisions"
            ),
            evidence={
                "targets": pair,
                "alternations": alternations,
                "total_decisions": len(target_sequence),
            },
        )]
    return []


def routing_report(
    trace: Trace,
    dominance_threshold: float = 0.95,
) -> RoutingReport:
    """Audit all routing decisions and flag suspicious patterns."""
    all_decisions = trace.get_events_by_type(RoutingDecision)
    if not all_decisions:
        return RoutingReport()

    # Group by source node (preserve order for oscillation detection)
    source_decisions: dict[str, list[RoutingDecision]] = defaultdict(list)
    for d in all_decisions:
        source_decisions[d.source_node].append(d)

    edges: list[RoutingEdge] = []
    flags: list[RoutingFlag] = []
    all_targets: set[str] = set()

    for source, decisions in sorted(source_decisions.items()):
        # Count targets
        target_counts: dict[str, int] = defaultdict(int)
        target_contexts: dict[str, list[dict[str, Any]]] = defaultdict(list)
        target_sequence: list[str] = []

        for d in decisions:
            target_counts[d.target_node] += 1
            if len(target_contexts[d.target_node]) < MAX_SAMPLE_CONTEXTS:
                target_contexts[d.target_node].append(d.decision_context)
            target_sequence.append(d.target_node)
            all_targets.add(d.target_node)

        total = len(decisions)

        # Build edges
        for target, count in sorted(target_counts.items()):
            edges.append(RoutingEdge(
                source_node=source,
                target_node=target,
                count=count,
                fraction=count / total,
                sample_contexts=target_contexts[target],
            ))

        # Detect patterns
        inv = _detect_invariant_routing(source, target_sequence)
        if inv:
            flags.append(inv)

        flags.extend(_detect_context_divergence(source, decisions))
        flags.extend(_detect_dominant_path(source, target_counts, total, dominance_threshold))
        flags.extend(_detect_oscillation(source, target_sequence))

    return RoutingReport(
        decisions=edges,
        flags=flags,
        total_decisions=len(all_decisions),
        unique_sources=len(source_decisions),
        unique_targets=len(all_targets),
    )
