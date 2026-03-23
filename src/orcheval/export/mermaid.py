"""Mermaid diagram export for orchestration traces.

Produces a ``graph LR`` Mermaid string showing the execution topology
with edge traversal counts and node invocation counts.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orcheval.trace import Trace


def _sanitize_id(name: str) -> str:
    """Convert a node name into a valid Mermaid node identifier."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def build_mermaid(trace: Trace) -> str:
    """Build a Mermaid graph string from a trace.

    If the trace contains ``RoutingDecision`` events, edges are derived
    from those directly.  Otherwise, edges are inferred from consecutive
    node transitions in ``node_sequence()``.

    Nodes display their invocation count (e.g. ``agent (3x)``).  Nodes
    that produced ``ErrorEvent`` entries are styled with the ``error``
    class.
    """
    from orcheval.events import ErrorEvent, RoutingDecision

    sequence = trace.node_sequence()
    if not sequence:
        return "graph LR\n"

    # --- Node metadata ---
    invocation_counts: Counter[str] = Counter(sequence)
    error_nodes: set[str] = {
        e.node_name for e in trace.get_events_by_type(ErrorEvent) if e.node_name
    }

    # --- Edge counts ---
    routing_decisions = trace.get_events_by_type(RoutingDecision)
    edge_counts: Counter[tuple[str, str]] = Counter()

    if routing_decisions:
        for rd in routing_decisions:
            edge_counts[(rd.source_node, rd.target_node)] += 1
    else:
        for src, tgt in zip(sequence, sequence[1:]):
            edge_counts[(src, tgt)] += 1

    # --- Collect all node names that appear (in nodes or edges) ---
    all_nodes: set[str] = set(sequence)
    for src, tgt in edge_counts:
        all_nodes.add(src)
        all_nodes.add(tgt)

    # --- Build output ---
    lines: list[str] = ["graph LR"]

    has_errors = bool(error_nodes & all_nodes)
    if has_errors:
        lines.append("    classDef error fill:#f96,stroke:#333,stroke-width:2px")

    # Node definitions (sorted for determinism)
    for name in sorted(all_nodes):
        nid = _sanitize_id(name)
        count = invocation_counts.get(name, 0)
        label = f"{name} ({count}x)" if count > 0 else name
        label = label.replace('"', "&quot;")
        suffix = ":::error" if name in error_nodes else ""
        lines.append(f'    {nid}["{label}"]{suffix}')

    # Edge definitions (sorted for determinism)
    for (src, tgt) in sorted(edge_counts):
        count = edge_counts[(src, tgt)]
        lines.append(f'    {_sanitize_id(src)} -->|"{count}x"| {_sanitize_id(tgt)}')

    return "\n".join(lines) + "\n"
