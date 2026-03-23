"""LLM call pattern analysis across node invocations.

Detects behavioral patterns invisible in individual events: prompt growth,
repeated outputs, redundant tool calls, system message variance, and
underutilized LLM output.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from orcheval.events import LLMCall, NodeEntry, NodeExit, ToolCall

if TYPE_CHECKING:
    from orcheval.trace import NodeInvocation, Trace

PROMPT_GROWTH_THRESHOLD = 0.50  # 50% growth triggers a warning


class LLMPattern(BaseModel):
    """A detected behavioral pattern in LLM call data."""

    model_config = {"frozen": True}

    pattern_type: Literal[
        "prompt_growth",
        "repeated_output",
        "redundant_tool_call",
        "system_message_variance",
        "output_not_utilized",
    ]
    node_name: str
    description: str
    evidence: dict[str, Any]
    severity: Literal["info", "warning"]


class NodeLLMSummary(BaseModel):
    """Aggregate LLM and tool call statistics for a single node."""

    model_config = {"frozen": True}

    node_name: str
    total_llm_calls: int
    total_tool_calls: int
    avg_input_tokens: float | None = None
    avg_output_tokens: float | None = None
    invocation_count: int


class LLMPatternsReport(BaseModel):
    """Full LLM call pattern analysis across all nodes."""

    model_config = {"frozen": True}

    patterns: list[LLMPattern] = Field(default_factory=list)
    total_llm_calls: int = 0
    nodes_analyzed: int = 0
    node_summaries: list[NodeLLMSummary] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _InvocationMap:
    """Groups LLM calls and tool calls by node name and invocation span."""

    __slots__ = ("llm_by_node", "tool_by_node", "all_llm", "all_tool")

    def __init__(
        self,
        llm_by_node: dict[str, list[list[LLMCall]]],
        tool_by_node: dict[str, list[list[ToolCall]]],
        all_llm: list[LLMCall],
        all_tool: list[ToolCall],
    ) -> None:
        self.llm_by_node = llm_by_node
        self.tool_by_node = tool_by_node
        self.all_llm = all_llm
        self.all_tool = all_tool


def _build_invocation_map(
    trace: Trace, invocations: list[NodeInvocation],
) -> _InvocationMap:
    """Build per-node, per-invocation groupings of LLM and tool calls.

    LLM/tool calls are matched to invocations via ``parent_span_id``.
    Calls whose ``parent_span_id`` is ``None`` or doesn't match any
    invocation are still included in the flat ``all_llm`` / ``all_tool``
    lists (and counted in summaries) but are excluded from per-invocation
    pattern detection.
    """

    # span_id -> (node_name, invocation_index)
    span_to_inv: dict[str, tuple[str, int]] = {}
    # node_name -> list of invocation indices (in order)
    node_inv_indices: dict[str, list[int]] = defaultdict(list)

    for idx, inv in enumerate(invocations):
        span_to_inv[inv.span_id] = (inv.node_name, idx)
        node_inv_indices[inv.node_name].append(idx)

    num_invocations = len(invocations)

    # Initialise per-invocation buckets
    llm_per_inv: list[list[LLMCall]] = [[] for _ in range(num_invocations)]
    tool_per_inv: list[list[ToolCall]] = [[] for _ in range(num_invocations)]

    all_llm = trace.get_llm_calls()
    all_tool = trace.get_tool_calls()

    for call in all_llm:
        if call.parent_span_id and call.parent_span_id in span_to_inv:
            _, inv_idx = span_to_inv[call.parent_span_id]
            llm_per_inv[inv_idx].append(call)

    for call in all_tool:
        if call.parent_span_id and call.parent_span_id in span_to_inv:
            _, inv_idx = span_to_inv[call.parent_span_id]
            tool_per_inv[inv_idx].append(call)

    # Build node-level groupings: node_name -> [invocation_0_calls, invocation_1_calls, ...]
    llm_by_node: dict[str, list[list[LLMCall]]] = {}
    tool_by_node: dict[str, list[list[ToolCall]]] = {}

    for node_name, indices in node_inv_indices.items():
        llm_by_node[node_name] = [llm_per_inv[i] for i in indices]
        tool_by_node[node_name] = [tool_per_inv[i] for i in indices]

    return _InvocationMap(
        llm_by_node=llm_by_node,
        tool_by_node=tool_by_node,
        all_llm=all_llm,
        all_tool=all_tool,
    )


# ---------------------------------------------------------------------------
# Pattern detectors
# ---------------------------------------------------------------------------


def _detect_prompt_growth(inv_map: _InvocationMap) -> list[LLMPattern]:
    """Flag nodes where prompt size grows >50% across invocations."""
    patterns: list[LLMPattern] = []

    for node_name, inv_llm_lists in inv_map.llm_by_node.items():
        if len(inv_llm_lists) < 2:
            continue

        # Take the first LLM call with input_tokens from each invocation
        tokens_per_inv: list[int] = []
        for calls in inv_llm_lists:
            for c in calls:
                if c.input_tokens is not None:
                    tokens_per_inv.append(c.input_tokens)
                    break

        if len(tokens_per_inv) < 2:
            continue

        first = tokens_per_inv[0]
        last = tokens_per_inv[-1]
        if first <= 0:
            continue

        growth = (last - first) / first
        if growth > PROMPT_GROWTH_THRESHOLD:
            patterns.append(LLMPattern(
                pattern_type="prompt_growth",
                node_name=node_name,
                description=(
                    f"Prompt size grew {growth:.0%} across "
                    f"{len(tokens_per_inv)} invocations of '{node_name}' "
                    f"({first} -> {last} tokens)"
                ),
                evidence={
                    "first_invocation_tokens": first,
                    "last_invocation_tokens": last,
                    "growth_pct": round(growth * 100, 1),
                    "invocation_count": len(tokens_per_inv),
                },
                severity="warning",
            ))

    return patterns


def _detect_repeated_output(inv_map: _InvocationMap) -> list[LLMPattern]:
    """Flag nodes producing identical output across invocations with different inputs."""
    patterns: list[LLMPattern] = []

    for node_name, inv_llm_lists in inv_map.llm_by_node.items():
        if len(inv_llm_lists) < 2:
            continue

        # Collect (response_summary, prompt_summary) for the first LLM call per invocation
        summaries: list[tuple[str | None, str | None]] = []
        for calls in inv_llm_lists:
            if calls:
                summaries.append((calls[0].response_summary, calls[0].prompt_summary))
            else:
                summaries.append((None, None))

        for i in range(len(summaries) - 1):
            resp_a, prompt_a = summaries[i]
            resp_b, prompt_b = summaries[i + 1]

            if resp_a is None or resp_b is None:
                continue
            if resp_a != resp_b:
                continue
            # Same output — only flag if inputs differ
            if prompt_a == prompt_b:
                continue

            patterns.append(LLMPattern(
                pattern_type="repeated_output",
                node_name=node_name,
                description=(
                    f"'{node_name}' produced identical output across invocations "
                    f"{i + 1} and {i + 2} despite different inputs"
                ),
                evidence={
                    "invocation_indices": [i, i + 1],
                    "matching_output": resp_a,
                    "input_summaries": [prompt_a, prompt_b],
                },
                severity="warning",
            ))

    return patterns


def _detect_redundant_tool_calls(inv_map: _InvocationMap) -> list[LLMPattern]:
    """Flag identical tool calls within a single invocation."""
    patterns: list[LLMPattern] = []

    for node_name, inv_tool_lists in inv_map.tool_by_node.items():
        for inv_idx, calls in enumerate(inv_tool_lists):
            if len(calls) < 2:
                continue

            # Group by (tool_name, serialized_input)
            call_keys: dict[str, int] = defaultdict(int)
            call_samples: dict[str, ToolCall] = {}
            for tc in calls:
                try:
                    key = tc.tool_name + "|" + json.dumps(tc.tool_input, sort_keys=True)
                except (TypeError, ValueError):
                    continue
                call_keys[key] += 1
                if key not in call_samples:
                    call_samples[key] = tc

            for key, count in call_keys.items():
                if count > 1:
                    sample = call_samples[key]
                    patterns.append(LLMPattern(
                        pattern_type="redundant_tool_call",
                        node_name=node_name,
                        description=(
                            f"Tool '{sample.tool_name}' called {count} times "
                            f"with identical input in invocation {inv_idx + 1} "
                            f"of '{node_name}'"
                        ),
                        evidence={
                            "tool_name": sample.tool_name,
                            "tool_input": sample.tool_input,
                            "call_count": count,
                            "invocation_index": inv_idx,
                        },
                        severity="warning",
                    ))

    return patterns


def _detect_system_message_variance(inv_map: _InvocationMap) -> list[LLMPattern]:
    """Flag nodes where system message differs across LLM calls."""
    patterns: list[LLMPattern] = []

    # Group all LLM calls by node_name (across all invocations)
    node_sys_messages: dict[str, set[str]] = defaultdict(set)
    node_call_counts: dict[str, int] = defaultdict(int)

    for call in inv_map.all_llm:
        node = call.node_name or "<unknown>"
        node_call_counts[node] += 1
        if call.system_message is not None:
            node_sys_messages[node].add(call.system_message)

    for node_name, messages in node_sys_messages.items():
        if len(messages) > 1:
            patterns.append(LLMPattern(
                pattern_type="system_message_variance",
                node_name=node_name,
                description=(
                    f"System message changed across {node_call_counts[node_name]} "
                    f"LLM calls in '{node_name}' "
                    f"({len(messages)} distinct messages)"
                ),
                evidence={
                    "distinct_messages": sorted(messages),
                    "call_count": node_call_counts[node_name],
                },
                severity="info",
            ))

    return patterns


def _detect_output_not_utilized(
    trace: Trace, inv_map: _InvocationMap, invocations: list[NodeInvocation],
) -> list[LLMPattern]:
    """Flag invocations where LLM produced output but state was unchanged."""
    patterns: list[LLMPattern] = []

    exits = trace.get_events_by_type(NodeExit)
    exit_by_span: dict[str, NodeExit] = {e.span_id: e for e in exits}

    # Precompute per-node local index: node_name -> {span_id -> local_idx}
    node_span_to_local: dict[str, dict[str, int]] = defaultdict(dict)
    for ni in invocations:
        idx_map = node_span_to_local[ni.node_name]
        idx_map[ni.span_id] = len(idx_map)

    for inv in invocations:
        node_exit = exit_by_span.get(inv.span_id)
        if node_exit is None:
            continue
        diff = node_exit.state_diff
        if not diff:
            continue

        # Check if state_diff has any changes
        added = diff.get("added", [])
        modified = diff.get("modified", [])
        if added or modified:
            continue

        # Check if this invocation had LLM calls with output
        if inv.node_name not in inv_map.llm_by_node:
            continue

        inv_lists = inv_map.llm_by_node[inv.node_name]
        local_idx = node_span_to_local.get(inv.node_name, {}).get(inv.span_id)
        if local_idx is None or local_idx >= len(inv_lists):
            continue

        llm_calls = inv_lists[local_idx]
        has_output = any(
            c.output_message is not None or c.response_summary is not None
            for c in llm_calls
        )
        if not has_output:
            continue

        output_summary = None
        for c in llm_calls:
            if c.response_summary:
                output_summary = c.response_summary
                break

        patterns.append(LLMPattern(
            pattern_type="output_not_utilized",
            node_name=inv.node_name,
            description=(
                f"LLM output in '{inv.node_name}' did not result in "
                f"state changes (no keys added or modified)"
            ),
            evidence={
                "llm_output_summary": output_summary,
                "state_diff": diff,
            },
            severity="info",
        ))

    return patterns


# ---------------------------------------------------------------------------
# Node summaries
# ---------------------------------------------------------------------------


def _build_node_summaries(
    inv_map: _InvocationMap, invocations: list[NodeInvocation],
) -> list[NodeLLMSummary]:
    """Build per-node aggregate statistics."""
    # Collect all node names from invocations and from LLM/tool calls
    node_names: set[str] = set(inv_map.llm_by_node.keys()) | set(inv_map.tool_by_node.keys())

    # Also include node names from unparented calls
    for c in inv_map.all_llm:
        if c.node_name:
            node_names.add(c.node_name)
    for c in inv_map.all_tool:
        if c.node_name:
            node_names.add(c.node_name)
    inv_counts: dict[str, int] = defaultdict(int)
    for inv in invocations:
        inv_counts[inv.node_name] += 1

    # Count all LLM/tool calls per node (including unparented)
    llm_counts: dict[str, int] = defaultdict(int)
    tool_counts: dict[str, int] = defaultdict(int)
    input_tokens_by_node: dict[str, list[int]] = defaultdict(list)
    output_tokens_by_node: dict[str, list[int]] = defaultdict(list)

    for c in inv_map.all_llm:
        node = c.node_name or "<unknown>"
        llm_counts[node] += 1
        if c.input_tokens is not None:
            input_tokens_by_node[node].append(c.input_tokens)
        if c.output_tokens is not None:
            output_tokens_by_node[node].append(c.output_tokens)

    for c in inv_map.all_tool:
        node = c.node_name or "<unknown>"
        tool_counts[node] += 1

    summaries: list[NodeLLMSummary] = []
    for name in sorted(node_names):
        in_toks = input_tokens_by_node.get(name)
        out_toks = output_tokens_by_node.get(name)
        summaries.append(NodeLLMSummary(
            node_name=name,
            total_llm_calls=llm_counts.get(name, 0),
            total_tool_calls=tool_counts.get(name, 0),
            avg_input_tokens=statistics.mean(in_toks) if in_toks else None,
            avg_output_tokens=statistics.mean(out_toks) if out_toks else None,
            invocation_count=inv_counts.get(name, 0),
        ))

    return summaries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def llm_patterns_report(trace: Trace) -> LLMPatternsReport:
    """Analyze LLM call patterns across node invocations.

    Detects prompt growth, repeated outputs, redundant tool calls,
    system message variance, and underutilized LLM output.
    """
    all_llm = trace.get_llm_calls()
    if not all_llm and not trace.get_tool_calls():
        return LLMPatternsReport()

    invocations = trace.node_invocations()
    inv_map = _build_invocation_map(trace, invocations)
    summaries = _build_node_summaries(inv_map, invocations)

    patterns: list[LLMPattern] = []
    patterns.extend(_detect_prompt_growth(inv_map))
    patterns.extend(_detect_repeated_output(inv_map))
    patterns.extend(_detect_redundant_tool_calls(inv_map))
    patterns.extend(_detect_system_message_variance(inv_map))
    patterns.extend(_detect_output_not_utilized(trace, inv_map, invocations))

    # Count unique nodes that have LLM or tool calls
    nodes_analyzed = len({s.node_name for s in summaries if s.total_llm_calls > 0 or s.total_tool_calls > 0})

    return LLMPatternsReport(
        patterns=patterns,
        total_llm_calls=len(inv_map.all_llm),
        nodes_analyzed=nodes_analyzed,
        node_summaries=summaries,
    )
