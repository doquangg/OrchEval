"""Compact text digest of a trace, optimized for LLM consumption.

Produces a Markdown-formatted summary that reads like a structured case file,
not a data dump. Designed for feeding to an LLM reasoning about what went wrong.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from orcheval.events import ErrorEvent, LLMCall, NodeExit, ToolCall
from orcheval.report import FullReport
from orcheval.report import report as generate_report

if TYPE_CHECKING:
    from orcheval.trace import NodeInvocation, Trace


def build_digest(
    trace: Trace,
    *,
    include_llm_content: bool = False,
    focus_nodes: list[str] | None = None,
    reports: FullReport | None = None,
    max_chars: int = 16_000,
) -> str:
    """Build a compact narrative text summary of a trace.

    Args:
        trace: The trace to summarize.
        include_llm_content: Include full LLM prompt/response content.
        focus_nodes: Only show these nodes in detail; collapse others.
        reports: Pre-computed FullReport to avoid redundant computation.
        max_chars: Character budget (~4 chars per token).

    Returns:
        Markdown-formatted digest string.
    """
    full = _get_reports(trace, reports)
    invocations = trace.node_invocations()

    sections: list[tuple[str, str, int]] = []  # (name, content, priority)

    title = f"# Trace Digest: {trace.trace_id}\n"
    overview = _render_overview(trace, full, invocations)
    sections.append(("title", title, 100))
    sections.append(("overview", overview, 90))

    anomalies = _render_anomalies(full)
    sections.append(("anomalies", anomalies, 80))

    execution = _render_execution_flow(trace, invocations, focus_nodes)
    sections.append(("execution", execution, 60))

    state_evo = _render_state_evolution(trace, focus_nodes)
    if state_evo:
        sections.append(("state", state_evo, 40))

    if include_llm_content:
        llm_detail = _render_llm_detail(trace, focus_nodes)
        if llm_detail:
            sections.append(("llm_detail", llm_detail, 20))

    return _apply_budget(sections, max_chars)


def _get_reports(trace: Trace, reports: FullReport | None) -> FullReport:
    """Return provided reports or compute them."""
    if reports is not None:
        return reports
    return generate_report(trace)


def _render_overview(
    trace: Trace, full: FullReport, invocations: list[NodeInvocation],
) -> str:
    """Render the Overview section."""
    node_seq = trace.node_sequence()
    unique_nodes = len(set(node_seq))
    inv_count = len(invocations)
    duration = trace.total_duration()
    cost = trace.total_cost()
    tokens = trace.total_tokens()
    error_count = full.retries.total_errors

    parts: list[str] = []
    if unique_nodes == inv_count:
        parts.append(f"{unique_nodes} nodes")
    else:
        parts.append(f"{unique_nodes} nodes ({inv_count} invocations)")

    if duration is not None:
        parts.append(f"{duration:.0f}ms")

    if cost is not None:
        parts.append(f"${cost:.4f}")

    if tokens["total"] > 0:
        parts.append(f"{tokens['total']} tokens")

    if error_count > 0:
        parts.append(f"{error_count} errors")
    else:
        parts.append("no errors")

    summary_line = " | ".join(parts)
    return f"## Overview\n\n{summary_line}\n"


def _render_execution_flow(
    trace: Trace, invocations: list[NodeInvocation], focus_nodes: list[str] | None,
) -> str:
    """Render the Execution Flow section."""
    if not invocations:
        return "## Execution Flow\n\nNo node executions recorded.\n"

    # Count invocations per node to detect retries
    node_inv_counts: dict[str, int] = {}
    for inv in invocations:
        node_inv_counts[inv.node_name] = node_inv_counts.get(inv.node_name, 0) + 1

    lines: list[str] = ["## Execution Flow\n"]
    collapsed_count = 0
    collapsed_duration = 0.0
    step = 0

    for inv in invocations:
        step += 1

        # Focus filtering
        if focus_nodes is not None and inv.node_name not in focus_nodes:
            collapsed_count += 1
            if inv.duration_ms is not None:
                collapsed_duration += inv.duration_ms
            continue

        # Flush collapsed nodes before showing a focused/visible node
        if collapsed_count > 0:
            lines.append(
                f"\n*... {collapsed_count} other node invocations"
                f" ({collapsed_duration:.0f}ms total) ...*\n"
            )
            collapsed_count = 0
            collapsed_duration = 0.0

        events = _invocation_events(trace, inv)
        errors = [e for e in events if isinstance(e, ErrorEvent)]
        is_retry = node_inv_counts[inv.node_name] > 1
        is_notable = bool(errors) or is_retry
        dur = f" ({inv.duration_ms:.0f}ms)" if inv.duration_ms is not None else ""

        if is_notable:
            lines.append(f"\n{step}. **{inv.node_name}**{dur}")
            for evt in events:
                lines.append(_format_child_event(evt))
        else:
            summary_parts = _summarize_child_events(events)
            suffix = f" — {', '.join(summary_parts)}" if summary_parts else ""
            lines.append(f"\n{step}. **{inv.node_name}**{dur}{suffix}")

        # State info
        _append_state_info(events, lines)

    # Flush trailing collapsed nodes
    if collapsed_count > 0:
        lines.append(
            f"\n*... {collapsed_count} other node invocations"
            f" ({collapsed_duration:.0f}ms total) ...*\n"
        )

    return "\n".join(lines) + "\n"


def _render_anomalies(full: FullReport) -> str:
    """Render the Anomalies Detected section."""
    items: list[str] = []

    # Routing flags
    for flag in full.routing.flags:
        items.append(f"- **Routing — {flag.flag_type}**: {flag.description}")

    # Retry sequences
    for seq in full.retries.retry_sequences:
        status = "succeeded" if seq.succeeded else "failed"
        dur = f" ({seq.total_retry_duration_ms:.0f}ms)" if seq.total_retry_duration_ms else ""
        items.append(
            f"- **Retry**: {seq.node_name} — {seq.attempt_count} attempts, {status}{dur}"
        )

    # Convergence issues
    for metric in full.convergence.per_metric:
        if metric.status in ("diverging", "oscillating"):
            items.append(f"- **Convergence**: metric '{metric.metric_name}' is {metric.status}")

    # LLM patterns (warning severity only)
    for pat in full.llm_patterns.patterns:
        if pat.severity == "warning":
            items.append(f"- **LLM Pattern — {pat.pattern_type}**: {pat.description}")

    if not items:
        return "## Anomalies Detected\n\nNo anomalies detected.\n"

    return "## Anomalies Detected\n\n" + "\n".join(items) + "\n"


def _render_state_evolution(trace: Trace, focus_nodes: list[str] | None) -> str:
    """Render the State Evolution section. Returns empty string if no state data."""
    exits = trace.get_events_by_type(NodeExit)
    entries_with_state: list[tuple[str, dict[str, Any]]] = []

    for ex in exits:
        if not ex.state_diff:
            continue
        if focus_nodes is not None and ex.node_name not in focus_nodes:
            continue
        diff = ex.state_diff
        # Only include if there are actual changes
        added = diff.get("added", [])
        removed = diff.get("removed", [])
        modified = diff.get("modified", [])
        if added or removed or modified:
            entries_with_state.append((ex.node_name, diff))

    if not entries_with_state:
        return ""

    lines = ["## State Evolution\n"]
    for node_name, diff in entries_with_state:
        parts: list[str] = []
        if diff.get("added"):
            parts.append(f"added {diff['added']}")
        if diff.get("modified"):
            parts.append(f"modified {diff['modified']}")
        if diff.get("removed"):
            parts.append(f"removed {diff['removed']}")
        lines.append(f"- **{node_name}**: {', '.join(parts)}")

    return "\n".join(lines) + "\n"


def _render_llm_detail(trace: Trace, focus_nodes: list[str] | None) -> str:
    """Render the LLM Call Detail section. Returns empty string if no LLM calls."""
    llm_calls = trace.get_events_by_type(LLMCall)
    if not llm_calls:
        return ""

    if focus_nodes is not None:
        llm_calls = [c for c in llm_calls if c.node_name in focus_nodes]

    if not llm_calls:
        return ""

    lines = ["## LLM Call Detail\n"]
    for call in llm_calls:
        model = call.model or "unknown"
        node = call.node_name or "unknown"
        lines.append(f"\n### {node} → {model}\n")

        if call.system_message:
            lines.append(f"**System**: {call.system_message}\n")

        if call.input_messages:
            lines.append("**Input messages**:")
            for msg in call.input_messages:
                role = msg.get("role", "unknown")
                if role == "system" and call.system_message:
                    continue
                content = msg.get("content", "")
                # Truncate very long messages
                if isinstance(content, str) and len(content) > 1000:
                    content = content[:1000] + "…[truncated]"
                lines.append(f"  - [{role}]: {content}")

        if call.output_message:
            content = call.output_message.get("content", "")
            if not content and call.output_message.get("tool_calls"):
                names = [tc.get("name", "") for tc in call.output_message["tool_calls"]]
                content = f"[Tool calls: {', '.join(names)}]"
            if isinstance(content, str) and len(content) > 1000:
                content = content[:1000] + "…[truncated]"
            lines.append(f"\n**Output**: {content}")

    return "\n".join(lines) + "\n"


# --- Helpers ---


def _invocation_events(trace: Trace, invocation: NodeInvocation) -> list[Any]:
    """Get all events for a specific node invocation.

    Uses a union of two matching strategies:
    1. Child events linked via parent_span_id == invocation.span_id
       (LLMCall, ToolCall, ErrorEvent)
    2. The NodeExit event linked via span_id == invocation.span_id
       (NodeExit shares span_id with its NodeEntry, not parent_span_id)
    """
    result = []
    for event in trace:
        if (
            event.parent_span_id == invocation.span_id
            or (isinstance(event, NodeExit) and event.span_id == invocation.span_id)
        ):
            result.append(event)
    return result


def _format_child_event(event: Any) -> str:
    """Format a single child event as a sub-bullet."""
    if isinstance(event, LLMCall):
        model = event.model or "unknown"
        tok = ""
        if event.input_tokens is not None and event.output_tokens is not None:
            tok = f", {event.input_tokens}→{event.output_tokens} tokens"
        return f"   - LLM call ({model}{tok})"
    if isinstance(event, ToolCall):
        return f"   - Tool call: {event.tool_name}"
    if isinstance(event, ErrorEvent):
        return f"   - ERROR: {event.error_type}: {event.error_message}"
    if isinstance(event, NodeExit) and event.duration_ms is not None:
        return f"   - Exit ({event.duration_ms:.0f}ms)"
    return ""


def _summarize_child_events(events: list[Any]) -> list[str]:
    """Produce a compact summary of child events for a clean one-liner."""
    parts: list[str] = []
    llm_calls = [e for e in events if isinstance(e, LLMCall)]
    tool_calls = [e for e in events if isinstance(e, ToolCall)]

    if llm_calls:
        if len(llm_calls) == 1:
            c = llm_calls[0]
            model = c.model or "unknown"
            tok = ""
            if c.input_tokens is not None and c.output_tokens is not None:
                tok = f", {c.input_tokens}→{c.output_tokens} tokens"
            parts.append(f"1 LLM call ({model}{tok})")
        else:
            parts.append(f"{len(llm_calls)} LLM calls")

    if tool_calls:
        if len(tool_calls) == 1:
            parts.append(f"1 tool call ({tool_calls[0].tool_name})")
        else:
            names = ", ".join(t.tool_name for t in tool_calls[:3])
            parts.append(f"{len(tool_calls)} tool calls ({names})")

    return parts


def _append_state_info(events: list[Any], lines: list[str]) -> None:
    """Append state snapshot info to lines if present in events."""
    for event in events:
        if isinstance(event, NodeExit) and event.state_diff:
            diff = event.state_diff
            added = diff.get("added", [])
            modified = diff.get("modified", [])
            removed = diff.get("removed", [])
            if added or modified or removed:
                parts = []
                if added:
                    parts.append(f"+{added}")
                if modified:
                    parts.append(f"~{modified}")
                if removed:
                    parts.append(f"-{removed}")
                lines.append(f"   - State: {', '.join(parts)}")


def _apply_budget(
    sections: list[tuple[str, str, int]],
    max_chars: int,
) -> str:
    """Assemble sections and enforce character budget.

    Priority ordering (highest first): title, overview, anomalies,
    execution, state, llm_detail. Drops lowest-priority sections first.
    """
    total = sum(len(s[1]) for s in sections)
    if total <= max_chars:
        return "\n".join(s[1] for s in sections)

    # Sort by priority descending — drop from the bottom (lowest priority)
    ordered = sorted(sections, key=lambda s: s[2], reverse=True)

    kept: list[tuple[str, str, int]] = []
    budget = max_chars
    for name, content, priority in ordered:
        if len(content) <= budget:
            kept.append((name, content, priority))
            budget -= len(content)
        elif priority >= 80:
            # Never drop title, overview, or anomalies — include even if over budget
            kept.append((name, content, priority))
            budget -= len(content)
        else:
            # Try to include a truncated version for execution flow
            if name == "execution" and budget > 200:
                truncated = content[:budget - 50] + "\n\n*... (truncated)*\n"
                kept.append((name, truncated, priority))
                budget -= len(truncated)

    # Re-sort into document order: title(100), overview(90),
    # anomalies(80), execution(60), state(40), llm_detail(20)
    kept.sort(key=lambda s: -s[2])

    return "\n".join(s[1] for s in kept)
