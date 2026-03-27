"""Self-contained HTML waterfall visualization for orchestration traces.

Generates a single HTML file with inline CSS and JS — zero external
dependencies. Designed for direct browser inspection of trace data.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from orcheval.events import ErrorEvent, LLMCall, NodeEntry, NodeExit, ToolCall
from orcheval.report import FullReport
from orcheval.report import report as generate_report

if TYPE_CHECKING:
    from orcheval.trace import Trace

_PALETTE = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
    "#5fa2ce",
    "#fc7d0b",
]


def build_html(
    trace: Trace,
    *,
    reports: FullReport | None = None,
) -> str:
    """Build a self-contained HTML waterfall visualization.

    Args:
        trace: The trace to visualize.
        reports: Pre-computed FullReport to avoid redundant computation.

    Returns:
        Complete HTML string with inline CSS and JS.
    """
    full = reports if reports is not None else generate_report(trace)
    payload = _build_data_payload(trace, full)
    data_json = json.dumps(payload, ensure_ascii=False, default=str)
    # Escape </ sequences to prevent premature </script> closing
    data_json = data_json.replace("</", r"<\/")
    return _render_html(trace.trace_id, data_json)


def _assign_colors(node_names: list[str]) -> dict[str, str]:
    """Assign colors to nodes by first-seen order."""
    colors: dict[str, str] = {}
    for name in node_names:
        if name not in colors:
            colors[name] = _PALETTE[len(colors) % len(_PALETTE)]
    return colors


def _build_data_payload(trace: Trace, full: FullReport) -> dict[str, Any]:
    """Assemble the data dict that gets embedded as JSON in the HTML."""
    tl = full.timeline

    # Collect unique node names in first-seen order from spans
    node_names = []
    for span in tl.spans:
        if span.node_name not in node_names:
            node_names.append(span.node_name)

    node_colors = _assign_colors(node_names)

    # Build events_detail by iterating events ONCE and bucketing
    events_detail: dict[str, dict[str, Any]] = {}
    for span in tl.spans:
        events_detail[span.span_id] = {
            "input_state": {},
            "output_state": {},
            "state_diff": {},
            "llm_calls": [],
            "tool_calls": [],
            "errors": [],
        }

    for event in trace:
        if isinstance(event, NodeEntry) and event.span_id in events_detail:
            events_detail[event.span_id]["input_state"] = event.input_state
        elif isinstance(event, NodeExit) and event.span_id in events_detail:
            events_detail[event.span_id]["output_state"] = event.output_state
            events_detail[event.span_id]["state_diff"] = event.state_diff
        elif isinstance(event, LLMCall) and event.parent_span_id in events_detail:
            events_detail[event.parent_span_id]["llm_calls"].append({
                "model": event.model,
                "input_messages": event.input_messages,
                "output_message": event.output_message,
                "system_message": event.system_message,
                "input_tokens": event.input_tokens,
                "output_tokens": event.output_tokens,
                "cost": event.cost,
                "duration_ms": event.duration_ms,
                "prompt_summary": event.prompt_summary,
                "response_summary": event.response_summary,
            })
        elif isinstance(event, ToolCall) and event.parent_span_id in events_detail:
            events_detail[event.parent_span_id]["tool_calls"].append({
                "tool_name": event.tool_name,
                "tool_input": event.tool_input,
                "tool_output": event.tool_output,
                "duration_ms": event.duration_ms,
            })
        elif isinstance(event, ErrorEvent) and event.parent_span_id in events_detail:
            events_detail[event.parent_span_id]["errors"].append({
                "error_type": event.error_type,
                "error_message": event.error_message,
                "stacktrace": event.stacktrace,
            })

    error_count = len(trace.get_events_by_type(ErrorEvent))

    return {
        "trace_id": trace.trace_id,
        "total_duration_ms": tl.total_duration_ms,
        "total_cost": trace.total_cost(),
        "total_tokens": trace.total_tokens(),
        "error_count": error_count,
        "node_count": len(node_names),
        "spans": [s.model_dump(mode="json") for s in tl.spans],
        "node_colors": node_colors,
        "events_detail": events_detail,
    }


def _render_html(trace_id: str, data_json: str) -> str:
    """Assemble the complete HTML document."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OrchEval Trace: {trace_id}</title>
<style>
{_css()}
</style>
</head>
<body>
<div id="app">
  <div id="summary-panel"></div>
  <div id="waterfall-container">
    <div id="time-axis"></div>
    <div id="swimlanes"></div>
  </div>
  <div id="detail-panel">
    <div id="detail-header"><span id="detail-title"></span><button id="detail-close">&times;</button></div>
    <div id="detail-body"></div>
  </div>
</div>
<script>
const TRACE_DATA = {data_json};
{_js()}
</script>
</body>
</html>"""


_EXPORT_DIR = Path(__file__).parent


@lru_cache(maxsize=1)
def _css() -> str:
    """Return the inline CSS, loaded from visualization.css."""
    return (_EXPORT_DIR / "visualization.css").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _js() -> str:
    """Return the inline JavaScript, loaded from visualization.js."""
    return (_EXPORT_DIR / "visualization.js").read_text(encoding="utf-8")
