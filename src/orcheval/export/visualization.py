"""Self-contained HTML waterfall visualization for orchestration traces.

Generates a single HTML file with inline CSS and JS — zero external
dependencies. Designed for direct browser inspection of trace data.
"""

from __future__ import annotations

import json
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


def _css() -> str:
    """Return the inline CSS."""
    return """
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #f8f9fa; color: #333; font-size: 14px; }
#app { max-width: 1400px; margin: 0 auto; padding: 16px; }

/* Summary panel */
#summary-panel { display: flex; gap: 16px; flex-wrap: wrap;
                 background: #fff; border-radius: 8px; padding: 16px;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 16px; }
.metric-card { text-align: center; min-width: 100px; }
.metric-value { font-size: 24px; font-weight: 700; color: #1a1a1a; }
.metric-label { font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-error .metric-value { color: #e15759; }

/* Waterfall */
#waterfall-container { background: #fff; border-radius: 8px; padding: 16px;
                       box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 16px;
                       overflow-x: auto; }
#time-axis { position: relative; height: 24px; margin-left: 140px; margin-bottom: 8px;
             border-bottom: 1px solid #ddd; }
.time-tick { position: absolute; top: 0; font-size: 11px; color: #999;
             transform: translateX(-50%); }
.time-tick::after { content: ''; position: absolute; left: 50%; top: 16px;
                    width: 1px; height: 6px; background: #ccc; }
#swimlanes { position: relative; }
.swimlane { display: flex; align-items: center; height: 44px; border-bottom: 1px solid #f0f0f0; }
.swimlane-label { width: 140px; flex-shrink: 0; font-size: 13px; font-weight: 600;
                  padding-right: 12px; text-align: right; overflow: hidden;
                  text-overflow: ellipsis; white-space: nowrap; }
.swimlane-track { position: relative; flex: 1; height: 32px; }
.span-bar { position: absolute; height: 28px; top: 2px; border-radius: 4px;
            cursor: pointer; display: flex; align-items: center; padding: 0 6px;
            font-size: 11px; color: #fff; font-weight: 500; overflow: hidden;
            white-space: nowrap; text-overflow: ellipsis; min-width: 4px;
            transition: opacity 0.15s; }
.span-bar:hover { opacity: 0.85; }
.span-bar.has-error { background-image: repeating-linear-gradient(
    45deg, transparent, transparent 4px, rgba(255,255,255,0.25) 4px, rgba(255,255,255,0.25) 8px); }
.span-bar.open-ended { border-right: 2px dashed rgba(255,255,255,0.6); }
.child-marker { position: absolute; width: 8px; height: 8px; border-radius: 50%;
                top: 50%; transform: translate(-50%, -50%); border: 1.5px solid #fff;
                cursor: pointer; z-index: 1; }
.child-marker.error { background: #e15759 !important; }

/* Tooltip */
.tooltip { position: fixed; background: #333; color: #fff; padding: 6px 10px;
           border-radius: 4px; font-size: 12px; pointer-events: none;
           z-index: 1000; max-width: 300px; display: none; }

/* Detail panel */
#detail-panel { display: none; background: #fff; border-radius: 8px;
                padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
#detail-header { display: flex; justify-content: space-between; align-items: center;
                 margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #eee; }
#detail-title { font-size: 16px; font-weight: 700; }
#detail-close { background: none; border: none; font-size: 20px; cursor: pointer;
                color: #999; padding: 4px 8px; }
#detail-close:hover { color: #333; }
#detail-body { font-size: 13px; line-height: 1.6; }
.detail-section { margin-bottom: 16px; }
.detail-section h4 { font-size: 13px; font-weight: 700; color: #555;
                     text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
.state-diff .added { color: #2da44e; }
.state-diff .modified { color: #bf8700; }
.state-diff .removed { color: #cf222e; }
.msg-role { font-weight: 600; color: #555; }
.msg-content { background: #f6f8fa; padding: 8px; border-radius: 4px;
               margin: 4px 0 8px 0; white-space: pre-wrap; word-break: break-word;
               max-height: 200px; overflow-y: auto; font-family: monospace; font-size: 12px; }
.error-badge { display: inline-block; background: #fff5f5; color: #cf222e;
               border: 1px solid #cf222e; border-radius: 3px; padding: 2px 6px;
               font-size: 12px; margin: 2px 0; }
.tool-badge { display: inline-block; background: #f0f7ff; color: #0969da;
              border: 1px solid #0969da; border-radius: 3px; padding: 2px 6px;
              font-size: 12px; margin: 2px 0; }

/* Empty state */
.empty-state { text-align: center; padding: 40px; color: #999; }
"""


def _js() -> str:
    """Return the inline JavaScript."""
    return """
(function() {
  const D = TRACE_DATA;
  const panel = document.getElementById('detail-panel');
  const detailBody = document.getElementById('detail-body');
  const detailTitle = document.getElementById('detail-title');

  // Summary panel
  const sp = document.getElementById('summary-panel');
  function addMetric(label, value, isError) {
    const card = document.createElement('div');
    card.className = 'metric-card' + (isError ? ' metric-error' : '');
    card.innerHTML = '<div class="metric-value">' + value + '</div>'
                   + '<div class="metric-label">' + label + '</div>';
    sp.appendChild(card);
  }

  if (D.total_duration_ms != null) addMetric('Duration', D.total_duration_ms.toFixed(0) + 'ms');
  if (D.total_cost != null) addMetric('Cost', '$' + D.total_cost.toFixed(4));
  if (D.total_tokens) addMetric('Tokens', D.total_tokens.total.toLocaleString());
  addMetric('Nodes', D.node_count);
  addMetric('Errors', D.error_count, D.error_count > 0);

  // Empty state
  if (!D.spans || D.spans.length === 0) {
    document.getElementById('waterfall-container').innerHTML =
      '<div class="empty-state">No trace data to display.</div>';
    return;
  }

  // Time axis
  const totalMs = D.total_duration_ms || 1;
  const axisEl = document.getElementById('time-axis');
  const tickCount = Math.min(10, Math.max(2, Math.ceil(totalMs / 500)));
  const tickInterval = totalMs / tickCount;

  // Round tick interval to a nice number
  function niceInterval(v) {
    const mag = Math.pow(10, Math.floor(Math.log10(v)));
    const residual = v / mag;
    if (residual <= 1.5) return mag;
    if (residual <= 3.5) return 2 * mag;
    if (residual <= 7.5) return 5 * mag;
    return 10 * mag;
  }
  const nice = niceInterval(tickInterval);
  for (let t = 0; t <= totalMs; t += nice) {
    const pct = (t / totalMs) * 100;
    if (pct > 100) break;
    const tick = document.createElement('div');
    tick.className = 'time-tick';
    tick.style.left = pct + '%';
    tick.textContent = t >= 1000 ? (t / 1000).toFixed(1) + 's' : t.toFixed(0) + 'ms';
    axisEl.appendChild(tick);
  }

  // Swimlanes
  const lanesEl = document.getElementById('swimlanes');
  // Deduplicate node names preserving first-seen order
  const nodeOrder = [];
  D.spans.forEach(function(s) {
    if (nodeOrder.indexOf(s.node_name) === -1) nodeOrder.push(s.node_name);
  });

  // Create one swimlane per unique node
  const laneEls = {};
  nodeOrder.forEach(function(name) {
    const lane = document.createElement('div');
    lane.className = 'swimlane';
    const label = document.createElement('div');
    label.className = 'swimlane-label';
    label.textContent = name;
    label.title = name;
    const track = document.createElement('div');
    track.className = 'swimlane-track';
    lane.appendChild(label);
    lane.appendChild(track);
    lanesEl.appendChild(lane);
    laneEls[name] = track;
  });

  // Tooltip
  const tooltip = document.createElement('div');
  tooltip.className = 'tooltip';
  document.body.appendChild(tooltip);

  // Render span bars
  D.spans.forEach(function(span) {
    const track = laneEls[span.node_name];
    if (!track) return;

    const startPct = (span.start_ms / totalMs) * 100;
    const endMs = span.end_ms != null ? span.end_ms : totalMs;
    const widthPct = Math.max(0.3, ((endMs - span.start_ms) / totalMs) * 100);

    const bar = document.createElement('div');
    bar.className = 'span-bar';
    bar.style.left = startPct + '%';
    bar.style.width = widthPct + '%';
    bar.style.background = D.node_colors[span.node_name] || '#999';

    const detail = D.events_detail[span.span_id];
    if (detail && detail.errors.length > 0) bar.classList.add('has-error');
    if (span.end_ms == null) bar.classList.add('open-ended');

    const durText = span.duration_ms != null ? span.duration_ms.toFixed(0) + 'ms' : '';
    bar.textContent = durText;
    bar.setAttribute('data-span-id', span.span_id);

    // Hover
    bar.addEventListener('mouseenter', function(e) {
      const childCount = span.children ? span.children.length : 0;
      tooltip.innerHTML = '<strong>' + span.node_name + '</strong><br>'
        + (durText || 'in progress') + ' &middot; ' + childCount + ' events';
      tooltip.style.display = 'block';
    });
    bar.addEventListener('mousemove', function(e) {
      tooltip.style.left = (e.clientX + 12) + 'px';
      tooltip.style.top = (e.clientY - 8) + 'px';
    });
    bar.addEventListener('mouseleave', function() {
      tooltip.style.display = 'none';
    });

    // Click
    bar.addEventListener('click', function() { showDetail(span); });

    // Child event markers
    if (span.children && span.duration_ms) {
      const spanDur = span.duration_ms || (endMs - span.start_ms) || 1;
      span.children.forEach(function(child) {
        if (child.event_type === 'node_entry' || child.event_type === 'node_exit') return;
        const relOffset = child.offset_ms - span.start_ms;
        // Clamp to [0, 100] — child timestamps may fall slightly outside the parent span
        // due to clock skew or rounding, so we defensively bound the position.
        const posPct = Math.max(0, Math.min(100, (relOffset / spanDur) * 100));
        const marker = document.createElement('div');
        marker.className = 'child-marker';
        if (child.event_type === 'error_event') marker.classList.add('error');
        marker.style.left = posPct + '%';
        marker.style.background = D.node_colors[span.node_name] || '#999';
        marker.title = child.summary;
        bar.appendChild(marker);
      });
    }

    track.appendChild(bar);
  });

  // Detail panel
  function showDetail(span) {
    const detail = D.events_detail[span.span_id];
    detailTitle.textContent = span.node_name + ' (' + (span.duration_ms != null ? span.duration_ms.toFixed(0) + 'ms' : 'no duration') + ')';
    let html = '';

    // State diff
    if (detail && detail.state_diff) {
      const diff = detail.state_diff;
      const hasChanges = (diff.added && diff.added.length) || (diff.removed && diff.removed.length) || (diff.modified && diff.modified.length);
      if (hasChanges) {
        html += '<div class="detail-section state-diff"><h4>State Changes</h4>';
        if (diff.added && diff.added.length) html += '<div class="added">+ Added: ' + diff.added.join(', ') + '</div>';
        if (diff.modified && diff.modified.length) html += '<div class="modified">~ Modified: ' + diff.modified.join(', ') + '</div>';
        if (diff.removed && diff.removed.length) html += '<div class="removed">&minus; Removed: ' + diff.removed.join(', ') + '</div>';
        html += '</div>';
      }
    }

    // Input/output state
    if (detail && detail.input_state && Object.keys(detail.input_state).length) {
      html += '<div class="detail-section"><h4>Input State</h4>';
      html += '<div class="msg-content">' + escHtml(JSON.stringify(detail.input_state, null, 2)) + '</div></div>';
    }
    if (detail && detail.output_state && Object.keys(detail.output_state).length) {
      html += '<div class="detail-section"><h4>Output State</h4>';
      html += '<div class="msg-content">' + escHtml(JSON.stringify(detail.output_state, null, 2)) + '</div></div>';
    }

    // LLM calls
    if (detail && detail.llm_calls.length) {
      html += '<div class="detail-section"><h4>LLM Calls (' + detail.llm_calls.length + ')</h4>';
      detail.llm_calls.forEach(function(llm) {
        html += '<div style="margin-bottom:12px;padding:8px;background:#fafafa;border-radius:4px;">';
        html += '<strong>' + (llm.model || 'unknown') + '</strong>';
        if (llm.input_tokens != null) html += ' &middot; ' + llm.input_tokens + '→' + (llm.output_tokens || 0) + ' tokens';
        if (llm.cost != null) html += ' &middot; $' + llm.cost.toFixed(4);
        if (llm.duration_ms != null) html += ' &middot; ' + llm.duration_ms.toFixed(0) + 'ms';
        if (llm.system_message) {
          html += '<div style="margin-top:6px"><span class="msg-role">System:</span>';
          html += '<div class="msg-content">' + escHtml(llm.system_message) + '</div></div>';
        }
        if (llm.input_messages && llm.input_messages.length) {
          llm.input_messages.forEach(function(msg) {
            html += '<div><span class="msg-role">' + escHtml(msg.role || 'user') + ':</span>';
            html += '<div class="msg-content">' + escHtml(typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)) + '</div></div>';
          });
        }
        if (llm.output_message) {
          html += '<div><span class="msg-role">Assistant:</span>';
          var outContent = llm.output_message.content || JSON.stringify(llm.output_message);
          html += '<div class="msg-content">' + escHtml(typeof outContent === 'string' ? outContent : JSON.stringify(outContent)) + '</div></div>';
        }
        html += '</div>';
      });
      html += '</div>';
    }

    // Tool calls
    if (detail && detail.tool_calls.length) {
      html += '<div class="detail-section"><h4>Tool Calls (' + detail.tool_calls.length + ')</h4>';
      detail.tool_calls.forEach(function(tc) {
        html += '<div style="margin-bottom:8px;">';
        html += '<span class="tool-badge">' + escHtml(tc.tool_name) + '</span>';
        if (tc.duration_ms != null) html += ' ' + tc.duration_ms.toFixed(0) + 'ms';
        if (tc.tool_input && Object.keys(tc.tool_input).length) {
          html += '<div class="msg-content">' + escHtml(JSON.stringify(tc.tool_input, null, 2)) + '</div>';
        }
        if (tc.tool_output) {
          html += '<div class="msg-content">' + escHtml(tc.tool_output) + '</div>';
        }
        html += '</div>';
      });
      html += '</div>';
    }

    // Errors
    if (detail && detail.errors.length) {
      html += '<div class="detail-section"><h4>Errors (' + detail.errors.length + ')</h4>';
      detail.errors.forEach(function(err) {
        html += '<div class="error-badge">' + escHtml(err.error_type) + ': ' + escHtml(err.error_message) + '</div>';
        if (err.stacktrace) {
          html += '<div class="msg-content">' + escHtml(err.stacktrace) + '</div>';
        }
      });
      html += '</div>';
    }

    // Child events timeline
    if (span.children && span.children.length) {
      html += '<div class="detail-section"><h4>Events Timeline</h4>';
      span.children.forEach(function(child) {
        html += '<div style="padding:2px 0;">';
        html += '<span style="color:#999;font-size:11px;">' + child.offset_ms.toFixed(0) + 'ms</span> ';
        html += escHtml(child.summary);
        html += '</div>';
      });
      html += '</div>';
    }

    if (!html) html = '<div style="color:#999;">No details available for this span.</div>';

    detailBody.innerHTML = html;
    panel.style.display = 'block';
    panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  document.getElementById('detail-close').addEventListener('click', function() {
    panel.style.display = 'none';
  });

  function escHtml(s) {
    if (s == null) return '';
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
})();
"""
