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
    if (widthPct > 3) {
      bar.textContent = durText;
    }
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

        const isDurationEvent = child.event_type === 'llm_call' || child.event_type === 'tool_call';
        const hasDuration = isDurationEvent && child.duration_ms > 0;

        const marker = document.createElement('div');
        marker.style.left = posPct + '%';
        marker.title = child.summary;

        if (hasDuration) {
          // Sub-bar: width represents actual call duration, adjacent calls don't overlap
          const widthPct = Math.max(0.4, (child.duration_ms / spanDur) * 100);
          marker.className = 'child-marker sub-bar' +
            (child.event_type === 'tool_call' ? ' tool-call' : '');
          marker.style.width = widthPct + '%';
        } else {
          // Tick mark: no duration data or non-duration event type
          marker.className = 'child-marker tick' +
            (child.event_type === 'error_event' ? ' error' : '');
        }

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
            if (msg.role === 'system' && llm.system_message) return;
            var content = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
            if (!content && msg.tool_calls && msg.tool_calls.length) {
              content = msg.tool_calls.map(function(tc) {
                return tc.name + '(' + (typeof tc.args === 'string' ? tc.args : JSON.stringify(tc.args)) + ')';
              }).join('\n');
            }
            if (!content) return;
            html += '<div><span class="msg-role">' + escHtml(msg.role || 'user') + ':</span>';
            html += '<div class="msg-content">' + escHtml(content) + '</div></div>';
          });
        }
        if (llm.output_message) {
          var outContent = llm.output_message.content;
          if (typeof outContent !== 'string') outContent = outContent ? JSON.stringify(outContent) : '';
          if (!outContent && llm.output_message.tool_calls && llm.output_message.tool_calls.length) {
            outContent = llm.output_message.tool_calls.map(function(tc) {
              return tc.name + '(' + (typeof tc.args === 'string' ? tc.args : JSON.stringify(tc.args)) + ')';
            }).join('\n');
          }
          if (outContent) {
            html += '<div><span class="msg-role">Assistant:</span>';
            html += '<div class="msg-content">' + escHtml(outContent) + '</div></div>';
          }
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
