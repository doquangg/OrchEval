"""Compare two runs to measure the impact of an architecture change.

Scenario: you have a multi-agent pipeline and you made three changes:
  1. Swapped the coder's model from gpt-4o to gpt-4o-mini (cheaper but riskier)
  2. Added a new "validator" node that checks code before review
  3. Fixed a prompt that was causing the reviewer to error out

OrchEval's compare_runs() gives you a structured diff across every dimension:
cost, duration, routing, errors, invocation counts, and LLM patterns.

Run:
    python examples/compare_two_runs.py

Outputs:
    orcheval_outputs/baseline.html      — waterfall for the original run
    orcheval_outputs/experiment.html    — waterfall for the modified run
    Printed: structured diff with natural-language summary
"""

from datetime import datetime, timedelta, timezone

from orcheval import Tracer, compare_runs

# Deterministic timestamps so total_duration() computes correctly.
# Events created without explicit timestamps all land at "now", which
# makes the trace appear to have 0ms duration. Real adapters set timestamps
# from actual callback times automatically.
_BASE = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _ts(seconds: float) -> datetime:
    return _BASE + timedelta(seconds=seconds)


# =============================================================================
# Baseline: planner → coder (gpt-4o) → reviewer
# The reviewer occasionally throws a ValueError due to a bad prompt.
# Total wall clock: ~7.7s, total cost: $0.022
# =============================================================================


def build_baseline() -> "Trace":
    tracer = Tracer(trace_id="baseline-run")
    a = tracer.adapter

    # Planner
    span = "span-b-planner"
    a.node_entry("planner", span_id=span, timestamp=_ts(0))
    a.llm_call(
        node_name="planner",
        parent_span_id=span,
        model="gpt-4o",
        input_tokens=200,
        output_tokens=100,
        cost=0.005,
        duration_ms=1000.0,
        timestamp=_ts(0.5),
    )
    a.node_exit("planner", span_id=span, duration_ms=1200.0, timestamp=_ts(1.2))

    # Coder (gpt-4o — expensive)
    span = "span-b-coder"
    a.node_entry("coder", span_id=span, timestamp=_ts(1.5))
    a.llm_call(
        node_name="coder",
        parent_span_id=span,
        model="gpt-4o",
        input_tokens=600,
        output_tokens=350,
        cost=0.015,
        duration_ms=3500.0,
        timestamp=_ts(2.0),
    )
    a.tool_call(
        "execute_code",
        node_name="coder",
        parent_span_id=span,
        tool_input={"code": "python main.py"},
        tool_output="OK",
        duration_ms=500.0,
        timestamp=_ts(5.5),
    )
    a.node_exit("coder", span_id=span, duration_ms=4500.0, timestamp=_ts(6.0))

    # Reviewer (errors due to bad prompt)
    span = "span-b-reviewer"
    a.node_entry("reviewer", span_id=span, timestamp=_ts(6.2))
    a.llm_call(
        node_name="reviewer",
        parent_span_id=span,
        model="gpt-4o-mini",
        input_tokens=400,
        output_tokens=150,
        cost=0.002,
        duration_ms=800.0,
        timestamp=_ts(6.5),
    )
    a.error(
        "ValueError",
        "Reviewer prompt produced unparseable JSON — missing closing brace",
        node_name="reviewer",
        parent_span_id=span,
        timestamp=_ts(7.3),
    )
    a.node_exit("reviewer", span_id=span, duration_ms=1200.0, timestamp=_ts(7.4))

    # Routing
    a.routing_decision("planner", "coder", node_name="planner", timestamp=_ts(1.2))
    a.routing_decision("coder", "reviewer", node_name="coder", timestamp=_ts(6.0))

    return tracer.collect()


# =============================================================================
# Experiment: planner → coder (gpt-4o-mini) → validator (new!) → reviewer
# Changes:
#   - Coder model swapped to gpt-4o-mini (cost savings, but more tokens needed)
#   - New validator node added between coder and reviewer
#   - Reviewer prompt fixed — no more ValueError
#   - Coder invoked twice (validator sent it back once)
# Total wall clock: ~13.5s (longer due to retry), total cost: $0.017 (cheaper)
# =============================================================================


def build_experiment() -> "Trace":
    tracer = Tracer(trace_id="experiment-run")
    a = tracer.adapter

    # Planner (unchanged)
    span = "span-e-planner"
    a.node_entry("planner", span_id=span, timestamp=_ts(0))
    a.llm_call(
        node_name="planner",
        parent_span_id=span,
        model="gpt-4o",
        input_tokens=200,
        output_tokens=100,
        cost=0.005,
        duration_ms=1000.0,
        timestamp=_ts(0.5),
    )
    a.node_exit("planner", span_id=span, duration_ms=1200.0, timestamp=_ts(1.2))

    # Coder attempt 1 (gpt-4o-mini — cheaper per call, but needs more tokens)
    span = "span-e-coder-1"
    a.node_entry("coder", span_id=span, timestamp=_ts(1.5))
    a.llm_call(
        node_name="coder",
        parent_span_id=span,
        model="gpt-4o-mini",
        input_tokens=800,      # more tokens needed with smaller model
        output_tokens=500,
        cost=0.004,            # but much cheaper
        duration_ms=2000.0,
        timestamp=_ts(2.0),
    )
    a.tool_call(
        "execute_code",
        node_name="coder",
        parent_span_id=span,
        tool_input={"code": "python main.py"},
        tool_output="FAIL: test_auth failed",
        duration_ms=500.0,
        timestamp=_ts(4.0),
    )
    a.node_exit("coder", span_id=span, duration_ms=3000.0, timestamp=_ts(4.5))

    # Validator (new node — catches the failing test)
    span = "span-e-validator"
    a.node_entry("validator", span_id=span, timestamp=_ts(4.8))
    a.llm_call(
        node_name="validator",
        parent_span_id=span,
        model="gpt-4o-mini",
        input_tokens=300,
        output_tokens=100,
        cost=0.001,
        duration_ms=600.0,
        timestamp=_ts(5.0),
    )
    a.node_exit("validator", span_id=span, duration_ms=800.0, timestamp=_ts(5.6))

    # Coder attempt 2 (retry after validator feedback)
    span = "span-e-coder-2"
    a.node_entry("coder", span_id=span, timestamp=_ts(5.8))
    a.llm_call(
        node_name="coder",
        parent_span_id=span,
        model="gpt-4o-mini",
        input_tokens=1000,
        output_tokens=450,
        cost=0.005,
        duration_ms=2200.0,
        timestamp=_ts(6.0),
    )
    a.tool_call(
        "execute_code",
        node_name="coder",
        parent_span_id=span,
        tool_input={"code": "python main.py"},
        tool_output="All tests passed",
        duration_ms=500.0,
        timestamp=_ts(8.5),
    )
    a.node_exit("coder", span_id=span, duration_ms=3200.0, timestamp=_ts(9.0))

    # Reviewer (prompt fixed — no more errors)
    span = "span-e-reviewer"
    a.node_entry("reviewer", span_id=span, timestamp=_ts(9.2))
    a.llm_call(
        node_name="reviewer",
        parent_span_id=span,
        model="gpt-4o-mini",
        input_tokens=500,
        output_tokens=200,
        cost=0.002,
        duration_ms=900.0,
        timestamp=_ts(9.5),
    )
    a.node_exit("reviewer", span_id=span, duration_ms=1100.0, timestamp=_ts(10.3))

    # Routing — now has the new validator hop and a coder retry loop
    a.routing_decision("planner", "coder", node_name="planner", timestamp=_ts(1.2))
    a.routing_decision("coder", "validator", node_name="coder", timestamp=_ts(4.5))
    a.routing_decision("validator", "coder", node_name="validator", timestamp=_ts(5.6))
    a.routing_decision("coder", "reviewer", node_name="coder", timestamp=_ts(9.0))

    return tracer.collect()


# =============================================================================
# Compare
# =============================================================================

baseline = build_baseline()
experiment = build_experiment()

diff = compare_runs(baseline, experiment)
# Equivalent: diff = baseline.compare(experiment)

print("=" * 60)
print("RUN COMPARISON: baseline vs experiment")
print("=" * 60)

# The summary is a natural-language description of all changes, prioritized
# by severity (new errors first, then cost, duration, patterns, routing, etc.)
print(f"\n--- Summary ---\n{diff.summary}")

# --- Cost ---
if diff.cost_total_delta:
    td = diff.cost_total_delta
    print(f"\n--- Cost ---")
    print(f"  Total: ${td.baseline:.4f} → ${td.experiment:.4f}", end="")
    if td.pct_change is not None:
        direction = "▲" if td.pct_change > 0 else "▼"
        print(f"  ({direction} {abs(td.pct_change):.1f}%)")
    else:
        print()

    print(f"\n  Per node:")
    for nd in diff.cost_node_deltas:
        b = f"${nd.baseline:.4f}" if nd.baseline is not None else "—"
        e = f"${nd.experiment:.4f}" if nd.experiment is not None else "—"
        print(f"    {nd.name}: {b} → {e}")

    print(f"\n  Per model:")
    for md in diff.cost_model_deltas:
        b = f"${md.baseline:.4f}" if md.baseline is not None else "—"
        e = f"${md.experiment:.4f}" if md.experiment is not None else "—"
        print(f"    {md.name}: {b} → {e}")

# --- Duration ---
if diff.duration_total_delta:
    dd = diff.duration_total_delta
    print(f"\n--- Duration ---")
    print(f"  Total: {dd.baseline_ms:.0f}ms → {dd.experiment_ms:.0f}ms", end="")
    if dd.pct_change is not None:
        direction = "▲" if dd.pct_change > 0 else "▼"
        flag = " ⚠️ FLAGGED" if dd.flagged else ""
        print(f"  ({direction} {abs(dd.pct_change):.1f}%){flag}")
    else:
        print()

    flagged = [d for d in diff.duration_node_deltas if d.flagged]
    if flagged:
        print(f"\n  Flagged nodes (>{20}% change):")
        for d in flagged:
            print(f"    {d.node_name}: {d.baseline_ms:.0f}ms → {d.experiment_ms:.0f}ms ({d.pct_change:+.1f}%)")

# --- Routing ---
if diff.routing_edges_added or diff.routing_edges_removed or diff.routing_edges_changed:
    print(f"\n--- Routing changes ---")
    for r in diff.routing_edges_added:
        print(f"  + {r.source_node} → {r.target_node} (new, {r.experiment_count}x)")
    for r in diff.routing_edges_removed:
        print(f"  - {r.source_node} → {r.target_node} (removed, was {r.baseline_count}x)")
    for r in diff.routing_edges_changed:
        print(f"  ~ {r.source_node} → {r.target_node}: {r.baseline_count}x → {r.experiment_count}x")

# --- Invocations ---
if diff.invocation_changes:
    print(f"\n--- Invocation count changes ---")
    for inv in diff.invocation_changes:
        print(f"  {inv.node_name}: {inv.baseline_count} → {inv.experiment_count} ({inv.delta:+d})")

# --- Errors ---
if diff.error_new:
    print(f"\n--- New errors ---")
    for e in diff.error_new:
        node = f" in {e.node_name}" if e.node_name else ""
        print(f"  {e.error_type}{node}: {e.experiment_count}x — \"{e.message_sample}\"")

if diff.error_resolved:
    print(f"\n--- Resolved errors ---")
    for e in diff.error_resolved:
        node = f" in {e.node_name}" if e.node_name else ""
        print(f"  {e.error_type}{node}: was {e.baseline_count}x — \"{e.message_sample}\"")

# --- LLM patterns ---
if diff.pattern_new:
    print(f"\n--- New LLM patterns ---")
    for p in diff.pattern_new:
        print(f"  [{p.severity}] {p.pattern_type} in {p.node_name}: {p.description}")

if diff.pattern_resolved:
    print(f"\n--- Resolved LLM patterns ---")
    for p in diff.pattern_resolved:
        print(f"  {p.pattern_type} in {p.node_name}: no longer detected")

# --- Export both runs ---
baseline.to_html("baseline.html")
experiment.to_html("experiment.html")
print(f"\nWrote: orcheval_outputs/baseline.html")
print(f"Wrote: orcheval_outputs/experiment.html")
print("Open both in a browser to compare the waterfall timelines side by side.")
