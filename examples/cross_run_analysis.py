"""Cross-run analysis: aggregate statistics, outliers, trends, and execution shapes.

Scenario: you've been running a multi-agent pipeline in production and have
collected 10 traces over time. Some runs are normal, one is suspiciously
expensive (possible regression), one is suspiciously cheap (possible failure),
and the overall cost is trending upward.

TraceCollection lets you analyze patterns across runs that are invisible when
looking at individual traces.

Run:
    python examples/cross_run_analysis.py

Outputs:
    orcheval_outputs/traces/run_*.json  — serialized traces (reload later)
    Printed: aggregate stats, outlier detection, trend analysis, shape clustering
"""

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from orcheval import Trace, TraceCollection, Tracer

# Each trace gets timestamps offset by run_index * 1 hour, simulating traces
# collected over a 10-hour period. Within each trace, events are spaced
# realistically (seconds apart).
_DAY_BASE = datetime(2025, 6, 1, 8, 0, 0, tzinfo=timezone.utc)


def _trace_base(run_index: int) -> datetime:
    """Base timestamp for run N — each run is 1 hour after the previous."""
    return _DAY_BASE + timedelta(hours=run_index)


def _ts(run_index: int, seconds: float) -> datetime:
    """Timestamp within a specific run."""
    return _trace_base(run_index) + timedelta(seconds=seconds)


# =============================================================================
# Generate 10 traces with realistic variation
# =============================================================================

def build_trace(
    run_index: int,
    trace_id: str,
    *,
    nodes: list[str],
    cost_per_node: dict[str, float],
    tokens_per_node: dict[str, tuple[int, int]],
    error_node: str | None = None,
) -> Trace:
    """Build a trace with specified characteristics.

    Each node gets one LLM call with the given cost and token counts.
    If error_node is set, that node also produces an error.
    """
    tracer = Tracer(trace_id=trace_id)
    a = tracer.adapter

    t = 0.0
    for node_name in nodes:
        span_id = f"span-{trace_id}-{node_name}"
        a.node_entry(node_name, span_id=span_id, timestamp=_ts(run_index, t))

        cost = cost_per_node.get(node_name, 0.001)
        in_tok, out_tok = tokens_per_node.get(node_name, (100, 50))
        duration = cost * 200_000  # roughly proportional to cost

        a.llm_call(
            node_name=node_name,
            parent_span_id=span_id,
            model="gpt-4o" if cost > 0.005 else "gpt-4o-mini",
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost=cost,
            duration_ms=duration,
            timestamp=_ts(run_index, t + 0.5),
        )

        if error_node == node_name:
            a.error(
                "RuntimeError",
                f"Timeout in {node_name} after 30s",
                node_name=node_name,
                parent_span_id=span_id,
                timestamp=_ts(run_index, t + duration / 1000),
            )

        node_wall = duration / 1000 + 0.5  # convert to seconds, add overhead
        a.node_exit(
            node_name,
            span_id=span_id,
            duration_ms=duration * 1.25,
            timestamp=_ts(run_index, t + node_wall),
        )
        t += node_wall + 0.3  # gap between nodes

    return tracer.collect()


# Typical run: planner → coder → reviewer
# ~$0.025 total, varying slightly each time
def normal_run(run_index: int, run_id: int) -> Trace:
    noise = random.uniform(0.9, 1.1)
    return build_trace(
        run_index,
        trace_id=f"run_{run_id:03d}",
        nodes=["planner", "coder", "reviewer"],
        cost_per_node={
            "planner": 0.005 * noise,
            "coder": 0.015 * noise,
            "reviewer": 0.003 * noise,
        },
        tokens_per_node={
            "planner": (200, 100),
            "coder": (600, 350),
            "reviewer": (400, 150),
        },
    )


# Different shape: planner → coder → coder (retry) → reviewer
def retry_run(run_index: int, run_id: int) -> Trace:
    tracer = Tracer(trace_id=f"run_{run_id:03d}")
    a = tracer.adapter

    span = "span-planner"
    a.node_entry("planner", span_id=span, timestamp=_ts(run_index, 0))
    a.llm_call(node_name="planner", parent_span_id=span, model="gpt-4o",
               input_tokens=200, output_tokens=100, cost=0.005, duration_ms=1000.0,
               timestamp=_ts(run_index, 0.5))
    a.node_exit("planner", span_id=span, duration_ms=1200.0, timestamp=_ts(run_index, 1.2))

    # Coder attempt 1 — fails
    span = "span-coder-1"
    a.node_entry("coder", span_id=span, timestamp=_ts(run_index, 1.5))
    a.llm_call(node_name="coder", parent_span_id=span, model="gpt-4o",
               input_tokens=600, output_tokens=350, cost=0.015, duration_ms=3500.0,
               timestamp=_ts(run_index, 2.0))
    a.error("SyntaxError", "Unterminated string", node_name="coder",
            parent_span_id=span, timestamp=_ts(run_index, 5.5))
    a.node_exit("coder", span_id=span, duration_ms=4000.0, timestamp=_ts(run_index, 5.5))

    # Coder attempt 2 — succeeds
    span = "span-coder-2"
    a.node_entry("coder", span_id=span, timestamp=_ts(run_index, 6.0))
    a.llm_call(node_name="coder", parent_span_id=span, model="gpt-4o",
               input_tokens=900, output_tokens=400, cost=0.018, duration_ms=4000.0,
               timestamp=_ts(run_index, 6.5))
    a.node_exit("coder", span_id=span, duration_ms=4500.0, timestamp=_ts(run_index, 10.5))

    span = "span-reviewer"
    a.node_entry("reviewer", span_id=span, timestamp=_ts(run_index, 11.0))
    a.llm_call(node_name="reviewer", parent_span_id=span, model="gpt-4o-mini",
               input_tokens=500, output_tokens=200, cost=0.003, duration_ms=900.0,
               timestamp=_ts(run_index, 11.5))
    a.node_exit("reviewer", span_id=span, duration_ms=1100.0, timestamp=_ts(run_index, 12.1))

    return tracer.collect()


# Seed for reproducibility
random.seed(42)

traces: list[Trace] = []

# Runs 1-4: normal, with slightly increasing cost (simulating prompt creep)
for i in range(1, 5):
    traces.append(normal_run(run_index=i - 1, run_id=i))

# Run 5: retry shape (coder fails once)
traces.append(retry_run(run_index=4, run_id=5))

# Runs 6-8: normal but cost is drifting upward
for i in range(6, 9):
    # Each run is ~10% more expensive than the previous normal runs
    drift = 1.0 + (i - 5) * 0.10
    t = build_trace(
        run_index=i - 1,
        trace_id=f"run_{i:03d}",
        nodes=["planner", "coder", "reviewer"],
        cost_per_node={
            "planner": 0.005 * drift * random.uniform(0.95, 1.05),
            "coder": 0.015 * drift * random.uniform(0.95, 1.05),
            "reviewer": 0.003 * drift * random.uniform(0.95, 1.05),
        },
        tokens_per_node={
            "planner": (200, 100),
            "coder": (int(600 * drift), int(350 * drift)),
            "reviewer": (400, 150),
        },
    )
    traces.append(t)

# Run 9: cost outlier — one of the nodes used gpt-4o instead of gpt-4o-mini
traces.append(build_trace(
    run_index=8,
    trace_id="run_009",
    nodes=["planner", "coder", "reviewer"],
    cost_per_node={"planner": 0.005, "coder": 0.080, "reviewer": 0.003},  # coder 5x normal
    tokens_per_node={"planner": (200, 100), "coder": (2000, 1000), "reviewer": (400, 150)},
))

# Run 10: suspiciously cheap — possible silent failure
traces.append(build_trace(
    run_index=9,
    trace_id="run_010",
    nodes=["planner", "coder", "reviewer"],
    cost_per_node={"planner": 0.001, "coder": 0.002, "reviewer": 0.001},  # 10x cheaper than normal
    tokens_per_node={"planner": (50, 20), "coder": (80, 30), "reviewer": (40, 15)},
    error_node="coder",  # failed silently with minimal output
))


# =============================================================================
# Save traces to disk (demonstrates from_json_dir loading)
# =============================================================================

output_dir = Path("orcheval_outputs/traces")
output_dir.mkdir(parents=True, exist_ok=True)
for t in traces:
    t.to_json(str(output_dir / f"{t.trace_id}.json"))
print(f"Saved {len(traces)} traces to {output_dir}/")

# Reload from directory — same as if you collected these over days/weeks
collection = TraceCollection.from_json_dir(output_dir)
print(f"Loaded {len(collection)} traces from disk\n")


# =============================================================================
# Aggregate statistics
# =============================================================================

summary = collection.summary()

print("=" * 60)
print(f"COLLECTION SUMMARY ({summary.trace_count} traces)")
print("=" * 60)

if summary.total_cost:
    c = summary.total_cost
    print(f"\n  Cost:     mean=${c.mean:.4f}  median=${c.median:.4f}  "
          f"p95=${c.p95:.4f}  range=[${c.min:.4f}, ${c.max:.4f}]")

if summary.total_duration:
    d = summary.total_duration
    print(f"  Duration: mean={d.mean:.0f}ms  median={d.median:.0f}ms  "
          f"p95={d.p95:.0f}ms")

if summary.total_tokens:
    t = summary.total_tokens
    print(f"  Tokens:   mean={t.mean:.0f}  median={t.median:.0f}  "
          f"p95={t.p95:.0f}")

print(f"\n  Unique nodes: {summary.unique_node_names}")


# =============================================================================
# Per-node breakdown
# =============================================================================

print(f"\n--- Per-node statistics ---")
for ns in summary.node_stats:
    parts = [f"  {ns.node_name}:"]
    if ns.duration:
        parts.append(f"duration={ns.duration.median:.0f}ms (median)")
    if ns.cost:
        parts.append(f"cost=${ns.cost.median:.4f} (median)")
    if ns.error_rate is not None:
        parts.append(f"error_rate={ns.error_rate:.0%}")
    print("  ".join(parts))


# =============================================================================
# Execution shapes — which paths does the pipeline take?
# =============================================================================

print(f"\n--- Execution shapes ---")
for shape in collection.execution_shapes():
    pct = f"{shape.fraction:.0%}"
    ids = ", ".join(shape.trace_ids[:3])
    more = f" +{len(shape.trace_ids) - 3} more" if len(shape.trace_ids) > 3 else ""
    print(f"  {shape.node_sequence}")
    print(f"    {shape.trace_count} traces ({pct}): {ids}{more}")


# =============================================================================
# Outlier detection — flag runs that deviate significantly from the norm
# =============================================================================

print(f"\n--- Cost outliers (>2x or <0.5x median) ---")
outliers = collection.find_outliers(metric="cost", threshold=2.0)
if outliers:
    for o in outliers:
        direction = "HIGH" if o.value > o.median else "LOW"
        print(f"  [{direction}] {o.trace_id}: ${o.value:.4f} — {o.reason}")
else:
    print("  No outliers detected at 2x threshold.")

print(f"\n--- Duration outliers ---")
dur_outliers = collection.find_outliers(metric="duration", threshold=2.0)
if dur_outliers:
    for o in dur_outliers:
        direction = "SLOW" if o.value > o.median else "FAST"
        print(f"  [{direction}] {o.trace_id}: {o.value:.0f}ms — {o.reason}")
else:
    print("  No duration outliers detected.")


# =============================================================================
# Trend analysis — is cost creeping up over time?
# =============================================================================

print(f"\n--- Cost trend ---")
trend = collection.trend(metric="cost")
print(f"  Direction: {trend.direction}")
if trend.change_pct is not None:
    print(f"  Change:    {trend.change_pct:+.1f}% (first run to last run)")
print(f"  Points:    {len(trend.points)} data points")
# Show first and last for reference
if trend.points:
    first = trend.points[0]
    last = trend.points[-1]
    print(f"  First run: {first.trace_id} = ${first.value:.4f}")
    print(f"  Last run:  {last.trace_id} = ${last.value:.4f}")

print(f"\n--- Duration trend ---")
dur_trend = collection.trend(metric="duration")
print(f"  Direction: {dur_trend.direction}")
if dur_trend.change_pct is not None:
    print(f"  Change:    {dur_trend.change_pct:+.1f}%")
