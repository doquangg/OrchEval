"""Diagnose a pathological multi-agent run using OrchEval's pattern detection.

This example simulates a broken architecture with several common anti-patterns:
  - Prompt growth: the coder's context balloons across retries
  - Redundant tool calls: the coder runs the same test twice in one invocation
  - Oscillating router: the supervisor bounces between coder and reviewer
  - Retry with eventual success: the coder fails once, then succeeds
  - Output not utilized: the reviewer produces output but doesn't change state

OrchEval detects all of these automatically. Run this to see what the report
surfaces without any manual instrumentation.

Run:
    python examples/diagnose_bad_architecture.py

Outputs:
    orcheval_outputs/bad_architecture.html — visual waterfall showing the mess
    Printed: every detected anomaly with severity and evidence
"""

from datetime import datetime, timedelta, timezone

from orcheval import Tracer, report

_BASE = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _ts(seconds: float) -> datetime:
    return _BASE + timedelta(seconds=seconds)


tracer = Tracer(trace_id="bad-architecture-demo")
a = tracer.adapter


# === Node 1: supervisor (routes between coder and reviewer) ==================
# This supervisor oscillates — it keeps bouncing back and forth instead of
# converging. OrchEval flags this as an "oscillation" routing pattern.

def emit_routing(source: str, target: str, t: float) -> None:
    """Helper to emit a routing decision."""
    a.routing_decision(
        source,
        target,
        node_name=source,
        decision_context={"reason": f"routing to {target}"},
        timestamp=_ts(t),
    )


# === Node 2: coder — retries with growing prompts ============================

# Attempt 1: coder fails with a syntax error
coder_span_1 = "span-coder-attempt-1"
a.node_entry("coder", span_id=coder_span_1, timestamp=_ts(0))
a.llm_call(
    node_name="coder",
    parent_span_id=coder_span_1,
    model="gpt-4o",
    input_tokens=400,       # <-- starts at 400 tokens
    output_tokens=200,
    cost=0.010,
    duration_ms=2500.0,
    timestamp=_ts(0.5),
    prompt_summary="Write a data pipeline",
    response_summary="def process(data): ...",
    system_message="You are a senior Python developer.",
    input_messages=[{"role": "user", "content": "Write a data pipeline for CSV processing"}],
    output_message={"role": "ai", "content": "def process(data): ..."},
)
a.error(
    "SyntaxError",
    "unexpected indent at line 15",
    node_name="coder",
    parent_span_id=coder_span_1,
    timestamp=_ts(3.0),
)
a.node_exit("coder", span_id=coder_span_1, duration_ms=3000.0, timestamp=_ts(3.0))

# Supervisor routes coder → reviewer → coder (oscillation starts)
for i, target in enumerate(["coder", "reviewer"] * 4):
    emit_routing("supervisor", target, t=3.5 + i * 0.2)

# Attempt 2: coder retries with the error context appended to the prompt.
# Input tokens grow from 400 → 900 (125% growth — well above the 50% threshold).
coder_span_2 = "span-coder-attempt-2"
a.node_entry("coder", span_id=coder_span_2, timestamp=_ts(6.0))
a.llm_call(
    node_name="coder",
    parent_span_id=coder_span_2,
    model="gpt-4o",
    input_tokens=900,       # <-- 125% growth from attempt 1
    output_tokens=250,
    cost=0.016,
    duration_ms=3800.0,
    timestamp=_ts(6.5),
    prompt_summary="Write a data pipeline (previous attempt failed with SyntaxError)",
    response_summary="def process(data): ...",  # <-- same output as attempt 1 (repeated output)
    system_message="You are a senior Python developer.",
    input_messages=[
        {"role": "user", "content": "Write a data pipeline for CSV processing"},
        {"role": "assistant", "content": "def process(data): ..."},
        {"role": "user", "content": "That had a SyntaxError at line 15. Fix it."},
    ],
    output_message={"role": "ai", "content": "def process(data): ..."},
)
# Redundant tool calls: coder runs the exact same test twice
a.tool_call(
    "run_tests",
    node_name="coder",
    parent_span_id=coder_span_2,
    tool_input={"command": "pytest tests/ -x"},
    tool_output="5 passed",
    duration_ms=1200.0,
    timestamp=_ts(10.5),
)
a.tool_call(
    "run_tests",
    node_name="coder",
    parent_span_id=coder_span_2,
    tool_input={"command": "pytest tests/ -x"},  # <-- identical input
    tool_output="5 passed",
    duration_ms=1100.0,
    timestamp=_ts(12.0),
)
a.node_exit("coder", span_id=coder_span_2, duration_ms=6500.0, timestamp=_ts(12.5))


# === Node 3: reviewer — produces output but state unchanged ==================
# The reviewer analyzes the code but its output doesn't lead to any state
# changes. OrchEval flags this as "output_not_utilized".

reviewer_span = "span-reviewer"
a.node_entry(
    "reviewer",
    span_id=reviewer_span,
    input_state={"code": "def process(data): ...", "tests_passed": True},
    timestamp=_ts(13.0),
)
a.llm_call(
    node_name="reviewer",
    parent_span_id=reviewer_span,
    model="gpt-4o-mini",
    input_tokens=500,
    output_tokens=300,
    cost=0.004,
    duration_ms=1500.0,
    timestamp=_ts(13.5),
    prompt_summary="Review this code for quality issues",
    response_summary="The code is acceptable but could use better error handling",
    output_message={"role": "ai", "content": "The code is acceptable but could use better error handling"},
)
a.node_exit(
    "reviewer",
    span_id=reviewer_span,
    duration_ms=2000.0,
    timestamp=_ts(15.0),
    # State is unchanged — the reviewer's suggestions weren't applied
    output_state={"code": "def process(data): ...", "tests_passed": True},
    state_diff={"added": [], "removed": [], "modified": []},
)


# === Collect and diagnose =====================================================

trace = tracer.collect()
full = report(trace)

# --- Print all detected anomalies ---
print("=" * 60)
print("ORCHEVAL DIAGNOSIS: bad-architecture-demo")
print("=" * 60)

# Routing flags
if full.routing.flags:
    print(f"\n--- Routing anomalies ({len(full.routing.flags)} detected) ---")
    for flag in full.routing.flags:
        print(f"  [{flag.flag_type}] {flag.description}")
else:
    print("\n--- No routing anomalies ---")

# Retry sequences
if full.retries.retry_sequences:
    print(f"\n--- Retry sequences ({len(full.retries.retry_sequences)} detected) ---")
    for seq in full.retries.retry_sequences:
        status = "succeeded" if seq.succeeded else "FAILED"
        print(f"  {seq.node_name}: {seq.attempt_count} attempts, {status}")
        for err in seq.errors:
            print(f"    Error: {err}")
else:
    print("\n--- No retry sequences ---")

# LLM behavioral patterns
if full.llm_patterns.patterns:
    print(f"\n--- LLM patterns ({len(full.llm_patterns.patterns)} detected) ---")
    for pat in full.llm_patterns.patterns:
        print(f"  [{pat.severity.upper()}] {pat.pattern_type}")
        print(f"    Node: {pat.node_name}")
        print(f"    {pat.description}")
else:
    print("\n--- No LLM patterns ---")

# Error summary
if full.retries.total_errors > 0:
    print(f"\n--- Errors ({full.retries.total_errors} total) ---")
    for cluster in full.retries.error_clusters:
        print(f"  {cluster.error_type}: {cluster.count}x in {cluster.nodes}")
        for msg in cluster.messages:
            print(f"    \"{msg}\"")

# --- Focused digest ---
# When debugging, you can zoom in on a specific node. Other nodes are collapsed
# into a single summary line, keeping the output focused.
print("\n" + "=" * 60)
print("FOCUSED DIGEST: coder node only")
print("=" * 60)
print(trace.to_digest(focus_nodes=["coder"], reports=full))

# --- Export ---
trace.to_html("bad_architecture.html", reports=full)
print("\nWrote: orcheval_outputs/bad_architecture.html")
print("Open it in a browser to see the waterfall timeline.")
