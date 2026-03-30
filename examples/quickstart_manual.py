"""OrchEval quickstart — build a trace, generate a report, export HTML.

This example simulates a 3-node pipeline (planner → coder → reviewer) using
the ManualAdapter. No LLM API keys or framework dependencies required.

Run:
    python examples/quickstart_manual.py

Outputs:
    orcheval_outputs/quickstart.html    — interactive waterfall visualization
    orcheval_outputs/quickstart.json    — serialized trace (reload later)
    Printed: text digest, cost breakdown, node durations
"""

from datetime import datetime, timedelta, timezone

from orcheval import Tracer, report

# Deterministic timestamps so the trace has realistic wall-clock duration.
# Without explicit timestamps, ManualAdapter events all land at "now" and
# total_duration() returns ~0ms. Real adapters (LangGraph, OpenAI Agents)
# set timestamps from actual callback times, so this isn't an issue there.
_BASE = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _ts(seconds: float) -> datetime:
    return _BASE + timedelta(seconds=seconds)


# --- Build a trace -----------------------------------------------------------
# The ManualAdapter is the default. You emit events directly — useful for
# frameworks without a built-in adapter, or for testing.

tracer = Tracer(trace_id="quickstart-demo")
a = tracer.adapter

# Node 1: planner — one LLM call that decides what to build
planner_span = "span-planner"
a.node_entry("planner", span_id=planner_span, timestamp=_ts(0))
a.llm_call(
    node_name="planner",
    parent_span_id=planner_span,
    model="gpt-4o",
    input_tokens=320,
    output_tokens=150,
    cost=0.008,
    duration_ms=1200.0,
    timestamp=_ts(0.3),
    system_message="You are a planning agent. Break tasks into steps.",
    input_messages=[{"role": "user", "content": "Build a REST API for user management"}],
    output_message={"role": "ai", "content": "Plan: 1) Define schema 2) Write endpoints 3) Add auth"},
    prompt_summary="Build a REST API for user management",
    response_summary="Plan: 1) Define schema 2) Write endpoints 3) Add auth",
)
a.node_exit("planner", span_id=planner_span, duration_ms=1500.0, timestamp=_ts(1.5))

# Node 2: coder — one LLM call + one tool call (code execution)
coder_span = "span-coder"
a.node_entry("coder", span_id=coder_span, timestamp=_ts(1.8))
a.llm_call(
    node_name="coder",
    parent_span_id=coder_span,
    model="gpt-4o",
    input_tokens=800,
    output_tokens=400,
    cost=0.018,
    duration_ms=3200.0,
    timestamp=_ts(2.0),
    input_messages=[{"role": "user", "content": "Implement the REST API based on this plan..."}],
    output_message={"role": "ai", "content": "```python\nfrom fastapi import FastAPI...```"},
    prompt_summary="Implement the REST API based on this plan...",
    response_summary="from fastapi import FastAPI...",
)
a.tool_call(
    "execute_code",
    node_name="coder",
    parent_span_id=coder_span,
    tool_input={"code": "python main.py --test"},
    tool_output="All 5 tests passed",
    duration_ms=800.0,
    timestamp=_ts(5.5),
)
a.node_exit("coder", span_id=coder_span, duration_ms=4500.0, timestamp=_ts(6.3))

# Node 3: reviewer — one LLM call using a cheaper model
reviewer_span = "span-reviewer"
a.node_entry("reviewer", span_id=reviewer_span, timestamp=_ts(6.5))
a.llm_call(
    node_name="reviewer",
    parent_span_id=reviewer_span,
    model="gpt-4o-mini",
    input_tokens=600,
    output_tokens=200,
    cost=0.003,
    duration_ms=900.0,
    timestamp=_ts(6.8),
    input_messages=[{"role": "user", "content": "Review this FastAPI implementation..."}],
    output_message={"role": "ai", "content": "Code looks good. Minor: add input validation on POST /users."},
    prompt_summary="Review this FastAPI implementation...",
    response_summary="Code looks good. Minor: add input validation.",
)
a.node_exit("reviewer", span_id=reviewer_span, duration_ms=1200.0, timestamp=_ts(7.7))

# --- Collect and analyze -----------------------------------------------------

trace = tracer.collect()

# Basic summary methods on the Trace object
print("=== Trace Summary ===")
print(f"Trace ID:  {trace.trace_id}")
print(f"Nodes:     {trace.node_sequence()}")
print(f"Duration:  {trace.total_duration():.0f}ms")
print(f"Cost:      ${trace.total_cost():.4f}")
print(f"Tokens:    {trace.total_tokens()}")
print()

# Full report — runs all 6 analysis modules
full = report(trace)

print("=== Cost Breakdown ===")
print(f"Most expensive node:  {full.cost.most_expensive_node}")
print(f"Most expensive model: {full.cost.most_expensive_model}")
for node in full.cost.nodes:
    print(f"  {node.node_name}: ${node.total_cost:.4f} ({node.call_count} calls)")
print()

# Text digest — designed for feeding to another LLM
print("=== Text Digest ===")
print(trace.to_digest(reports=full))

# --- Export -------------------------------------------------------------------

# HTML waterfall — open in any browser
trace.to_html("quickstart.html", reports=full)
print("Wrote: orcheval_outputs/quickstart.html")

# Serialized trace — reload later with Trace.from_json_file()
trace.to_json("quickstart.json")
print("Wrote: orcheval_outputs/quickstart.json")

# Mermaid topology diagram — renders on GitHub natively
print("\n=== Mermaid Diagram ===")
print(trace.to_mermaid())
