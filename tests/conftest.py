"""Shared fixtures for orcheval tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orcheval.events import (
    ErrorEvent,
    LLMCall,
    NodeEntry,
    NodeExit,
    PassBoundary,
    RoutingDecision,
    ToolCall,
)
from orcheval.trace import Trace

TRACE_ID = "test-trace-0001"

# Fixed base time for deterministic timestamps
BASE_TIME = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)


def _ts(seconds: float) -> datetime:
    """Helper: offset from BASE_TIME by the given number of seconds."""
    return BASE_TIME + timedelta(seconds=seconds)


@pytest.fixture
def trace_id() -> str:
    return TRACE_ID


@pytest.fixture
def sample_events() -> list[NodeEntry | NodeExit | LLMCall | ToolCall | ErrorEvent]:
    """A realistic sequence of events from a 2-node graph run.

    Simulates: node_entry(agent) -> llm_call -> tool_call -> node_exit(agent)
               -> node_entry(summarizer) -> llm_call -> node_exit(summarizer)
    """
    span_agent = "span-agent-001"
    span_summarizer = "span-summarizer-002"

    return [
        NodeEntry(
            trace_id=TRACE_ID,
            span_id=span_agent,
            timestamp=_ts(0),
            node_name="agent",
        ),
        LLMCall(
            trace_id=TRACE_ID,
            span_id="span-llm-001",
            parent_span_id=span_agent,
            timestamp=_ts(1),
            node_name="agent",
            model="gpt-4o",
            input_tokens=150,
            output_tokens=80,
            cost=0.005,
            duration_ms=800.0,
            prompt_summary="Analyze the data",
            response_summary="I'll use the search tool",
            input_messages=[{"role": "user", "content": "Analyze the data"}],
            output_message={"role": "ai", "content": "I'll use the search tool"},
        ),
        ToolCall(
            trace_id=TRACE_ID,
            span_id="span-tool-001",
            parent_span_id=span_agent,
            timestamp=_ts(2),
            node_name="agent",
            tool_name="search",
            tool_input={"query": "test data"},
            tool_output="Found 3 results",
            duration_ms=500.0,
        ),
        NodeExit(
            trace_id=TRACE_ID,
            span_id=span_agent,
            timestamp=_ts(3),
            node_name="agent",
            duration_ms=3000.0,
        ),
        NodeEntry(
            trace_id=TRACE_ID,
            span_id=span_summarizer,
            timestamp=_ts(4),
            node_name="summarizer",
        ),
        LLMCall(
            trace_id=TRACE_ID,
            span_id="span-llm-002",
            parent_span_id=span_summarizer,
            timestamp=_ts(5),
            node_name="summarizer",
            model="gpt-4o-mini",
            input_tokens=200,
            output_tokens=100,
            cost=0.002,
            duration_ms=600.0,
            prompt_summary="Summarize findings",
            response_summary="The analysis shows...",
            input_messages=[{"role": "user", "content": "Summarize findings"}],
            output_message={"role": "ai", "content": "The analysis shows..."},
        ),
        NodeExit(
            trace_id=TRACE_ID,
            span_id=span_summarizer,
            timestamp=_ts(6),
            node_name="summarizer",
            duration_ms=2000.0,
        ),
    ]


@pytest.fixture
def sample_trace(sample_events: list[NodeEntry | NodeExit | LLMCall | ToolCall]) -> Trace:
    return Trace(events=sample_events, trace_id=TRACE_ID)


# --- Report-layer fixtures ---


@pytest.fixture
def routing_events() -> list[RoutingDecision]:
    """Routing decisions covering multiple flag scenarios.

    - router_a: always routes to "node_x" (invariant routing, 3 decisions)
    - router_b: routes to "node_y" and "node_z" with identical context (context divergence)
    - router_c: routes to "node_p" 19 times and "node_q" once (dominant path at 95%)
    """
    events: list[RoutingDecision] = []

    # Invariant routing: router_a -> node_x (3 times, different contexts)
    for i in range(3):
        events.append(RoutingDecision(
            trace_id=TRACE_ID,
            timestamp=_ts(i),
            node_name="router_a",
            source_node="router_a",
            target_node="node_x",
            decision_context={"score": i * 10},
        ))

    # Context divergence: router_b with same context -> different targets
    events.append(RoutingDecision(
        trace_id=TRACE_ID,
        timestamp=_ts(10),
        node_name="router_b",
        source_node="router_b",
        target_node="node_y",
        decision_context={"quality": "high", "count": 5},
    ))
    events.append(RoutingDecision(
        trace_id=TRACE_ID,
        timestamp=_ts(11),
        node_name="router_b",
        source_node="router_b",
        target_node="node_z",
        decision_context={"quality": "high", "count": 5},
    ))

    # Dominant path: router_c -> node_p (19x) and node_q (1x)
    for i in range(19):
        events.append(RoutingDecision(
            trace_id=TRACE_ID,
            timestamp=_ts(20 + i),
            node_name="router_c",
            source_node="router_c",
            target_node="node_p",
            decision_context={"iteration": i},
        ))
    events.append(RoutingDecision(
        trace_id=TRACE_ID,
        timestamp=_ts(39),
        node_name="router_c",
        source_node="router_c",
        target_node="node_q",
        decision_context={"iteration": 19},
    ))

    return events


@pytest.fixture
def oscillation_events() -> list[RoutingDecision]:
    """Routing decisions showing oscillation between two targets."""
    events: list[RoutingDecision] = []
    targets = ["node_a", "node_b"]
    for i in range(8):
        events.append(RoutingDecision(
            trace_id=TRACE_ID,
            timestamp=_ts(i),
            node_name="oscillator",
            source_node="oscillator",
            target_node=targets[i % 2],
            decision_context={"step": i},
        ))
    return events


@pytest.fixture
def multipass_events() -> list[PassBoundary]:
    """Three passes with improving metrics (converging)."""
    events: list[PassBoundary] = []
    metrics = [
        {"violation_count": 50, "quality_score": 0.3, "rows_remaining": 1000},
        {"violation_count": 30, "quality_score": 0.5, "rows_remaining": 800},
        {"violation_count": 20, "quality_score": 0.6, "rows_remaining": 600},
        {"violation_count": 15, "quality_score": 0.65, "rows_remaining": 500},
        {"violation_count": 12, "quality_score": 0.68, "rows_remaining": 450},
        {"violation_count": 10, "quality_score": 0.7, "rows_remaining": 400},
    ]
    for i in range(3):
        events.append(PassBoundary(
            trace_id=TRACE_ID,
            timestamp=_ts(i * 10),
            pass_number=i + 1,
            direction="enter",
            metrics_snapshot=metrics[i * 2],
        ))
        events.append(PassBoundary(
            trace_id=TRACE_ID,
            timestamp=_ts(i * 10 + 5),
            pass_number=i + 1,
            direction="exit",
            metrics_snapshot=metrics[i * 2 + 1],
        ))
    return events


@pytest.fixture
def error_retry_events() -> list[NodeEntry | NodeExit | ErrorEvent | LLMCall]:
    """Events showing retry behavior.

    - codegen: enters, errors, re-enters (retry), succeeds
    - validator: enters, errors, never retried
    """
    span_cg1 = "span-cg-001"
    span_cg2 = "span-cg-002"
    span_val = "span-val-001"

    return [
        # codegen attempt 1: enter -> error
        NodeEntry(
            trace_id=TRACE_ID, span_id=span_cg1, timestamp=_ts(0), node_name="codegen",
        ),
        ErrorEvent(
            trace_id=TRACE_ID, span_id="span-err-001", parent_span_id=span_cg1,
            timestamp=_ts(1), node_name="codegen",
            error_type="SyntaxError", error_message="unexpected indent",
        ),
        NodeExit(
            trace_id=TRACE_ID, span_id=span_cg1, timestamp=_ts(2), node_name="codegen",
            duration_ms=2000.0,
        ),
        # codegen attempt 2 (retry): enter -> llm_call -> exit (success)
        NodeEntry(
            trace_id=TRACE_ID, span_id=span_cg2, timestamp=_ts(3), node_name="codegen",
        ),
        LLMCall(
            trace_id=TRACE_ID, span_id="span-llm-retry", parent_span_id=span_cg2,
            timestamp=_ts(4), node_name="codegen",
            model="gpt-4o", input_tokens=100, output_tokens=50,
            cost=0.003, duration_ms=500.0,
        ),
        NodeExit(
            trace_id=TRACE_ID, span_id=span_cg2, timestamp=_ts(5), node_name="codegen",
            duration_ms=2000.0,
        ),
        # validator: enter -> error -> exit (no retry)
        NodeEntry(
            trace_id=TRACE_ID, span_id=span_val, timestamp=_ts(6), node_name="validator",
        ),
        ErrorEvent(
            trace_id=TRACE_ID, span_id="span-err-002", parent_span_id=span_val,
            timestamp=_ts(7), node_name="validator",
            error_type="ValueError", error_message="invalid schema",
        ),
        NodeExit(
            trace_id=TRACE_ID, span_id=span_val, timestamp=_ts(8), node_name="validator",
            duration_ms=2000.0,
        ),
    ]


@pytest.fixture
def multi_model_events() -> list[NodeEntry | NodeExit | LLMCall]:
    """LLM calls from 2 nodes using different models."""
    span_a = "span-node-a"
    span_b = "span-node-b"

    return [
        NodeEntry(trace_id=TRACE_ID, span_id=span_a, timestamp=_ts(0), node_name="planner"),
        LLMCall(
            trace_id=TRACE_ID, span_id="span-llm-a1", parent_span_id=span_a,
            timestamp=_ts(1), node_name="planner",
            model="gpt-4o", input_tokens=200, output_tokens=100,
            cost=0.008, duration_ms=1000.0,
        ),
        LLMCall(
            trace_id=TRACE_ID, span_id="span-llm-a2", parent_span_id=span_a,
            timestamp=_ts(2), node_name="planner",
            model="gpt-4o-mini", input_tokens=50, output_tokens=30,
            cost=0.001, duration_ms=300.0,
        ),
        NodeExit(trace_id=TRACE_ID, span_id=span_a, timestamp=_ts(3), node_name="planner",
                 duration_ms=3000.0),
        NodeEntry(trace_id=TRACE_ID, span_id=span_b, timestamp=_ts(4), node_name="executor"),
        LLMCall(
            trace_id=TRACE_ID, span_id="span-llm-b1", parent_span_id=span_b,
            timestamp=_ts(5), node_name="executor",
            model="gpt-4o", input_tokens=300, output_tokens=150,
            cost=0.012, duration_ms=1200.0,
        ),
        NodeExit(trace_id=TRACE_ID, span_id=span_b, timestamp=_ts(6), node_name="executor",
                 duration_ms=2000.0),
    ]


@pytest.fixture
def stateful_trace() -> Trace:
    """Trace with state capture data for testing state-aware rendering."""
    span_a = "span-state-a"
    span_b = "span-state-b"

    return Trace(events=[
        NodeEntry(
            trace_id=TRACE_ID, span_id=span_a, timestamp=_ts(0),
            node_name="preprocessor",
            input_state={"data": [1, 2, 3], "config": {"mode": "fast"}},
        ),
        NodeExit(
            trace_id=TRACE_ID, span_id=span_a, timestamp=_ts(2),
            node_name="preprocessor", duration_ms=2000.0,
            output_state={"data": [1, 2, 3], "config": {"mode": "fast"}, "result": "done"},
            state_diff={"added": ["result"], "removed": [], "modified": []},
        ),
        NodeEntry(
            trace_id=TRACE_ID, span_id=span_b, timestamp=_ts(3),
            node_name="analyzer",
            input_state={"data": [1, 2, 3], "result": "done"},
        ),
        LLMCall(
            trace_id=TRACE_ID, span_id="span-llm-state", parent_span_id=span_b,
            timestamp=_ts(4), node_name="analyzer",
            model="gpt-4o", input_tokens=100, output_tokens=50,
            cost=0.003, duration_ms=500.0,
        ),
        NodeExit(
            trace_id=TRACE_ID, span_id=span_b, timestamp=_ts(5),
            node_name="analyzer", duration_ms=2000.0,
            output_state={"data": [1, 2, 3], "result": "analyzed", "score": 0.95},
            state_diff={"added": ["score"], "removed": [], "modified": ["result"]},
        ),
    ], trace_id=TRACE_ID)


@pytest.fixture
def llm_pattern_events() -> list[NodeEntry | NodeExit | LLMCall | ToolCall]:
    """Events exercising all five LLM pattern detectors.

    - agent: 2 invocations with prompt growth (100 -> 200 tokens, 100% growth),
      identical response_summary but different prompt_summary (repeated output),
      and redundant tool calls in invocation 2.
    - planner: 2 LLM calls with different system_message (system message variance).
    - formatter: 1 invocation with LLM output but empty state_diff (output not utilized).
    """
    span_agent_1 = "span-pat-agent-1"
    span_agent_2 = "span-pat-agent-2"
    span_planner = "span-pat-planner"
    span_formatter = "span-pat-formatter"

    return [
        # --- agent invocation 1 ---
        NodeEntry(
            trace_id=TRACE_ID, span_id=span_agent_1, timestamp=_ts(0),
            node_name="agent",
        ),
        LLMCall(
            trace_id=TRACE_ID, span_id="span-pat-llm-a1", parent_span_id=span_agent_1,
            timestamp=_ts(1), node_name="agent",
            model="gpt-4o", input_tokens=100, output_tokens=50,
            cost=0.003, duration_ms=500.0,
            prompt_summary="Analyze the dataset",
            response_summary="I'll use the search tool",
            input_messages=[{"role": "user", "content": "Analyze the dataset"}],
            output_message={"role": "ai", "content": "I'll use the search tool"},
        ),
        NodeExit(
            trace_id=TRACE_ID, span_id=span_agent_1, timestamp=_ts(2),
            node_name="agent", duration_ms=2000.0,
        ),
        # --- agent invocation 2 (retry) ---
        NodeEntry(
            trace_id=TRACE_ID, span_id=span_agent_2, timestamp=_ts(3),
            node_name="agent",
        ),
        LLMCall(
            trace_id=TRACE_ID, span_id="span-pat-llm-a2", parent_span_id=span_agent_2,
            timestamp=_ts(4), node_name="agent",
            model="gpt-4o", input_tokens=200, output_tokens=50,
            cost=0.005, duration_ms=600.0,
            prompt_summary="Analyze the dataset with error context appended",
            response_summary="I'll use the search tool",
            input_messages=[{"role": "user", "content": "Analyze the dataset with error context"}],
            output_message={"role": "ai", "content": "I'll use the search tool"},
        ),
        # Redundant tool calls in invocation 2
        ToolCall(
            trace_id=TRACE_ID, span_id="span-pat-tool-a2a", parent_span_id=span_agent_2,
            timestamp=_ts(4.5), node_name="agent",
            tool_name="search", tool_input={"query": "test data"},
            tool_output="Found 3 results", duration_ms=200.0,
        ),
        ToolCall(
            trace_id=TRACE_ID, span_id="span-pat-tool-a2b", parent_span_id=span_agent_2,
            timestamp=_ts(4.7), node_name="agent",
            tool_name="search", tool_input={"query": "test data"},
            tool_output="Found 3 results", duration_ms=200.0,
        ),
        NodeExit(
            trace_id=TRACE_ID, span_id=span_agent_2, timestamp=_ts(5),
            node_name="agent", duration_ms=2000.0,
        ),
        # --- planner: system message variance ---
        NodeEntry(
            trace_id=TRACE_ID, span_id=span_planner, timestamp=_ts(6),
            node_name="planner",
        ),
        LLMCall(
            trace_id=TRACE_ID, span_id="span-pat-llm-p1", parent_span_id=span_planner,
            timestamp=_ts(7), node_name="planner",
            model="gpt-4o", input_tokens=150, output_tokens=80,
            cost=0.004, duration_ms=700.0,
            system_message="You are a planning agent.",
        ),
        LLMCall(
            trace_id=TRACE_ID, span_id="span-pat-llm-p2", parent_span_id=span_planner,
            timestamp=_ts(8), node_name="planner",
            model="gpt-4o", input_tokens=150, output_tokens=80,
            cost=0.004, duration_ms=700.0,
            system_message="You are an expert planner. Be concise.",
        ),
        NodeExit(
            trace_id=TRACE_ID, span_id=span_planner, timestamp=_ts(9),
            node_name="planner", duration_ms=3000.0,
        ),
        # --- formatter: output not utilized ---
        NodeEntry(
            trace_id=TRACE_ID, span_id=span_formatter, timestamp=_ts(10),
            node_name="formatter",
            input_state={"data": [1, 2, 3]},
        ),
        LLMCall(
            trace_id=TRACE_ID, span_id="span-pat-llm-f1", parent_span_id=span_formatter,
            timestamp=_ts(11), node_name="formatter",
            model="gpt-4o", input_tokens=80, output_tokens=40,
            cost=0.002, duration_ms=400.0,
            response_summary="Formatted output ready",
            output_message={"role": "ai", "content": "Formatted output ready"},
        ),
        NodeExit(
            trace_id=TRACE_ID, span_id=span_formatter, timestamp=_ts(12),
            node_name="formatter", duration_ms=2000.0,
            output_state={"data": [1, 2, 3]},
            state_diff={"added": [], "removed": [], "modified": []},
        ),
    ]


@pytest.fixture
def llm_pattern_trace(llm_pattern_events: list) -> Trace:
    return Trace(events=llm_pattern_events, trace_id=TRACE_ID)
