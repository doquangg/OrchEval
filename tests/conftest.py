"""Shared fixtures for orcheval tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orcheval.events import (
    ErrorEvent,
    LLMCall,
    NodeEntry,
    NodeExit,
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
