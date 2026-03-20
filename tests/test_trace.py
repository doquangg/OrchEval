"""Tests for orcheval.trace — trace container."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orcheval.events import (
    LLMCall,
    NodeEntry,
    NodeExit,
)
from orcheval.trace import Trace

TRACE_ID = "test-trace-container"
BASE_TIME = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)


def _ts(seconds: float) -> datetime:
    return BASE_TIME + timedelta(seconds=seconds)


class TestTraceBasics:
    def test_sorts_events_by_timestamp(self) -> None:
        e1 = NodeEntry(trace_id=TRACE_ID, node_name="a", timestamp=_ts(2))
        e2 = NodeEntry(trace_id=TRACE_ID, node_name="b", timestamp=_ts(1))
        trace = Trace(events=[e1, e2])
        assert trace[0].node_name == "b"
        assert trace[1].node_name == "a"

    def test_len(self, sample_trace: Trace) -> None:
        assert len(sample_trace) == 7

    def test_iter(self, sample_trace: Trace) -> None:
        events = list(sample_trace)
        assert len(events) == 7

    def test_getitem_int(self, sample_trace: Trace) -> None:
        event = sample_trace[0]
        assert isinstance(event, NodeEntry)

    def test_getitem_slice(self, sample_trace: Trace) -> None:
        events = sample_trace[0:2]
        assert len(events) == 2

    def test_bool_nonempty(self, sample_trace: Trace) -> None:
        assert sample_trace

    def test_bool_empty(self) -> None:
        assert not Trace(events=[])

    def test_trace_id_from_constructor(self) -> None:
        trace = Trace(events=[], trace_id="custom-id")
        assert trace.trace_id == "custom-id"

    def test_trace_id_from_events(self) -> None:
        e = NodeEntry(trace_id="from-event", node_name="a")
        trace = Trace(events=[e])
        assert trace.trace_id == "from-event"

    def test_events_returns_copy(self, sample_trace: Trace) -> None:
        events = sample_trace.events
        events.clear()
        assert len(sample_trace) == 7  # original unchanged


class TestTraceQuery:
    def test_get_events_by_type(self, sample_trace: Trace) -> None:
        llm_calls = sample_trace.get_events_by_type(LLMCall)
        assert len(llm_calls) == 2
        assert all(isinstance(e, LLMCall) for e in llm_calls)

    def test_get_events_by_node(self, sample_trace: Trace) -> None:
        agent_events = sample_trace.get_events_by_node("agent")
        assert len(agent_events) == 4  # entry, llm, tool, exit

    def test_get_events_by_node_no_match(self, sample_trace: Trace) -> None:
        events = sample_trace.get_events_by_node("nonexistent")
        assert events == []

    def test_get_llm_calls(self, sample_trace: Trace) -> None:
        calls = sample_trace.get_llm_calls()
        assert len(calls) == 2

    def test_get_tool_calls(self, sample_trace: Trace) -> None:
        calls = sample_trace.get_tool_calls()
        assert len(calls) == 1
        assert calls[0].tool_name == "search"

    def test_get_timeline(self, sample_trace: Trace) -> None:
        timeline = sample_trace.get_timeline()
        assert len(timeline) == 7
        for i in range(len(timeline) - 1):
            assert timeline[i].timestamp <= timeline[i + 1].timestamp


class TestTraceSummary:
    def test_total_duration(self, sample_trace: Trace) -> None:
        duration = sample_trace.total_duration()
        assert duration is not None
        assert duration == 6000.0  # 6 seconds in ms

    def test_total_duration_empty(self) -> None:
        trace = Trace(events=[])
        assert trace.total_duration() is None

    def test_total_duration_single_event(self) -> None:
        trace = Trace(events=[NodeEntry(trace_id=TRACE_ID, node_name="a")])
        assert trace.total_duration() is None

    def test_total_cost(self, sample_trace: Trace) -> None:
        cost = sample_trace.total_cost()
        assert cost is not None
        assert abs(cost - 0.007) < 1e-9  # 0.005 + 0.002

    def test_total_cost_no_costs(self) -> None:
        events = [LLMCall(trace_id=TRACE_ID)]
        trace = Trace(events=events)
        assert trace.total_cost() is None

    def test_total_tokens(self, sample_trace: Trace) -> None:
        tokens = sample_trace.total_tokens()
        assert tokens["prompt"] == 350  # 150 + 200
        assert tokens["completion"] == 180  # 80 + 100
        assert tokens["total"] == 530

    def test_total_tokens_empty(self) -> None:
        trace = Trace(events=[])
        tokens = trace.total_tokens()
        assert tokens == {"prompt": 0, "completion": 0, "total": 0}

    def test_node_durations(self, sample_trace: Trace) -> None:
        durations = sample_trace.node_durations()
        assert durations["agent"] == 3000.0
        assert durations["summarizer"] == 2000.0

    def test_node_durations_computed_from_timestamps(self) -> None:
        """When duration_ms is None, compute from entry/exit timestamps."""
        span = "span-1"
        events = [
            NodeEntry(trace_id=TRACE_ID, span_id=span, node_name="a", timestamp=_ts(0)),
            NodeExit(trace_id=TRACE_ID, span_id=span, node_name="a", timestamp=_ts(2)),
        ]
        trace = Trace(events=events)
        durations = trace.node_durations()
        assert durations["a"] == 2000.0

    def test_node_sequence(self, sample_trace: Trace) -> None:
        seq = sample_trace.node_sequence()
        assert seq == ["agent", "summarizer"]


class TestTraceMerge:
    def test_merge_two_traces(self) -> None:
        e1 = NodeEntry(trace_id="t1", node_name="a", timestamp=_ts(0))
        e2 = NodeEntry(trace_id="t2", node_name="b", timestamp=_ts(1))
        t1 = Trace(events=[e1], trace_id="t1")
        t2 = Trace(events=[e2], trace_id="t2")
        merged = Trace.merge(t1, t2)
        assert len(merged) == 2
        assert merged[0].node_name == "a"
        assert merged[1].node_name == "b"

    def test_merge_empty(self) -> None:
        merged = Trace.merge()
        assert len(merged) == 0

    def test_merge_preserves_sort(self) -> None:
        e1 = NodeEntry(trace_id="t1", node_name="a", timestamp=_ts(2))
        e2 = NodeEntry(trace_id="t2", node_name="b", timestamp=_ts(1))
        t1 = Trace(events=[e1])
        t2 = Trace(events=[e2])
        merged = Trace.merge(t1, t2)
        assert merged[0].node_name == "b"
