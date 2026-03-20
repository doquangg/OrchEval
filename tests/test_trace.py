"""Tests for orcheval.trace — trace container."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from orcheval.events import (
    AgentMessage,
    ErrorEvent,
    LLMCall,
    NodeEntry,
    NodeExit,
    PassBoundary,
    RoutingDecision,
    ToolCall,
)
from orcheval.trace import NodeInvocation, Trace

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


class TestNodeInvocations:
    def test_basic(self, sample_trace: Trace) -> None:
        invocations = sample_trace.node_invocations()
        assert len(invocations) == 2
        assert all(isinstance(inv, NodeInvocation) for inv in invocations)
        assert invocations[0].node_name == "agent"
        assert invocations[0].duration_ms == 3000.0
        assert invocations[1].node_name == "summarizer"
        assert invocations[1].duration_ms == 2000.0

    def test_repeated_node(self) -> None:
        """Node that executes 3 times returns 3 separate invocations."""
        events = []
        for i in range(3):
            span = f"span-{i}"
            events.append(
                NodeEntry(
                    trace_id=TRACE_ID, span_id=span, node_name="retry_node", timestamp=_ts(i * 2)
                )
            )
            events.append(
                NodeExit(
                    trace_id=TRACE_ID,
                    span_id=span,
                    node_name="retry_node",
                    timestamp=_ts(i * 2 + 1),
                    duration_ms=1000.0,
                )
            )
        trace = Trace(events=events)
        invocations = trace.node_invocations()
        assert len(invocations) == 3
        assert all(inv.node_name == "retry_node" for inv in invocations)
        span_ids = [inv.span_id for inv in invocations]
        assert len(set(span_ids)) == 3  # all distinct

    def test_timestamp_fallback(self) -> None:
        """When duration_ms is None, compute from entry/exit timestamps."""
        span = "span-ts"
        events = [
            NodeEntry(trace_id=TRACE_ID, span_id=span, node_name="a", timestamp=_ts(0)),
            NodeExit(trace_id=TRACE_ID, span_id=span, node_name="a", timestamp=_ts(2)),
        ]
        trace = Trace(events=events)
        invocations = trace.node_invocations()
        assert len(invocations) == 1
        assert invocations[0].duration_ms == 2000.0

    def test_empty_trace(self) -> None:
        trace = Trace(events=[])
        assert trace.node_invocations() == []

    def test_orphaned_exit(self) -> None:
        """NodeExit without matching NodeEntry gets duration=None."""
        events = [
            NodeExit(trace_id=TRACE_ID, span_id="orphan", node_name="a", timestamp=_ts(0)),
        ]
        trace = Trace(events=events)
        invocations = trace.node_invocations()
        assert len(invocations) == 1
        assert invocations[0].duration_ms is None

    def test_named_tuple_fields(self) -> None:
        """Verify NodeInvocation fields are accessible by name."""
        inv = NodeInvocation(node_name="test", span_id="s1", duration_ms=100.0)
        assert inv.node_name == "test"
        assert inv.span_id == "s1"
        assert inv.duration_ms == 100.0


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


class TestTraceSerialization:
    def test_to_dict_structure(self, sample_trace: Trace) -> None:
        d = sample_trace.to_dict()
        assert "trace_id" in d
        assert "events" in d
        assert isinstance(d["events"], list)
        assert len(d["events"]) == 7
        assert all("event_type" in e for e in d["events"])

    def test_from_dict_round_trip(self, sample_trace: Trace) -> None:
        d = sample_trace.to_dict()
        restored = Trace.from_dict(d)
        assert len(restored) == len(sample_trace)
        assert restored.trace_id == sample_trace.trace_id
        for orig, rest in zip(sample_trace, restored, strict=True):
            assert type(orig) is type(rest)
            assert orig.span_id == rest.span_id

    def test_to_json_returns_valid_json(self, sample_trace: Trace) -> None:
        j = sample_trace.to_json()
        assert isinstance(j, str)
        parsed = json.loads(j)
        assert "trace_id" in parsed
        assert "events" in parsed

    def test_from_json_round_trip(self, sample_trace: Trace) -> None:
        j = sample_trace.to_json()
        restored = Trace.from_json(j)
        assert len(restored) == len(sample_trace)
        assert restored.trace_id == sample_trace.trace_id
        for orig, rest in zip(sample_trace, restored, strict=True):
            assert type(orig) is type(rest)

    def test_all_event_types_round_trip(self) -> None:
        """Trace with all 8 event types survives JSON round-trip."""
        events = [
            NodeEntry(trace_id=TRACE_ID, node_name="n", timestamp=_ts(0)),
            NodeExit(trace_id=TRACE_ID, node_name="n", timestamp=_ts(1)),
            LLMCall(trace_id=TRACE_ID, model="gpt-4o", timestamp=_ts(2)),
            ToolCall(trace_id=TRACE_ID, tool_name="t", timestamp=_ts(3)),
            RoutingDecision(
                trace_id=TRACE_ID, source_node="a", target_node="b", timestamp=_ts(4)
            ),
            AgentMessage(trace_id=TRACE_ID, sender="a", receiver="b", timestamp=_ts(5)),
            ErrorEvent(
                trace_id=TRACE_ID, error_type="E", error_message="msg", timestamp=_ts(6)
            ),
            PassBoundary(
                trace_id=TRACE_ID, pass_number=1, direction="enter", timestamp=_ts(7)
            ),
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        restored = Trace.from_json(trace.to_json())
        assert len(restored) == 8
        for orig, rest in zip(trace, restored, strict=True):
            assert type(orig) is type(rest)
            assert orig.event_type == rest.event_type

    def test_timestamps_preserved(self) -> None:
        """Timezone-aware datetimes survive serialization."""
        event = NodeEntry(trace_id=TRACE_ID, node_name="a", timestamp=_ts(42))
        trace = Trace(events=[event], trace_id=TRACE_ID)
        restored = Trace.from_json(trace.to_json())
        assert restored[0].timestamp == event.timestamp
        assert restored[0].timestamp.tzinfo is not None

    def test_empty_trace_round_trip(self) -> None:
        trace = Trace(events=[], trace_id="empty")
        restored = Trace.from_json(trace.to_json())
        assert len(restored) == 0
        assert restored.trace_id == "empty"

    def test_invalid_event_raises(self) -> None:
        data = {"trace_id": TRACE_ID, "events": [{"event_type": "nonexistent"}]}
        with pytest.raises(ValidationError):
            Trace.from_dict(data)
