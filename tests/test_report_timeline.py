"""Tests for the timeline report module."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orcheval.events import LLMCall, NodeEntry, NodeExit, ToolCall
from orcheval.report.timeline import TimelineReport, timeline_report
from orcheval.trace import Trace

from .conftest import TRACE_ID, _ts


class TestTimelineReportBasic:
    def test_empty_trace(self) -> None:
        trace = Trace(events=[], trace_id=TRACE_ID)
        result = timeline_report(trace)
        assert isinstance(result, TimelineReport)
        assert result.spans == []
        assert result.events == []
        assert result.total_duration_ms is None
        assert result.start_time is None

    def test_total_duration_delegates_to_trace(self, sample_trace: Trace) -> None:
        result = timeline_report(sample_trace)
        assert result.total_duration_ms == sample_trace.total_duration()

    def test_start_and_end_time(self, sample_trace: Trace) -> None:
        result = timeline_report(sample_trace)
        assert result.start_time == _ts(0)
        assert result.end_time == _ts(6)


class TestTimelineSpans:
    def test_span_count_matches_nodes(self, sample_trace: Trace) -> None:
        result = timeline_report(sample_trace)
        assert len(result.spans) == 2  # agent and summarizer

    def test_span_node_names(self, sample_trace: Trace) -> None:
        result = timeline_report(sample_trace)
        names = [s.node_name for s in result.spans]
        assert names == ["agent", "summarizer"]

    def test_span_offsets(self, sample_trace: Trace) -> None:
        result = timeline_report(sample_trace)
        agent_span = result.spans[0]
        assert agent_span.start_ms == 0.0
        assert agent_span.end_ms == 3000.0  # _ts(3) - _ts(0)
        assert agent_span.duration_ms == 3000.0

    def test_span_children(self, sample_trace: Trace) -> None:
        result = timeline_report(sample_trace)
        agent_span = result.spans[0]
        # agent has llm_call and tool_call as children
        child_types = [c.event_type for c in agent_span.children]
        assert "llm_call" in child_types
        assert "tool_call" in child_types

    def test_orphaned_entry_no_exit(self) -> None:
        events = [
            NodeEntry(
                trace_id=TRACE_ID, span_id="orphan", timestamp=_ts(0),
                node_name="broken_node",
            ),
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = timeline_report(trace)
        assert len(result.spans) == 1
        assert result.spans[0].end_ms is None
        assert result.spans[0].duration_ms is None


class TestTimelineEvents:
    def test_flat_event_count(self, sample_trace: Trace) -> None:
        result = timeline_report(sample_trace)
        # 7 events total in sample_trace
        assert len(result.events) == 7

    def test_event_offsets_non_negative(self, sample_trace: Trace) -> None:
        result = timeline_report(sample_trace)
        for event in result.events:
            assert event.offset_ms >= 0.0

    def test_event_offsets_monotonic(self, sample_trace: Trace) -> None:
        result = timeline_report(sample_trace)
        offsets = [e.offset_ms for e in result.events]
        assert offsets == sorted(offsets)


class TestTimelineEventSummaries:
    def test_node_entry_summary(self) -> None:
        events = [
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="agent"),
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = timeline_report(trace)
        assert result.events[0].summary == "Enter agent"

    def test_node_exit_summary_with_duration(self) -> None:
        events = [
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="agent"),
            NodeExit(
                trace_id=TRACE_ID, span_id="s1", timestamp=_ts(1), node_name="agent",
                duration_ms=1000.0,
            ),
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = timeline_report(trace)
        assert result.events[1].summary == "Exit agent (1000ms)"

    def test_llm_call_summary(self) -> None:
        events = [
            LLMCall(
                trace_id=TRACE_ID, timestamp=_ts(0),
                model="gpt-4o", input_tokens=100, output_tokens=50,
            ),
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = timeline_report(trace)
        assert result.events[0].summary == "LLM call to gpt-4o (100+50 tokens)"

    def test_tool_call_summary(self) -> None:
        events = [
            ToolCall(trace_id=TRACE_ID, timestamp=_ts(0), tool_name="search"),
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = timeline_report(trace)
        assert result.events[0].summary == "Tool call: search"

    def test_frozen_result(self, sample_trace: Trace) -> None:
        result = timeline_report(sample_trace)
        with pytest.raises(ValidationError):
            result.total_duration_ms = 999.0  # type: ignore[misc]
