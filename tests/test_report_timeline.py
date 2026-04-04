"""Tests for the timeline report module."""

from __future__ import annotations

from orcheval.events import NodeEntry
from orcheval.report.timeline import TimelineReport, timeline_report
from orcheval.trace import Trace

from .conftest import TRACE_ID, _ts


class TestTimelineReportBasic:
    def test_empty_trace(self) -> None:
        trace = Trace(events=[], trace_id=TRACE_ID)
        result = timeline_report(trace)
        assert isinstance(result, TimelineReport)
        assert result.spans == []
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


class TestTimelineEventSummaries:
    def test_child_event_summaries(self, sample_trace: Trace) -> None:
        result = timeline_report(sample_trace)
        agent_span = result.spans[0]
        summaries = [c.summary for c in agent_span.children]
        assert any("LLM call" in s for s in summaries)
        assert any("Tool call" in s for s in summaries)


class TestTimelineEventsProperty:
    def test_events_empty_report(self) -> None:
        report = TimelineReport()
        assert report.events == []

    def test_events_flattens_span_children(self, sample_trace: Trace) -> None:
        result = timeline_report(sample_trace)
        total_children = sum(len(s.children) for s in result.spans)
        assert len(result.events) == total_children

    def test_events_sorted_by_offset(self, sample_trace: Trace) -> None:
        result = timeline_report(sample_trace)
        offsets = [e.offset_ms for e in result.events]
        assert offsets == sorted(offsets)

