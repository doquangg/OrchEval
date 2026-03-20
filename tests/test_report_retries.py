"""Tests for the retries report module."""

from __future__ import annotations

from orcheval.events import ErrorEvent, NodeEntry, NodeExit
from orcheval.report.retries import UNKNOWN_NODE, RetryReport, retry_report
from orcheval.trace import Trace

from .conftest import TRACE_ID, _ts


class TestRetryReportBasic:
    def test_empty_trace(self) -> None:
        trace = Trace(events=[], trace_id=TRACE_ID)
        result = retry_report(trace)
        assert isinstance(result, RetryReport)
        assert result.error_clusters == []
        assert result.retry_sequences == []
        assert result.total_errors == 0
        assert result.overall_retry_success_rate is None

    def test_no_errors(self) -> None:
        events = [
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="agent"),
            NodeExit(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(1), node_name="agent"),
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = retry_report(trace)
        assert result.total_errors == 0
        assert result.retry_sequences == []


class TestErrorClustering:
    def test_errors_grouped_by_type(self, error_retry_events: list) -> None:
        trace = Trace(events=error_retry_events, trace_id=TRACE_ID)
        result = retry_report(trace)
        types = {c.error_type for c in result.error_clusters}
        assert types == {"SyntaxError", "ValueError"}

    def test_cluster_counts(self, error_retry_events: list) -> None:
        trace = Trace(events=error_retry_events, trace_id=TRACE_ID)
        result = retry_report(trace)
        syntax = next(c for c in result.error_clusters if c.error_type == "SyntaxError")
        assert syntax.count == 1
        assert syntax.messages == ["unexpected indent"]
        assert "codegen" in syntax.nodes

    def test_total_errors(self, error_retry_events: list) -> None:
        trace = Trace(events=error_retry_events, trace_id=TRACE_ID)
        result = retry_report(trace)
        assert result.total_errors == 2
        assert result.unique_error_types == 2

    def test_nodes_with_errors(self, error_retry_events: list) -> None:
        trace = Trace(events=error_retry_events, trace_id=TRACE_ID)
        result = retry_report(trace)
        assert set(result.nodes_with_errors) == {"codegen", "validator"}

    def test_message_deduplication(self) -> None:
        events = [
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="n"),
            ErrorEvent(
                trace_id=TRACE_ID, timestamp=_ts(1), node_name="n",
                error_type="Err", error_message="same msg",
            ),
            ErrorEvent(
                trace_id=TRACE_ID, timestamp=_ts(2), node_name="n",
                error_type="Err", error_message="same msg",
            ),
            ErrorEvent(
                trace_id=TRACE_ID, timestamp=_ts(3), node_name="n",
                error_type="Err", error_message="different msg",
            ),
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = retry_report(trace)
        cluster = result.error_clusters[0]
        assert cluster.count == 3
        assert len(cluster.messages) == 2  # deduplicated


class TestRetrySequenceDetection:
    def test_successful_retry_detected(self, error_retry_events: list) -> None:
        trace = Trace(events=error_retry_events, trace_id=TRACE_ID)
        result = retry_report(trace)
        cg_seqs = [s for s in result.retry_sequences if s.node_name == "codegen"]
        assert len(cg_seqs) == 1
        assert cg_seqs[0].succeeded is True
        assert cg_seqs[0].attempt_count == 2

    def test_no_retry_for_single_entry_node(self, error_retry_events: list) -> None:
        trace = Trace(events=error_retry_events, trace_id=TRACE_ID)
        result = retry_report(trace)
        # validator only has one entry, so no retry sequence
        val_seqs = [s for s in result.retry_sequences if s.node_name == "validator"]
        assert val_seqs == []

    def test_retry_errors_in_order(self, error_retry_events: list) -> None:
        trace = Trace(events=error_retry_events, trace_id=TRACE_ID)
        result = retry_report(trace)
        cg_seq = next(s for s in result.retry_sequences if s.node_name == "codegen")
        assert cg_seq.errors == ["unexpected indent"]

    def test_retry_duration(self, error_retry_events: list) -> None:
        trace = Trace(events=error_retry_events, trace_id=TRACE_ID)
        result = retry_report(trace)
        cg_seq = next(s for s in result.retry_sequences if s.node_name == "codegen")
        # First entry at _ts(0), last exit at _ts(5) -> 5000ms
        assert cg_seq.total_retry_duration_ms == 5000.0


class TestRetrySuccessRate:
    def test_success_rate(self, error_retry_events: list) -> None:
        trace = Trace(events=error_retry_events, trace_id=TRACE_ID)
        result = retry_report(trace)
        # Only codegen has a retry sequence, and it succeeded
        assert result.overall_retry_success_rate == 1.0

    def test_failed_retry(self) -> None:
        events = [
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="n"),
            ErrorEvent(
                trace_id=TRACE_ID, timestamp=_ts(1), node_name="n",
                error_type="Err", error_message="fail",
            ),
            NodeExit(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(2), node_name="n"),
            NodeEntry(trace_id=TRACE_ID, span_id="s2", timestamp=_ts(3), node_name="n"),
            ErrorEvent(
                trace_id=TRACE_ID, timestamp=_ts(4), node_name="n",
                error_type="Err", error_message="fail again",
            ),
            NodeExit(trace_id=TRACE_ID, span_id="s2", timestamp=_ts(5), node_name="n"),
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = retry_report(trace)
        assert len(result.retry_sequences) == 1
        assert result.retry_sequences[0].succeeded is False
        assert result.overall_retry_success_rate == 0.0


class TestRetryEdgeCases:
    def test_error_without_node_name(self) -> None:
        events = [
            ErrorEvent(
                trace_id=TRACE_ID, timestamp=_ts(0),
                error_type="RuntimeError", error_message="something broke",
            ),
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = retry_report(trace)
        assert result.total_errors == 1
        assert UNKNOWN_NODE in result.nodes_with_errors
        assert result.retry_sequences == []  # no retry without node context

