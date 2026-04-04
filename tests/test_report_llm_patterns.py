"""Tests for the LLM call pattern analysis report module."""

from __future__ import annotations

from orcheval.events import LLMCall, NodeEntry, NodeExit, ToolCall
from orcheval.report import FullReport, report
from orcheval.report.llm_patterns import (
    LLMPatternsReport,
    llm_patterns_report,
)
from orcheval.trace import Trace

from .conftest import TRACE_ID, _ts

# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


class TestLLMPatternsReportBasic:
    def test_empty_trace(self) -> None:
        trace = Trace(events=[], trace_id=TRACE_ID)
        result = llm_patterns_report(trace)
        assert isinstance(result, LLMPatternsReport)
        assert result.patterns == []
        assert result.total_llm_calls == 0
        assert result.nodes_analyzed == 0
        assert result.node_summaries == []

    def test_no_llm_calls(self) -> None:
        trace = Trace(events=[
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="agent"),
            NodeExit(
                trace_id=TRACE_ID, span_id="s1", timestamp=_ts(1),
                node_name="agent", duration_ms=1000.0),
        ], trace_id=TRACE_ID)
        result = llm_patterns_report(trace)
        assert result.patterns == []
        assert result.total_llm_calls == 0

    def test_totals(self, llm_pattern_trace: Trace) -> None:
        result = llm_patterns_report(llm_pattern_trace)
        # agent: 2 LLM calls, planner: 2 LLM calls, formatter: 1 LLM call = 5 total
        assert result.total_llm_calls == 5
        # Nodes with LLM or tool calls: agent, planner, formatter
        assert result.nodes_analyzed == 3

    def test_single_invocation_no_patterns(self) -> None:
        """A node with only one invocation should not trigger prompt_growth or repeated_output."""
        trace = Trace(events=[
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="worker"),
            LLMCall(
                trace_id=TRACE_ID, span_id="llm1", parent_span_id="s1",
                timestamp=_ts(1), node_name="worker",
                model="gpt-4o", input_tokens=100, output_tokens=50,
            ),
            NodeExit(
                trace_id=TRACE_ID, span_id="s1", timestamp=_ts(2),
                node_name="worker", duration_ms=2000.0),
        ], trace_id=TRACE_ID)
        result = llm_patterns_report(trace)
        growth = [p for p in result.patterns if p.pattern_type == "prompt_growth"]
        repeated = [p for p in result.patterns if p.pattern_type == "repeated_output"]
        assert growth == []
        assert repeated == []

    def test_unparented_llm_calls_counted(self) -> None:
        """LLM calls without matching parent_span_id still count in totals."""
        trace = Trace(events=[
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="agent"),
            LLMCall(
                trace_id=TRACE_ID, span_id="llm1", parent_span_id="s1",
                timestamp=_ts(1), node_name="agent",
                model="gpt-4o", input_tokens=100, output_tokens=50,
            ),
            LLMCall(
                trace_id=TRACE_ID, span_id="llm2", parent_span_id=None,
                timestamp=_ts(2), node_name="agent",
                model="gpt-4o", input_tokens=80, output_tokens=30,
            ),
            NodeExit(
                trace_id=TRACE_ID, span_id="s1", timestamp=_ts(3),
                node_name="agent", duration_ms=3000.0),
        ], trace_id=TRACE_ID)
        result = llm_patterns_report(trace)
        assert result.total_llm_calls == 2

    def test_no_false_positives_on_sample_trace(self, sample_trace: Trace) -> None:
        """sample_trace has clean, non-anomalous LLM calls. No patterns should fire."""
        result = llm_patterns_report(sample_trace)
        assert result.patterns == []


# ---------------------------------------------------------------------------
# Prompt growth
# ---------------------------------------------------------------------------


class TestPromptGrowthDetection:
    def test_growth_detected(self, llm_pattern_trace: Trace) -> None:
        result = llm_patterns_report(llm_pattern_trace)
        growth = [p for p in result.patterns if p.pattern_type == "prompt_growth"]
        assert len(growth) == 1
        assert growth[0].node_name == "agent"
        assert growth[0].severity == "warning"
        assert growth[0].evidence["growth_pct"] == 100.0
        assert growth[0].evidence["first_invocation_tokens"] == 100
        assert growth[0].evidence["last_invocation_tokens"] == 200

    def test_no_growth_below_threshold(self) -> None:
        """40% growth should not trigger (threshold is 50%)."""
        trace = Trace(events=[
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="x"),
            LLMCall(trace_id=TRACE_ID, span_id="l1", parent_span_id="s1",
                    timestamp=_ts(1), node_name="x", input_tokens=100, output_tokens=50),
            NodeExit(
                trace_id=TRACE_ID, span_id="s1", timestamp=_ts(2),
                node_name="x", duration_ms=2000.0),
            NodeEntry(trace_id=TRACE_ID, span_id="s2", timestamp=_ts(3), node_name="x"),
            LLMCall(trace_id=TRACE_ID, span_id="l2", parent_span_id="s2",
                    timestamp=_ts(4), node_name="x", input_tokens=140, output_tokens=50),
            NodeExit(
                trace_id=TRACE_ID, span_id="s2", timestamp=_ts(5),
                node_name="x", duration_ms=2000.0),
        ], trace_id=TRACE_ID)
        result = llm_patterns_report(trace)
        growth = [p for p in result.patterns if p.pattern_type == "prompt_growth"]
        assert growth == []

    def test_none_tokens_skipped(self) -> None:
        """Invocations where input_tokens is None should not cause errors."""
        trace = Trace(events=[
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="x"),
            LLMCall(trace_id=TRACE_ID, span_id="l1", parent_span_id="s1",
                    timestamp=_ts(1), node_name="x", input_tokens=None, output_tokens=50),
            NodeExit(
                trace_id=TRACE_ID, span_id="s1", timestamp=_ts(2),
                node_name="x", duration_ms=2000.0),
            NodeEntry(trace_id=TRACE_ID, span_id="s2", timestamp=_ts(3), node_name="x"),
            LLMCall(trace_id=TRACE_ID, span_id="l2", parent_span_id="s2",
                    timestamp=_ts(4), node_name="x", input_tokens=200, output_tokens=50),
            NodeExit(
                trace_id=TRACE_ID, span_id="s2", timestamp=_ts(5),
                node_name="x", duration_ms=2000.0),
        ], trace_id=TRACE_ID)
        result = llm_patterns_report(trace)
        # Only 1 valid token count, so no growth comparison possible
        growth = [p for p in result.patterns if p.pattern_type == "prompt_growth"]
        assert growth == []


# ---------------------------------------------------------------------------
# Repeated output
# ---------------------------------------------------------------------------


class TestRepeatedOutputDetection:
    def test_exact_match_detected(self, llm_pattern_trace: Trace) -> None:
        result = llm_patterns_report(llm_pattern_trace)
        repeated = [p for p in result.patterns if p.pattern_type == "repeated_output"]
        assert len(repeated) == 1
        assert repeated[0].node_name == "agent"
        assert repeated[0].severity == "warning"
        assert repeated[0].evidence["matching_output"] == "I'll use the search tool"

    def test_different_outputs_no_flag(self) -> None:
        trace = Trace(events=[
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="x"),
            LLMCall(trace_id=TRACE_ID, span_id="l1", parent_span_id="s1",
                    timestamp=_ts(1), node_name="x",
                    prompt_summary="prompt A", response_summary="output A"),
            NodeExit(
                trace_id=TRACE_ID, span_id="s1", timestamp=_ts(2),
                node_name="x", duration_ms=2000.0),
            NodeEntry(trace_id=TRACE_ID, span_id="s2", timestamp=_ts(3), node_name="x"),
            LLMCall(trace_id=TRACE_ID, span_id="l2", parent_span_id="s2",
                    timestamp=_ts(4), node_name="x",
                    prompt_summary="prompt B", response_summary="output B"),
            NodeExit(
                trace_id=TRACE_ID, span_id="s2", timestamp=_ts(5),
                node_name="x", duration_ms=2000.0),
        ], trace_id=TRACE_ID)
        result = llm_patterns_report(trace)
        repeated = [p for p in result.patterns if p.pattern_type == "repeated_output"]
        assert repeated == []

    def test_same_input_same_output_no_flag(self) -> None:
        """Same input producing same output is expected, not anomalous."""
        trace = Trace(events=[
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="x"),
            LLMCall(trace_id=TRACE_ID, span_id="l1", parent_span_id="s1",
                    timestamp=_ts(1), node_name="x",
                    prompt_summary="same prompt", response_summary="same output"),
            NodeExit(
                trace_id=TRACE_ID, span_id="s1", timestamp=_ts(2),
                node_name="x", duration_ms=2000.0),
            NodeEntry(trace_id=TRACE_ID, span_id="s2", timestamp=_ts(3), node_name="x"),
            LLMCall(trace_id=TRACE_ID, span_id="l2", parent_span_id="s2",
                    timestamp=_ts(4), node_name="x",
                    prompt_summary="same prompt", response_summary="same output"),
            NodeExit(
                trace_id=TRACE_ID, span_id="s2", timestamp=_ts(5),
                node_name="x", duration_ms=2000.0),
        ], trace_id=TRACE_ID)
        result = llm_patterns_report(trace)
        repeated = [p for p in result.patterns if p.pattern_type == "repeated_output"]
        assert repeated == []


# ---------------------------------------------------------------------------
# Redundant tool calls
# ---------------------------------------------------------------------------


class TestRedundantToolCallDetection:
    def test_duplicate_tool_detected(self, llm_pattern_trace: Trace) -> None:
        result = llm_patterns_report(llm_pattern_trace)
        redundant = [p for p in result.patterns if p.pattern_type == "redundant_tool_call"]
        assert len(redundant) == 1
        assert redundant[0].node_name == "agent"
        assert redundant[0].severity == "warning"
        assert redundant[0].evidence["tool_name"] == "search"
        assert redundant[0].evidence["call_count"] == 2

    def test_different_inputs_no_flag(self) -> None:
        trace = Trace(events=[
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="x"),
            ToolCall(trace_id=TRACE_ID, span_id="t1", parent_span_id="s1",
                     timestamp=_ts(1), node_name="x",
                     tool_name="search", tool_input={"query": "alpha"}),
            ToolCall(trace_id=TRACE_ID, span_id="t2", parent_span_id="s1",
                     timestamp=_ts(2), node_name="x",
                     tool_name="search", tool_input={"query": "beta"}),
            NodeExit(
                trace_id=TRACE_ID, span_id="s1", timestamp=_ts(3),
                node_name="x", duration_ms=3000.0),
        ], trace_id=TRACE_ID)
        result = llm_patterns_report(trace)
        redundant = [p for p in result.patterns if p.pattern_type == "redundant_tool_call"]
        assert redundant == []


# ---------------------------------------------------------------------------
# System message variance
# ---------------------------------------------------------------------------


class TestSystemMessageVariance:
    def test_variance_detected(self, llm_pattern_trace: Trace) -> None:
        result = llm_patterns_report(llm_pattern_trace)
        variance = [p for p in result.patterns if p.pattern_type == "system_message_variance"]
        assert len(variance) == 1
        assert variance[0].node_name == "planner"
        assert variance[0].severity == "info"
        assert len(variance[0].evidence["distinct_messages"]) == 2

    def test_consistent_messages_no_flag(self) -> None:
        trace = Trace(events=[
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="x"),
            LLMCall(trace_id=TRACE_ID, span_id="l1", parent_span_id="s1",
                    timestamp=_ts(1), node_name="x", system_message="You are helpful."),
            LLMCall(trace_id=TRACE_ID, span_id="l2", parent_span_id="s1",
                    timestamp=_ts(2), node_name="x", system_message="You are helpful."),
            NodeExit(
                trace_id=TRACE_ID, span_id="s1", timestamp=_ts(3),
                node_name="x", duration_ms=3000.0),
        ], trace_id=TRACE_ID)
        result = llm_patterns_report(trace)
        variance = [p for p in result.patterns if p.pattern_type == "system_message_variance"]
        assert variance == []

    def test_none_messages_ignored(self) -> None:
        trace = Trace(events=[
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="x"),
            LLMCall(trace_id=TRACE_ID, span_id="l1", parent_span_id="s1",
                    timestamp=_ts(1), node_name="x", system_message=None),
            LLMCall(trace_id=TRACE_ID, span_id="l2", parent_span_id="s1",
                    timestamp=_ts(2), node_name="x", system_message=None),
            NodeExit(
                trace_id=TRACE_ID, span_id="s1", timestamp=_ts(3),
                node_name="x", duration_ms=3000.0),
        ], trace_id=TRACE_ID)
        result = llm_patterns_report(trace)
        variance = [p for p in result.patterns if p.pattern_type == "system_message_variance"]
        assert variance == []


# ---------------------------------------------------------------------------
# Output not utilized
# ---------------------------------------------------------------------------


class TestOutputNotUtilized:
    def test_output_not_utilized(self, llm_pattern_trace: Trace) -> None:
        result = llm_patterns_report(llm_pattern_trace)
        unused = [p for p in result.patterns if p.pattern_type == "output_not_utilized"]
        assert len(unused) == 1
        assert unused[0].node_name == "formatter"
        assert unused[0].severity == "info"

    def test_output_utilized_no_flag(self, stateful_trace: Trace) -> None:
        """stateful_trace has state changes — should not flag output_not_utilized."""
        result = llm_patterns_report(stateful_trace)
        unused = [p for p in result.patterns if p.pattern_type == "output_not_utilized"]
        assert unused == []

    def test_no_state_data_skipped(self) -> None:
        """Without state_diff data, output_not_utilized should not fire."""
        trace = Trace(events=[
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="x"),
            LLMCall(trace_id=TRACE_ID, span_id="l1", parent_span_id="s1",
                    timestamp=_ts(1), node_name="x",
                    response_summary="some output",
                    output_message={"role": "ai", "content": "some output"}),
            NodeExit(
                trace_id=TRACE_ID, span_id="s1", timestamp=_ts(2),
                node_name="x", duration_ms=2000.0),
        ], trace_id=TRACE_ID)
        result = llm_patterns_report(trace)
        unused = [p for p in result.patterns if p.pattern_type == "output_not_utilized"]
        assert unused == []


# ---------------------------------------------------------------------------
# Node summaries
# ---------------------------------------------------------------------------


class TestNodeLLMSummary:
    def test_summary_counts(self, llm_pattern_trace: Trace) -> None:
        result = llm_patterns_report(llm_pattern_trace)
        by_name = {s.node_name: s for s in result.node_summaries}

        assert by_name["agent"].total_llm_calls == 2
        assert by_name["agent"].total_tool_calls == 2
        assert by_name["agent"].invocation_count == 2

        assert by_name["planner"].total_llm_calls == 2
        assert by_name["planner"].total_tool_calls == 0
        assert by_name["planner"].invocation_count == 1

        assert by_name["formatter"].total_llm_calls == 1
        assert by_name["formatter"].invocation_count == 1

    def test_avg_tokens(self, llm_pattern_trace: Trace) -> None:
        result = llm_patterns_report(llm_pattern_trace)
        by_name = {s.node_name: s for s in result.node_summaries}
        # agent: input_tokens = [100, 200], avg = 150.0
        assert by_name["agent"].avg_input_tokens == 150.0
        # agent: output_tokens = [50, 50], avg = 50.0
        assert by_name["agent"].avg_output_tokens == 50.0


# ---------------------------------------------------------------------------
# Integration with FullReport
# ---------------------------------------------------------------------------


class TestIntegrationWithFullReport:
    def test_llm_patterns_in_full_report(self, llm_pattern_trace: Trace) -> None:
        result = report(llm_pattern_trace)
        assert isinstance(result, FullReport)
        assert isinstance(result.llm_patterns, LLMPatternsReport)
        assert result.llm_patterns.total_llm_calls == 5
        # Should have patterns detected
        assert len(result.llm_patterns.patterns) > 0
