"""Tests for orcheval.digest — compact text digest for LLM consumption."""

from __future__ import annotations

from typing import TYPE_CHECKING

from orcheval.report import report
from orcheval.trace import Trace

if TYPE_CHECKING:
    from orcheval.events import (
        ErrorEvent,
        NodeEntry,
        NodeExit,
        PassBoundary,
        RoutingDecision,
    )

TRACE_ID = "test-trace-0001"


class TestDigestOverview:
    def test_contains_node_count(self, sample_trace: Trace) -> None:
        digest = sample_trace.to_digest()
        assert "2 nodes" in digest

    def test_contains_duration(self, sample_trace: Trace) -> None:
        digest = sample_trace.to_digest()
        assert "6000" in digest

    def test_contains_cost(self, sample_trace: Trace) -> None:
        digest = sample_trace.to_digest()
        assert "0.007" in digest

    def test_contains_token_count(self, sample_trace: Trace) -> None:
        digest = sample_trace.to_digest()
        assert "530" in digest

    def test_contains_no_errors(self, sample_trace: Trace) -> None:
        digest = sample_trace.to_digest()
        assert "no errors" in digest

    def test_contains_trace_id(self, sample_trace: Trace) -> None:
        digest = sample_trace.to_digest()
        assert TRACE_ID in digest


class TestDigestExecutionFlow:
    def test_nodes_appear_in_order(self, sample_trace: Trace) -> None:
        digest = sample_trace.to_digest()
        agent_pos = digest.index("agent")
        summarizer_pos = digest.index("summarizer")
        assert agent_pos < summarizer_pos

    def test_llm_info_present(self, sample_trace: Trace) -> None:
        digest = sample_trace.to_digest()
        assert "gpt-4o" in digest

    def test_tool_info_present(self, sample_trace: Trace) -> None:
        digest = sample_trace.to_digest()
        assert "search" in digest

    def test_duration_per_node(self, sample_trace: Trace) -> None:
        digest = sample_trace.to_digest()
        assert "3000" in digest  # agent duration


class TestDigestAnomalies:
    def test_no_anomalies_for_clean_trace(self, sample_trace: Trace) -> None:
        digest = sample_trace.to_digest()
        assert "No anomalies detected" in digest

    def test_routing_flags_appear(self, routing_events: list[RoutingDecision]) -> None:
        trace = Trace(events=routing_events, trace_id=TRACE_ID)
        digest = trace.to_digest()
        # The routing_events fixture has invariant, divergence, and dominant path patterns
        assert "Routing" in digest

    def test_retry_info_appears(
        self, error_retry_events: list[NodeEntry | NodeExit | ErrorEvent],
    ) -> None:
        trace = Trace(events=error_retry_events, trace_id=TRACE_ID)
        digest = trace.to_digest()
        assert "Retry" in digest
        assert "codegen" in digest

    def test_convergence_info_appears(self, multipass_events: list[PassBoundary]) -> None:
        trace = Trace(events=multipass_events, trace_id=TRACE_ID)
        digest = trace.to_digest()
        # multipass_events fixture has converging metrics — no diverging/oscillating
        # so anomalies section should say no anomalies for convergence
        assert "## Anomalies Detected" in digest


class TestDigestFocusNodes:
    def test_focused_node_detailed(self, sample_trace: Trace) -> None:
        digest = sample_trace.to_digest(focus_nodes=["agent"])
        assert "agent" in digest
        assert "gpt-4o" in digest

    def test_unfocused_nodes_collapsed(self, sample_trace: Trace) -> None:
        digest = sample_trace.to_digest(focus_nodes=["agent"])
        # The "other node invocations" collapse message appears
        assert "other node invocation" in digest

    def test_nonexistent_focus_node(self, sample_trace: Trace) -> None:
        digest = sample_trace.to_digest(focus_nodes=["nonexistent"])
        # Should still produce valid output, just with all nodes collapsed
        assert "## Overview" in digest
        assert "other node invocation" in digest


class TestDigestIncludeLLMContent:
    def test_default_excludes_content(self, sample_trace: Trace) -> None:
        digest = sample_trace.to_digest()
        # Full message content should NOT appear by default
        assert "LLM Call Detail" not in digest

    def test_include_shows_messages(self, sample_trace: Trace) -> None:
        digest = sample_trace.to_digest(include_llm_content=True)
        assert "LLM Call Detail" in digest
        assert "Analyze the data" in digest


class TestDigestStateEvolution:
    def test_state_section_present(self, stateful_trace: Trace) -> None:
        digest = stateful_trace.to_digest()
        assert "State Evolution" in digest
        assert "added" in digest
        assert "result" in digest

    def test_state_section_absent(self, sample_trace: Trace) -> None:
        digest = sample_trace.to_digest()
        assert "State Evolution" not in digest


class TestDigestBudget:
    def test_respects_max_chars(self, sample_trace: Trace) -> None:
        digest = sample_trace.to_digest(max_chars=16_000)
        assert len(digest) <= 16_000

    def test_small_budget_still_has_overview(self) -> None:
        # Even with tiny budget, overview and anomalies survive
        trace = Trace(events=[], trace_id=TRACE_ID)
        digest = trace.to_digest(max_chars=500)
        assert "## Overview" in digest


class TestDigestPrecomputedReports:
    def test_accepts_precomputed(self, sample_trace: Trace) -> None:
        full = report(sample_trace)
        digest = sample_trace.to_digest(reports=full)
        assert "## Overview" in digest

    def test_matches_without_precomputed(self, sample_trace: Trace) -> None:
        full = report(sample_trace)
        digest_with = sample_trace.to_digest(reports=full)
        digest_without = sample_trace.to_digest()
        # Both should contain the same key data
        assert "2 nodes" in digest_with
        assert "2 nodes" in digest_without


class TestDigestEmpty:
    def test_empty_trace(self) -> None:
        trace = Trace(events=[], trace_id=TRACE_ID)
        digest = trace.to_digest()
        assert "## Overview" in digest
        assert isinstance(digest, str)
