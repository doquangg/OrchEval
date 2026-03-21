"""Tests for the routing report module."""

from __future__ import annotations

from orcheval.events import RoutingDecision
from orcheval.report.routing import RoutingReport, routing_report
from orcheval.trace import Trace

from .conftest import TRACE_ID, _ts


class TestRoutingReportBasic:
    def test_empty_trace(self) -> None:
        trace = Trace(events=[], trace_id=TRACE_ID)
        result = routing_report(trace)
        assert isinstance(result, RoutingReport)
        assert result.decisions == []
        assert result.flags == []
        assert result.total_decisions == 0

    def test_single_decision_no_flags(self) -> None:
        events = [
            RoutingDecision(
                trace_id=TRACE_ID, timestamp=_ts(0),
                source_node="router", target_node="target_a",
                decision_context={"score": 0.8},
            ),
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = routing_report(trace)
        assert result.total_decisions == 1
        assert len(result.decisions) == 1
        assert result.flags == []  # single decision cannot trigger any flag

    def test_edge_counts_and_fractions(self, routing_events: list) -> None:
        trace = Trace(events=routing_events, trace_id=TRACE_ID)
        result = routing_report(trace)

        # router_a has 3 decisions all to node_x
        ra_edges = [e for e in result.decisions if e.source_node == "router_a"]
        assert len(ra_edges) == 1
        assert ra_edges[0].target_node == "node_x"
        assert ra_edges[0].count == 3
        assert ra_edges[0].fraction == 1.0

    def test_unique_sources_and_targets(self, routing_events: list) -> None:
        trace = Trace(events=routing_events, trace_id=TRACE_ID)
        result = routing_report(trace)
        assert result.unique_sources == 3  # router_a, router_b, router_c
        assert result.unique_targets == 5  # node_x, node_y, node_z, node_p, node_q

    def test_sample_contexts_capped(self, routing_events: list) -> None:
        trace = Trace(events=routing_events, trace_id=TRACE_ID)
        result = routing_report(trace)
        # router_c -> node_p has 19 decisions but sample_contexts capped at 3
        rc_np = next(
            e for e in result.decisions
            if e.source_node == "router_c" and e.target_node == "node_p"
        )
        assert len(rc_np.sample_contexts) <= 3


class TestInvariantRouting:
    def test_invariant_detected(self, routing_events: list) -> None:
        trace = Trace(events=routing_events, trace_id=TRACE_ID)
        result = routing_report(trace)
        inv_flags = [f for f in result.flags if f.flag_type == "invariant_routing"]
        # router_a always goes to node_x
        ra_flags = [f for f in inv_flags if f.source_node == "router_a"]
        assert len(ra_flags) == 1
        assert ra_flags[0].target_node == "node_x"

    def test_no_invariant_for_single_decision(self) -> None:
        events = [
            RoutingDecision(
                trace_id=TRACE_ID, timestamp=_ts(0),
                source_node="router", target_node="target",
            ),
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = routing_report(trace)
        inv_flags = [f for f in result.flags if f.flag_type == "invariant_routing"]
        assert inv_flags == []


class TestContextDivergence:
    def test_divergence_detected(self, routing_events: list) -> None:
        trace = Trace(events=routing_events, trace_id=TRACE_ID)
        result = routing_report(trace)
        div_flags = [f for f in result.flags if f.flag_type == "context_divergence"]
        rb_flags = [f for f in div_flags if f.source_node == "router_b"]
        assert len(rb_flags) == 1
        assert set(rb_flags[0].evidence["targets"]) == {"node_y", "node_z"}

    def test_empty_context_divergence(self) -> None:
        events = [
            RoutingDecision(
                trace_id=TRACE_ID, timestamp=_ts(0),
                source_node="router", target_node="a", decision_context={},
            ),
            RoutingDecision(
                trace_id=TRACE_ID, timestamp=_ts(1),
                source_node="router", target_node="b", decision_context={},
            ),
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = routing_report(trace)
        div_flags = [f for f in result.flags if f.flag_type == "context_divergence"]
        assert len(div_flags) == 1


class TestDominantPath:
    def test_dominant_path_detected(self, routing_events: list) -> None:
        trace = Trace(events=routing_events, trace_id=TRACE_ID)
        result = routing_report(trace)
        dom_flags = [f for f in result.flags if f.flag_type == "dominant_path"]
        rc_flags = [f for f in dom_flags if f.source_node == "router_c"]
        assert len(rc_flags) == 1
        assert rc_flags[0].target_node == "node_p"
        assert rc_flags[0].evidence["fraction"] == 19 / 20

    def test_custom_threshold(self, routing_events: list) -> None:
        trace = Trace(events=routing_events, trace_id=TRACE_ID)
        # With threshold 0.99, router_c (95%) should not be flagged
        result = routing_report(trace, dominance_threshold=0.99)
        dom_flags = [f for f in result.flags if f.flag_type == "dominant_path"]
        rc_flags = [f for f in dom_flags if f.source_node == "router_c"]
        assert rc_flags == []


class TestOscillation:
    def test_oscillation_detected(self, oscillation_events: list) -> None:
        trace = Trace(events=oscillation_events, trace_id=TRACE_ID)
        result = routing_report(trace)
        osc_flags = [f for f in result.flags if f.flag_type == "oscillation"]
        assert len(osc_flags) == 1
        assert osc_flags[0].source_node == "oscillator"
        assert set(osc_flags[0].evidence["targets"]) == {"node_a", "node_b"}

    def test_no_oscillation_with_few_decisions(self) -> None:
        # Only 3 decisions — not enough to observe 3 alternations
        events = [
            RoutingDecision(
                trace_id=TRACE_ID, timestamp=_ts(i),
                source_node="router",
                target_node="a" if i % 2 == 0 else "b",
            )
            for i in range(3)
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = routing_report(trace)
        osc_flags = [f for f in result.flags if f.flag_type == "oscillation"]
        assert osc_flags == []

    def test_oscillation_detected_at_minimum_threshold(self) -> None:
        # 4 decisions [a, b, a, b] = 3 alternations, exactly at threshold
        events = [
            RoutingDecision(
                trace_id=TRACE_ID, timestamp=_ts(i),
                source_node="router",
                target_node="a" if i % 2 == 0 else "b",
            )
            for i in range(4)
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = routing_report(trace)
        osc_flags = [f for f in result.flags if f.flag_type == "oscillation"]
        assert len(osc_flags) == 1

