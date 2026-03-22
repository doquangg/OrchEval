"""Tests for the run comparison report module."""

from __future__ import annotations

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
from orcheval.report import report
from orcheval.report.comparison import (
    RunComparison,
    compare_runs,
)
from orcheval.trace import Trace

from .conftest import TRACE_ID, _ts

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _baseline_trace() -> Trace:
    """Baseline: agent(gpt-4o, $0.005, 3s) -> summarizer(gpt-4o-mini, $0.002, 2s).

    Includes: routing A->B (2x), one ValueError in validator, one pass (converging).
    """
    span_agent = "span-b-agent"
    span_summarizer = "span-b-summarizer"
    span_validator = "span-b-validator"

    events = [
        NodeEntry(trace_id=TRACE_ID, span_id=span_agent, timestamp=_ts(0), node_name="agent"),
        LLMCall(
            trace_id=TRACE_ID,
            span_id="span-b-llm1",
            parent_span_id=span_agent,
            timestamp=_ts(1),
            node_name="agent",
            model="gpt-4o",
            input_tokens=150,
            output_tokens=80,
            cost=0.005,
            duration_ms=800.0,
            input_messages=[{"role": "user", "content": "analyze"}],
            output_message={"role": "ai", "content": "done"},
        ),
        NodeExit(
            trace_id=TRACE_ID,
            span_id=span_agent,
            timestamp=_ts(3),
            node_name="agent",
            duration_ms=3000.0,
        ),
        NodeEntry(
            trace_id=TRACE_ID, span_id=span_summarizer, timestamp=_ts(4), node_name="summarizer"
        ),
        LLMCall(
            trace_id=TRACE_ID,
            span_id="span-b-llm2",
            parent_span_id=span_summarizer,
            timestamp=_ts(5),
            node_name="summarizer",
            model="gpt-4o-mini",
            input_tokens=200,
            output_tokens=100,
            cost=0.002,
            duration_ms=600.0,
            input_messages=[{"role": "user", "content": "summarize"}],
            output_message={"role": "ai", "content": "summary"},
        ),
        NodeExit(
            trace_id=TRACE_ID,
            span_id=span_summarizer,
            timestamp=_ts(6),
            node_name="summarizer",
            duration_ms=2000.0,
        ),
        # Routing
        RoutingDecision(
            trace_id=TRACE_ID,
            timestamp=_ts(7),
            node_name="router",
            source_node="router",
            target_node="agent",
        ),
        RoutingDecision(
            trace_id=TRACE_ID,
            timestamp=_ts(8),
            node_name="router",
            source_node="router",
            target_node="agent",
        ),
        # Error in validator
        NodeEntry(
            trace_id=TRACE_ID, span_id=span_validator, timestamp=_ts(9), node_name="validator"
        ),
        ErrorEvent(
            trace_id=TRACE_ID,
            span_id="span-b-err1",
            parent_span_id=span_validator,
            timestamp=_ts(10),
            node_name="validator",
            error_type="ValueError",
            error_message="invalid schema",
        ),
        NodeExit(
            trace_id=TRACE_ID,
            span_id=span_validator,
            timestamp=_ts(11),
            node_name="validator",
            duration_ms=2000.0,
        ),
        # Convergence: 1 pass
        PassBoundary(
            trace_id=TRACE_ID,
            timestamp=_ts(12),
            pass_number=1,
            direction="enter",
            metrics_snapshot={"score": 0.5},
        ),
        PassBoundary(
            trace_id=TRACE_ID,
            timestamp=_ts(15),
            pass_number=1,
            direction="exit",
            metrics_snapshot={"score": 0.7},
        ),
    ]
    return Trace(events=events, trace_id="baseline-trace")


def _experiment_trace() -> Trace:
    """Experiment: agent(gpt-4o, $0.008, 5s) -> summarizer(gpt-4o-mini, $0.001, 1s).

    Changes from baseline:
    - agent cost increased ($0.005 -> $0.008), duration increased (3s -> 5s, >20%)
    - summarizer cost decreased ($0.002 -> $0.001), duration decreased (2s -> 1s, 50%)
    - routing: A->B (1x) + A->C (1x, new edge), reduced count on A->B
    - new SyntaxError in codegen, ValueError in validator resolved
    - agent invoked twice (was once)
    - convergence: 2 passes
    """
    span_agent_1 = "span-e-agent-1"
    span_agent_2 = "span-e-agent-2"
    span_summarizer = "span-e-summarizer"
    span_codegen = "span-e-codegen"

    events = [
        # Agent invocation 1
        NodeEntry(trace_id=TRACE_ID, span_id=span_agent_1, timestamp=_ts(0), node_name="agent"),
        LLMCall(
            trace_id=TRACE_ID,
            span_id="span-e-llm1",
            parent_span_id=span_agent_1,
            timestamp=_ts(1),
            node_name="agent",
            model="gpt-4o",
            input_tokens=200,
            output_tokens=100,
            cost=0.004,
            duration_ms=900.0,
            input_messages=[{"role": "user", "content": "analyze"}],
            output_message={"role": "ai", "content": "retrying"},
        ),
        NodeExit(
            trace_id=TRACE_ID,
            span_id=span_agent_1,
            timestamp=_ts(2.5),
            node_name="agent",
            duration_ms=2500.0,
        ),
        # Agent invocation 2
        NodeEntry(trace_id=TRACE_ID, span_id=span_agent_2, timestamp=_ts(3), node_name="agent"),
        LLMCall(
            trace_id=TRACE_ID,
            span_id="span-e-llm2",
            parent_span_id=span_agent_2,
            timestamp=_ts(4),
            node_name="agent",
            model="gpt-4o",
            input_tokens=200,
            output_tokens=100,
            cost=0.004,
            duration_ms=900.0,
            input_messages=[{"role": "user", "content": "analyze again"}],
            output_message={"role": "ai", "content": "done"},
        ),
        NodeExit(
            trace_id=TRACE_ID,
            span_id=span_agent_2,
            timestamp=_ts(5),
            node_name="agent",
            duration_ms=2500.0,
        ),
        # Summarizer
        NodeEntry(
            trace_id=TRACE_ID, span_id=span_summarizer, timestamp=_ts(6), node_name="summarizer"
        ),
        LLMCall(
            trace_id=TRACE_ID,
            span_id="span-e-llm3",
            parent_span_id=span_summarizer,
            timestamp=_ts(6.5),
            node_name="summarizer",
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
            cost=0.001,
            duration_ms=400.0,
            input_messages=[{"role": "user", "content": "summarize"}],
            output_message={"role": "ai", "content": "summary"},
        ),
        NodeExit(
            trace_id=TRACE_ID,
            span_id=span_summarizer,
            timestamp=_ts(7),
            node_name="summarizer",
            duration_ms=1000.0,
        ),
        # Routing: router->agent (1x), router->codegen (new, 1x)
        RoutingDecision(
            trace_id=TRACE_ID,
            timestamp=_ts(8),
            node_name="router",
            source_node="router",
            target_node="agent",
        ),
        RoutingDecision(
            trace_id=TRACE_ID,
            timestamp=_ts(9),
            node_name="router",
            source_node="router",
            target_node="codegen",
        ),
        # New error in codegen
        NodeEntry(trace_id=TRACE_ID, span_id=span_codegen, timestamp=_ts(10), node_name="codegen"),
        ErrorEvent(
            trace_id=TRACE_ID,
            span_id="span-e-err1",
            parent_span_id=span_codegen,
            timestamp=_ts(11),
            node_name="codegen",
            error_type="SyntaxError",
            error_message="unexpected indent",
        ),
        NodeExit(
            trace_id=TRACE_ID,
            span_id=span_codegen,
            timestamp=_ts(12),
            node_name="codegen",
            duration_ms=2000.0,
        ),
        # Convergence: 2 passes
        PassBoundary(
            trace_id=TRACE_ID,
            timestamp=_ts(13),
            pass_number=1,
            direction="enter",
            metrics_snapshot={"score": 0.5},
        ),
        PassBoundary(
            trace_id=TRACE_ID,
            timestamp=_ts(15),
            pass_number=1,
            direction="exit",
            metrics_snapshot={"score": 0.7},
        ),
        PassBoundary(
            trace_id=TRACE_ID,
            timestamp=_ts(16),
            pass_number=2,
            direction="enter",
            metrics_snapshot={"score": 0.7},
        ),
        PassBoundary(
            trace_id=TRACE_ID,
            timestamp=_ts(18),
            pass_number=2,
            direction="exit",
            metrics_snapshot={"score": 0.85},
        ),
    ]
    return Trace(events=events, trace_id="experiment-trace")


@pytest.fixture
def baseline() -> Trace:
    return _baseline_trace()


@pytest.fixture
def experiment() -> Trace:
    return _experiment_trace()


# ---------------------------------------------------------------------------
# Tests: empty traces
# ---------------------------------------------------------------------------


class TestCompareRunsEmpty:
    def test_both_empty(self) -> None:
        empty = Trace(events=[], trace_id="empty")
        result = compare_runs(empty, empty)
        assert isinstance(result, RunComparison)
        assert result.cost.total_delta is None
        assert result.cost.node_deltas == []
        assert result.duration.total_delta is None
        assert result.routing.edges_added == []
        assert result.invocations.changes == []
        assert result.errors.new_errors == []
        assert result.convergence is None
        assert result.summary == ""

    def test_baseline_empty(self) -> None:
        empty = Trace(events=[], trace_id="empty")
        exp = _experiment_trace()
        result = compare_runs(empty, exp)
        # Experiment has cost data, baseline does not
        assert result.cost.total_delta is not None
        assert result.cost.total_delta.baseline is None
        assert result.cost.total_delta.experiment is not None
        # All routing edges are "added"
        assert len(result.routing.edges_added) > 0
        assert result.routing.edges_removed == []

    def test_experiment_empty(self) -> None:
        base = _baseline_trace()
        empty = Trace(events=[], trace_id="empty")
        result = compare_runs(base, empty)
        assert result.cost.total_delta is not None
        assert result.cost.total_delta.experiment is None
        # All routing edges are "removed"
        assert len(result.routing.edges_removed) > 0
        assert result.routing.edges_added == []


# ---------------------------------------------------------------------------
# Tests: cost comparison
# ---------------------------------------------------------------------------


class TestCostComparison:
    def test_total_cost_delta(self, baseline: Trace, experiment: Trace) -> None:
        result = compare_runs(baseline, experiment)
        total = result.cost.total_delta
        assert total is not None
        assert total.baseline == pytest.approx(0.007)  # 0.005 + 0.002
        assert total.experiment == pytest.approx(0.009)  # 0.004 + 0.004 + 0.001
        assert total.delta == pytest.approx(0.002)
        assert total.pct_change is not None
        assert total.pct_change == pytest.approx((0.002 / 0.007) * 100)

    def test_node_cost_deltas(self, baseline: Trace, experiment: Trace) -> None:
        result = compare_runs(baseline, experiment)
        node_map = {d.name: d for d in result.cost.node_deltas}
        assert "agent" in node_map
        assert "summarizer" in node_map
        # Agent: 0.005 -> 0.008
        agent = node_map["agent"]
        assert agent.baseline == pytest.approx(0.005)
        assert agent.experiment == pytest.approx(0.008)  # 0.004 + 0.004
        # Summarizer: 0.002 -> 0.001
        summarizer = node_map["summarizer"]
        assert summarizer.baseline == pytest.approx(0.002)
        assert summarizer.experiment == pytest.approx(0.001)
        assert summarizer.delta is not None
        assert summarizer.delta < 0  # got cheaper

    def test_model_cost_deltas(self, baseline: Trace, experiment: Trace) -> None:
        result = compare_runs(baseline, experiment)
        model_map = {d.name: d for d in result.cost.model_deltas}
        assert "gpt-4o" in model_map
        assert "gpt-4o-mini" in model_map

    def test_no_cost_data(self) -> None:
        """Traces with LLM calls but no cost produce None deltas."""
        events = [
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="a"),
            LLMCall(
                trace_id=TRACE_ID,
                span_id="s2",
                parent_span_id="s1",
                timestamp=_ts(1),
                node_name="a",
                model="gpt-4o",
                input_tokens=10,
                output_tokens=5,
            ),
            NodeExit(
                trace_id=TRACE_ID,
                span_id="s1",
                timestamp=_ts(2),
                node_name="a",
                duration_ms=2000.0,
            ),
        ]
        t = Trace(events=events, trace_id=TRACE_ID)
        result = compare_runs(t, t)
        assert result.cost.total_delta is None

    def test_node_only_in_one_trace(self) -> None:
        """A node in experiment but not baseline has baseline=None."""
        base = Trace(events=[], trace_id="b")
        exp_events = [
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="new_node"),
            LLMCall(
                trace_id=TRACE_ID,
                span_id="s2",
                parent_span_id="s1",
                timestamp=_ts(1),
                node_name="new_node",
                model="gpt-4o",
                input_tokens=10,
                output_tokens=5,
                cost=0.001,
            ),
            NodeExit(
                trace_id=TRACE_ID,
                span_id="s1",
                timestamp=_ts(2),
                node_name="new_node",
                duration_ms=2000.0,
            ),
        ]
        exp = Trace(events=exp_events, trace_id="e")
        result = compare_runs(base, exp)
        assert len(result.cost.node_deltas) == 1
        assert result.cost.node_deltas[0].name == "new_node"
        assert result.cost.node_deltas[0].baseline is None
        assert result.cost.node_deltas[0].experiment == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# Tests: duration comparison
# ---------------------------------------------------------------------------


class TestDurationComparison:
    def test_total_duration_delta(self, baseline: Trace, experiment: Trace) -> None:
        result = compare_runs(baseline, experiment)
        total = result.duration.total_delta
        assert total is not None
        assert total.baseline_ms is not None
        assert total.experiment_ms is not None
        assert total.delta_ms is not None

    def test_node_duration_flagged(self, baseline: Trace, experiment: Trace) -> None:
        result = compare_runs(baseline, experiment)
        node_map = {d.node_name: d for d in result.duration.node_deltas}
        # Summarizer: 2000ms -> 1000ms = -50%, should be flagged
        summarizer = node_map["summarizer"]
        assert summarizer.flagged is True
        assert summarizer.pct_change is not None
        assert summarizer.pct_change < 0

    def test_custom_threshold(self) -> None:
        """A 10% change is not flagged at default 20% but is at 5%."""
        events_b = [
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="a"),
            NodeExit(
                trace_id=TRACE_ID,
                span_id="s1",
                timestamp=_ts(1),
                node_name="a",
                duration_ms=1000.0,
            ),
        ]
        events_e = [
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="a"),
            NodeExit(
                trace_id=TRACE_ID,
                span_id="s1",
                timestamp=_ts(1),
                node_name="a",
                duration_ms=1100.0,
            ),
        ]
        b = Trace(events=events_b, trace_id="b")
        e = Trace(events=events_e, trace_id="e")

        result_default = compare_runs(b, e)
        assert result_default.duration.node_deltas[0].flagged is False

        result_strict = compare_runs(b, e, duration_flag_threshold=0.05)
        assert result_strict.duration.node_deltas[0].flagged is True

    def test_node_only_in_one_trace(self) -> None:
        events_b = [
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="old_node"),
            NodeExit(
                trace_id=TRACE_ID,
                span_id="s1",
                timestamp=_ts(1),
                node_name="old_node",
                duration_ms=1000.0,
            ),
        ]
        b = Trace(events=events_b, trace_id="b")
        e = Trace(events=[], trace_id="e")
        result = compare_runs(b, e)
        node_map = {d.node_name: d for d in result.duration.node_deltas}
        assert "old_node" in node_map
        assert node_map["old_node"].experiment_ms is None


# ---------------------------------------------------------------------------
# Tests: routing comparison
# ---------------------------------------------------------------------------


class TestRoutingComparison:
    def test_edge_added(self, baseline: Trace, experiment: Trace) -> None:
        result = compare_runs(baseline, experiment)
        added_edges = {(r.source_node, r.target_node) for r in result.routing.edges_added}
        assert ("router", "codegen") in added_edges

    def test_edge_changed(self, baseline: Trace, experiment: Trace) -> None:
        result = compare_runs(baseline, experiment)
        changed_map = {(r.source_node, r.target_node): r for r in result.routing.edges_changed}
        # router->agent was 2x in baseline, 1x in experiment
        assert ("router", "agent") in changed_map
        r = changed_map[("router", "agent")]
        assert r.baseline_count == 2
        assert r.experiment_count == 1

    def test_edge_removed(self) -> None:
        """An edge in baseline but not experiment is removed."""
        b_events = [
            RoutingDecision(
                trace_id=TRACE_ID,
                timestamp=_ts(0),
                node_name="r",
                source_node="r",
                target_node="x",
            ),
        ]
        e_events: list = []
        b = Trace(events=b_events, trace_id="b")
        e = Trace(events=e_events, trace_id="e")
        result = compare_runs(b, e)
        assert len(result.routing.edges_removed) == 1
        assert result.routing.edges_removed[0].source_node == "r"
        assert result.routing.edges_removed[0].target_node == "x"

    def test_identical_routing(self) -> None:
        events = [
            RoutingDecision(
                trace_id=TRACE_ID,
                timestamp=_ts(0),
                node_name="r",
                source_node="r",
                target_node="a",
            ),
        ]
        t = Trace(events=events, trace_id="t")
        result = compare_runs(t, t)
        assert result.routing.edges_added == []
        assert result.routing.edges_removed == []
        assert result.routing.edges_changed == []


# ---------------------------------------------------------------------------
# Tests: invocation comparison
# ---------------------------------------------------------------------------


class TestInvocationComparison:
    def test_invocation_count_changes(self, baseline: Trace, experiment: Trace) -> None:
        result = compare_runs(baseline, experiment)
        inv_map = {c.node_name: c for c in result.invocations.changes}
        # Agent: 1 -> 2 invocations
        assert "agent" in inv_map
        assert inv_map["agent"].baseline_count == 1
        assert inv_map["agent"].experiment_count == 2
        assert inv_map["agent"].delta == 1

    def test_new_node_in_experiment(self, baseline: Trace, experiment: Trace) -> None:
        result = compare_runs(baseline, experiment)
        inv_map = {c.node_name: c for c in result.invocations.changes}
        # codegen only in experiment
        assert "codegen" in inv_map
        assert inv_map["codegen"].baseline_count == 0
        assert inv_map["codegen"].experiment_count == 1

    def test_node_removed_in_experiment(self, baseline: Trace, experiment: Trace) -> None:
        result = compare_runs(baseline, experiment)
        inv_map = {c.node_name: c for c in result.invocations.changes}
        # validator only in baseline
        assert "validator" in inv_map
        assert inv_map["validator"].baseline_count == 1
        assert inv_map["validator"].experiment_count == 0
        assert inv_map["validator"].delta == -1

    def test_no_changes_for_identical(self) -> None:
        events = [
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="a"),
            NodeExit(
                trace_id=TRACE_ID,
                span_id="s1",
                timestamp=_ts(1),
                node_name="a",
                duration_ms=1000.0,
            ),
        ]
        t = Trace(events=events, trace_id="t")
        result = compare_runs(t, t)
        assert result.invocations.changes == []


# ---------------------------------------------------------------------------
# Tests: error comparison
# ---------------------------------------------------------------------------


class TestErrorComparison:
    def test_new_error_detected(self, baseline: Trace, experiment: Trace) -> None:
        result = compare_runs(baseline, experiment)
        new_keys = [(e.error_type, e.node_name) for e in result.errors.new_errors]
        assert ("SyntaxError", "codegen") in new_keys

    def test_resolved_error_detected(self, baseline: Trace, experiment: Trace) -> None:
        result = compare_runs(baseline, experiment)
        resolved_keys = [(e.error_type, e.node_name) for e in result.errors.resolved_errors]
        assert ("ValueError", "validator") in resolved_keys

    def test_matching_by_type_and_node(self) -> None:
        """Same error_type in different nodes are treated as different errors."""
        b_events = [
            ErrorEvent(
                trace_id=TRACE_ID,
                timestamp=_ts(0),
                node_name="node_a",
                error_type="ValueError",
                error_message="bad a",
            ),
        ]
        e_events = [
            ErrorEvent(
                trace_id=TRACE_ID,
                timestamp=_ts(0),
                node_name="node_b",
                error_type="ValueError",
                error_message="bad b",
            ),
        ]
        b = Trace(events=b_events, trace_id="b")
        e = Trace(events=e_events, trace_id="e")
        result = compare_runs(b, e)
        # ValueError in node_a is resolved, ValueError in node_b is new
        resolved_keys = [(err.error_type, err.node_name) for err in result.errors.resolved_errors]
        new_keys = [(err.error_type, err.node_name) for err in result.errors.new_errors]
        assert ("ValueError", "node_a") in resolved_keys
        assert ("ValueError", "node_b") in new_keys

    def test_count_changed(self) -> None:
        """Same error in both traces but different counts shows up as count_changed."""
        b_events = [
            ErrorEvent(
                trace_id=TRACE_ID,
                timestamp=_ts(0),
                node_name="n",
                error_type="ValueError",
                error_message="bad",
            ),
        ]
        e_events = [
            ErrorEvent(
                trace_id=TRACE_ID,
                timestamp=_ts(0),
                node_name="n",
                error_type="ValueError",
                error_message="bad",
            ),
            ErrorEvent(
                trace_id=TRACE_ID,
                timestamp=_ts(1),
                node_name="n",
                error_type="ValueError",
                error_message="bad again",
            ),
        ]
        b = Trace(events=b_events, trace_id="b")
        e = Trace(events=e_events, trace_id="e")
        result = compare_runs(b, e)
        assert len(result.errors.count_changes) == 1
        cc = result.errors.count_changes[0]
        assert cc.baseline_count == 1
        assert cc.experiment_count == 2
        assert cc.status == "count_changed"

    def test_no_errors_in_either(self) -> None:
        empty = Trace(events=[], trace_id="t")
        result = compare_runs(empty, empty)
        assert result.errors.new_errors == []
        assert result.errors.resolved_errors == []
        assert result.errors.count_changes == []

    def test_message_sample_populated(self, baseline: Trace, experiment: Trace) -> None:
        result = compare_runs(baseline, experiment)
        for e in result.errors.new_errors:
            if e.error_type == "SyntaxError":
                assert e.message_sample == "unexpected indent"
        for e in result.errors.resolved_errors:
            if e.error_type == "ValueError":
                assert e.message_sample == "invalid schema"


# ---------------------------------------------------------------------------
# Tests: convergence comparison
# ---------------------------------------------------------------------------


class TestConvergenceComparison:
    def test_convergence_pass_count_diff(self, baseline: Trace, experiment: Trace) -> None:
        result = compare_runs(baseline, experiment)
        conv = result.convergence
        assert conv is not None
        assert conv.baseline_total_passes == 1
        assert conv.experiment_total_passes == 2

    def test_metric_diffs(self, baseline: Trace, experiment: Trace) -> None:
        result = compare_runs(baseline, experiment)
        conv = result.convergence
        assert conv is not None
        assert "score" in conv.metric_diffs
        md = conv.metric_diffs["score"]
        assert md["baseline_final"] == pytest.approx(0.7)
        assert md["experiment_final"] == pytest.approx(0.85)

    def test_both_no_convergence(self) -> None:
        empty = Trace(events=[], trace_id="t")
        result = compare_runs(empty, empty)
        assert result.convergence is None

    def test_one_side_has_convergence(self) -> None:
        events = [
            PassBoundary(
                trace_id=TRACE_ID,
                timestamp=_ts(0),
                pass_number=1,
                direction="enter",
                metrics_snapshot={"x": 1},
            ),
            PassBoundary(
                trace_id=TRACE_ID,
                timestamp=_ts(1),
                pass_number=1,
                direction="exit",
                metrics_snapshot={"x": 2},
            ),
        ]
        b = Trace(events=events, trace_id="b")
        e = Trace(events=[], trace_id="e")
        result = compare_runs(b, e)
        assert result.convergence is not None
        assert result.convergence.baseline_total_passes == 1
        assert result.convergence.experiment_total_passes == 0


# ---------------------------------------------------------------------------
# Tests: LLM pattern comparison
# ---------------------------------------------------------------------------


class TestPatternComparison:
    def test_new_pattern(self) -> None:
        """A pattern in experiment but not baseline is reported as new."""
        # Baseline: no patterns (simple clean trace)
        b_events = [
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="agent"),
            LLMCall(
                trace_id=TRACE_ID,
                span_id="s2",
                parent_span_id="s1",
                timestamp=_ts(1),
                node_name="agent",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
                cost=0.003,
                input_messages=[{"role": "user", "content": "hi"}],
                output_message={"role": "ai", "content": "hello"},
            ),
            NodeExit(
                trace_id=TRACE_ID,
                span_id="s1",
                timestamp=_ts(2),
                node_name="agent",
                duration_ms=2000.0,
            ),
        ]
        # Experiment: redundant tool calls -> triggers redundant_tool_call pattern
        span1 = "span-e-a1"
        e_events = [
            NodeEntry(trace_id=TRACE_ID, span_id=span1, timestamp=_ts(0), node_name="agent"),
            LLMCall(
                trace_id=TRACE_ID,
                span_id="s2",
                parent_span_id=span1,
                timestamp=_ts(1),
                node_name="agent",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
                cost=0.003,
                input_messages=[{"role": "user", "content": "hi"}],
                output_message={"role": "ai", "content": "hello"},
            ),
            ToolCall(
                trace_id=TRACE_ID,
                span_id="t1",
                parent_span_id=span1,
                timestamp=_ts(1.5),
                node_name="agent",
                tool_name="search",
                tool_input={"q": "x"},
                tool_output="r",
            ),
            ToolCall(
                trace_id=TRACE_ID,
                span_id="t2",
                parent_span_id=span1,
                timestamp=_ts(1.7),
                node_name="agent",
                tool_name="search",
                tool_input={"q": "x"},
                tool_output="r",
            ),
            NodeExit(
                trace_id=TRACE_ID,
                span_id=span1,
                timestamp=_ts(2),
                node_name="agent",
                duration_ms=2000.0,
            ),
        ]
        b = Trace(events=b_events, trace_id="b")
        e = Trace(events=e_events, trace_id="e")
        result = compare_runs(b, e)
        new_keys = [(p.pattern_type, p.node_name) for p in result.llm_patterns.new_patterns]
        assert ("redundant_tool_call", "agent") in new_keys

    def test_resolved_pattern(self) -> None:
        """A pattern in baseline but not experiment is reported as resolved."""
        # Baseline: redundant tool calls
        span1 = "span-b-a1"
        b_events = [
            NodeEntry(trace_id=TRACE_ID, span_id=span1, timestamp=_ts(0), node_name="agent"),
            LLMCall(
                trace_id=TRACE_ID,
                span_id="s2",
                parent_span_id=span1,
                timestamp=_ts(1),
                node_name="agent",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
                cost=0.003,
                input_messages=[{"role": "user", "content": "hi"}],
                output_message={"role": "ai", "content": "hello"},
            ),
            ToolCall(
                trace_id=TRACE_ID,
                span_id="t1",
                parent_span_id=span1,
                timestamp=_ts(1.5),
                node_name="agent",
                tool_name="search",
                tool_input={"q": "x"},
                tool_output="r",
            ),
            ToolCall(
                trace_id=TRACE_ID,
                span_id="t2",
                parent_span_id=span1,
                timestamp=_ts(1.7),
                node_name="agent",
                tool_name="search",
                tool_input={"q": "x"},
                tool_output="r",
            ),
            NodeExit(
                trace_id=TRACE_ID,
                span_id=span1,
                timestamp=_ts(2),
                node_name="agent",
                duration_ms=2000.0,
            ),
        ]
        # Experiment: clean
        e_events = [
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="agent"),
            LLMCall(
                trace_id=TRACE_ID,
                span_id="s2",
                parent_span_id="s1",
                timestamp=_ts(1),
                node_name="agent",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
                cost=0.003,
                input_messages=[{"role": "user", "content": "hi"}],
                output_message={"role": "ai", "content": "hello"},
            ),
            NodeExit(
                trace_id=TRACE_ID,
                span_id="s1",
                timestamp=_ts(2),
                node_name="agent",
                duration_ms=2000.0,
            ),
        ]
        b = Trace(events=b_events, trace_id="b")
        e = Trace(events=e_events, trace_id="e")
        result = compare_runs(b, e)
        resolved_keys = [
            (p.pattern_type, p.node_name) for p in result.llm_patterns.resolved_patterns
        ]
        assert ("redundant_tool_call", "agent") in resolved_keys

    def test_matching_by_type_and_node(self) -> None:
        """Same pattern_type in different nodes treated as different."""
        # Both have redundant tool calls, but in different nodes
        span_a = "span-a"
        span_b = "span-b"
        b_events = [
            NodeEntry(trace_id=TRACE_ID, span_id=span_a, timestamp=_ts(0), node_name="node_a"),
            LLMCall(
                trace_id=TRACE_ID,
                span_id="la",
                parent_span_id=span_a,
                timestamp=_ts(0.5),
                node_name="node_a",
                model="gpt-4o",
                input_tokens=10,
                output_tokens=5,
            ),
            ToolCall(
                trace_id=TRACE_ID,
                span_id="t1",
                parent_span_id=span_a,
                timestamp=_ts(1),
                node_name="node_a",
                tool_name="s",
                tool_input={"q": "x"},
                tool_output="r",
            ),
            ToolCall(
                trace_id=TRACE_ID,
                span_id="t2",
                parent_span_id=span_a,
                timestamp=_ts(1.5),
                node_name="node_a",
                tool_name="s",
                tool_input={"q": "x"},
                tool_output="r",
            ),
            NodeExit(
                trace_id=TRACE_ID,
                span_id=span_a,
                timestamp=_ts(2),
                node_name="node_a",
                duration_ms=2000.0,
            ),
        ]
        e_events = [
            NodeEntry(trace_id=TRACE_ID, span_id=span_b, timestamp=_ts(0), node_name="node_b"),
            LLMCall(
                trace_id=TRACE_ID,
                span_id="lb",
                parent_span_id=span_b,
                timestamp=_ts(0.5),
                node_name="node_b",
                model="gpt-4o",
                input_tokens=10,
                output_tokens=5,
            ),
            ToolCall(
                trace_id=TRACE_ID,
                span_id="t3",
                parent_span_id=span_b,
                timestamp=_ts(1),
                node_name="node_b",
                tool_name="s",
                tool_input={"q": "x"},
                tool_output="r",
            ),
            ToolCall(
                trace_id=TRACE_ID,
                span_id="t4",
                parent_span_id=span_b,
                timestamp=_ts(1.5),
                node_name="node_b",
                tool_name="s",
                tool_input={"q": "x"},
                tool_output="r",
            ),
            NodeExit(
                trace_id=TRACE_ID,
                span_id=span_b,
                timestamp=_ts(2),
                node_name="node_b",
                duration_ms=2000.0,
            ),
        ]
        b = Trace(events=b_events, trace_id="b")
        e = Trace(events=e_events, trace_id="e")
        result = compare_runs(b, e)
        # redundant_tool_call in node_a resolved, in node_b new
        resolved_keys = [
            (p.pattern_type, p.node_name) for p in result.llm_patterns.resolved_patterns
        ]
        new_keys = [(p.pattern_type, p.node_name) for p in result.llm_patterns.new_patterns]
        assert ("redundant_tool_call", "node_a") in resolved_keys
        assert ("redundant_tool_call", "node_b") in new_keys


# ---------------------------------------------------------------------------
# Tests: summary generation
# ---------------------------------------------------------------------------


class TestSummaryGeneration:
    def test_summary_includes_new_errors(self, baseline: Trace, experiment: Trace) -> None:
        result = compare_runs(baseline, experiment)
        assert "SyntaxError" in result.summary
        assert "codegen" in result.summary

    def test_summary_includes_cost_change(self, baseline: Trace, experiment: Trace) -> None:
        result = compare_runs(baseline, experiment)
        assert "cost" in result.summary.lower()

    def test_summary_includes_resolved(self, baseline: Trace, experiment: Trace) -> None:
        result = compare_runs(baseline, experiment)
        assert "Resolved" in result.summary
        assert "ValueError" in result.summary

    def test_summary_empty_for_identical_traces(self) -> None:
        events = [
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="a"),
            NodeExit(
                trace_id=TRACE_ID,
                span_id="s1",
                timestamp=_ts(1),
                node_name="a",
                duration_ms=1000.0,
            ),
        ]
        t = Trace(events=events, trace_id="t")
        result = compare_runs(t, t)
        assert result.summary == ""

    def test_summary_prioritizes_new_errors_over_cost(self) -> None:
        """New errors should appear before cost changes in summary."""
        b_events = [
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="a"),
            LLMCall(
                trace_id=TRACE_ID,
                span_id="s2",
                parent_span_id="s1",
                timestamp=_ts(1),
                node_name="a",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
                cost=0.001,
            ),
            NodeExit(
                trace_id=TRACE_ID,
                span_id="s1",
                timestamp=_ts(2),
                node_name="a",
                duration_ms=2000.0,
            ),
        ]
        e_events = [
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="a"),
            LLMCall(
                trace_id=TRACE_ID,
                span_id="s2",
                parent_span_id="s1",
                timestamp=_ts(1),
                node_name="a",
                model="gpt-4o",
                input_tokens=200,
                output_tokens=100,
                cost=0.005,
            ),
            ErrorEvent(
                trace_id=TRACE_ID,
                timestamp=_ts(1.5),
                node_name="a",
                error_type="RuntimeError",
                error_message="boom",
            ),
            NodeExit(
                trace_id=TRACE_ID,
                span_id="s1",
                timestamp=_ts(2),
                node_name="a",
                duration_ms=2000.0,
            ),
        ]
        b = Trace(events=b_events, trace_id="b")
        e = Trace(events=e_events, trace_id="e")
        result = compare_runs(b, e)
        # "New error" should appear before "cost"
        error_pos = result.summary.find("New error")
        cost_pos = result.summary.lower().find("cost")
        assert error_pos >= 0
        assert cost_pos >= 0
        assert error_pos < cost_pos


# ---------------------------------------------------------------------------
# Tests: Trace.compare() convenience method
# ---------------------------------------------------------------------------


class TestTraceCompare:
    def test_trace_compare_method(self, baseline: Trace, experiment: Trace) -> None:
        result = baseline.compare(experiment)
        assert isinstance(result, RunComparison)

    def test_compare_matches_direct_call(self, baseline: Trace, experiment: Trace) -> None:
        direct = compare_runs(baseline, experiment)
        convenience = baseline.compare(experiment)
        assert direct.summary == convenience.summary
        assert direct.cost == convenience.cost
        assert direct.errors == convenience.errors


# ---------------------------------------------------------------------------
# Tests: pre-computed reports
# ---------------------------------------------------------------------------


class TestPrecomputedReports:
    def test_accepts_precomputed(self, baseline: Trace, experiment: Trace) -> None:
        b_report = report(baseline)
        e_report = report(experiment)
        result = compare_runs(
            baseline,
            experiment,
            baseline_report=b_report,
            experiment_report=e_report,
        )
        direct = compare_runs(baseline, experiment)
        assert result.cost == direct.cost
        assert result.routing == direct.routing

    def test_partial_precomputed(self, baseline: Trace, experiment: Trace) -> None:
        """Only one side pre-computed, other generated internally."""
        b_report = report(baseline)
        result = compare_runs(baseline, experiment, baseline_report=b_report)
        assert isinstance(result, RunComparison)
