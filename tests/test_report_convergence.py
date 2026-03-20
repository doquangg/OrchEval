"""Tests for the convergence report module."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orcheval.events import PassBoundary
from orcheval.report.convergence import ConvergenceReport, convergence_report
from orcheval.trace import Trace

from .conftest import TRACE_ID, _ts


def _make_passes(exit_values_by_metric: dict[str, list[float]]) -> list[PassBoundary]:
    """Helper: build PassBoundary events from per-metric exit value lists."""
    n_passes = max(len(v) for v in exit_values_by_metric.values())
    events: list[PassBoundary] = []
    for i in range(n_passes):
        enter_snapshot = {k: 0 for k in exit_values_by_metric}
        exit_snapshot = {k: vs[i] for k, vs in exit_values_by_metric.items() if i < len(vs)}
        events.append(PassBoundary(
            trace_id=TRACE_ID, timestamp=_ts(i * 10),
            pass_number=i + 1, direction="enter",
            metrics_snapshot=enter_snapshot,
        ))
        events.append(PassBoundary(
            trace_id=TRACE_ID, timestamp=_ts(i * 10 + 5),
            pass_number=i + 1, direction="exit",
            metrics_snapshot=exit_snapshot,
        ))
    return events


class TestConvergenceReportBasic:
    def test_empty_trace(self) -> None:
        trace = Trace(events=[], trace_id=TRACE_ID)
        result = convergence_report(trace)
        assert isinstance(result, ConvergenceReport)
        assert result.passes == []
        assert result.total_passes == 0
        assert result.is_converging is None

    def test_pass_count(self, multipass_events: list) -> None:
        trace = Trace(events=multipass_events, trace_id=TRACE_ID)
        result = convergence_report(trace)
        assert result.total_passes == 3

    def test_passes_ordered_by_number(self, multipass_events: list) -> None:
        trace = Trace(events=multipass_events, trace_id=TRACE_ID)
        result = convergence_report(trace)
        numbers = [p.pass_number for p in result.passes]
        assert numbers == [1, 2, 3]


class TestPassSummary:
    def test_metrics_enter_and_exit(self, multipass_events: list) -> None:
        trace = Trace(events=multipass_events, trace_id=TRACE_ID)
        result = convergence_report(trace)
        p1 = result.passes[0]
        assert p1.metrics_enter["violation_count"] == 50
        assert p1.metrics_exit["violation_count"] == 30

    def test_metric_deltas(self, multipass_events: list) -> None:
        trace = Trace(events=multipass_events, trace_id=TRACE_ID)
        result = convergence_report(trace)
        p1 = result.passes[0]
        # violation_count: 30 - 50 = -20
        assert p1.metric_deltas["violation_count"] == -20.0
        # quality_score: 0.5 - 0.3 = 0.2
        assert abs(p1.metric_deltas["quality_score"] - 0.2) < 1e-9

    def test_duration_ms(self, multipass_events: list) -> None:
        trace = Trace(events=multipass_events, trace_id=TRACE_ID)
        result = convergence_report(trace)
        # Each pass: enter at i*10, exit at i*10+5 -> 5 seconds = 5000ms
        assert result.passes[0].duration_ms == 5000.0


class TestConvergenceDetection:
    def test_is_converging_true(self, multipass_events: list) -> None:
        trace = Trace(events=multipass_events, trace_id=TRACE_ID)
        result = convergence_report(trace)
        # Violation count decreasing: 30, 15, 10 (exit values)
        # Deltas: |15-30|=15, |10-15|=5 -> decreasing -> converging
        assert result.is_converging is True

    def test_is_converging_false(self) -> None:
        # Create passes where metrics diverge
        events = _make_passes({"score": [10, 20, 5]})
        # abs deltas: 10, 15 -> increasing -> diverging
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = convergence_report(trace)
        assert result.is_converging is False

    def test_is_converging_none_single_pass(self) -> None:
        events = [
            PassBoundary(
                trace_id=TRACE_ID, timestamp=_ts(0),
                pass_number=1, direction="enter", metrics_snapshot={"x": 10},
            ),
            PassBoundary(
                trace_id=TRACE_ID, timestamp=_ts(5),
                pass_number=1, direction="exit", metrics_snapshot={"x": 5},
            ),
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = convergence_report(trace)
        assert result.is_converging is None

    def test_is_converging_none_two_passes(self) -> None:
        events = []
        for i in range(2):
            events.append(PassBoundary(
                trace_id=TRACE_ID, timestamp=_ts(i * 10),
                pass_number=i + 1, direction="enter",
                metrics_snapshot={"x": 10 - i * 3},
            ))
            events.append(PassBoundary(
                trace_id=TRACE_ID, timestamp=_ts(i * 10 + 5),
                pass_number=i + 1, direction="exit",
                metrics_snapshot={"x": 8 - i * 3},
            ))
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = convergence_report(trace)
        # Only 2 exit values = 1 delta, can't determine trend
        assert result.is_converging is None


class TestPerMetricConvergence:
    """Tests for the per-metric convergence classification."""

    def test_all_converging(self, multipass_events: list) -> None:
        trace = Trace(events=multipass_events, trace_id=TRACE_ID)
        result = convergence_report(trace)
        assert result.is_converging is True
        assert len(result.per_metric) > 0
        for mc in result.per_metric:
            assert mc.status in ("converging", "plateaued")

    def test_mixed_converging_and_diverging(self) -> None:
        """quality_score converges while violations diverge."""
        events = _make_passes({
            "quality_score": [0.63, 0.82, 0.86, 0.9, 0.9],
            "violations_found": [30, 6, 8, 6, 14],
        })
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = convergence_report(trace)
        # Overall should be False since not all metrics converge
        assert result.is_converging is False
        by_name = {m.metric_name: m for m in result.per_metric}
        assert by_name["quality_score"].status in ("converging", "plateaued")
        assert by_name["violations_found"].status in ("diverging", "oscillating")

    def test_plateaued_metric(self) -> None:
        """Metric that reaches a stable value."""
        events = _make_passes({"score": [0.5, 0.8, 0.9, 0.9, 0.9]})
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = convergence_report(trace)
        by_name = {m.metric_name: m for m in result.per_metric}
        assert by_name["score"].status == "plateaued"

    def test_oscillating_metric(self) -> None:
        """Metric that alternates direction repeatedly."""
        # 6 values with direction changes: up, down, up, down, up = 4 changes >= 3
        events = _make_passes({"val": [10, 20, 10, 20, 10, 20]})
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = convergence_report(trace)
        by_name = {m.metric_name: m for m in result.per_metric}
        assert by_name["val"].status == "oscillating"

    def test_per_metric_has_values_and_deltas(self) -> None:
        events = _make_passes({"x": [10, 5, 3]})
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = convergence_report(trace)
        mc = result.per_metric[0]
        assert mc.metric_name == "x"
        assert mc.values == [10.0, 5.0, 3.0]
        assert mc.abs_deltas == [5.0, 2.0]

    def test_per_metric_empty_with_fewer_than_3_passes(self) -> None:
        events = _make_passes({"x": [10, 5]})
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = convergence_report(trace)
        assert result.per_metric == []
        assert result.is_converging is None


class TestMetricTrends:
    def test_trends_from_exit_values(self, multipass_events: list) -> None:
        trace = Trace(events=multipass_events, trace_id=TRACE_ID)
        result = convergence_report(trace)
        # Exit violation_count values: 30, 15, 10
        assert result.metric_trends["violation_count"] == [30.0, 15.0, 10.0]

    def test_final_metrics(self, multipass_events: list) -> None:
        trace = Trace(events=multipass_events, trace_id=TRACE_ID)
        result = convergence_report(trace)
        assert result.final_metrics["violation_count"] == 10


class TestConvergenceEdgeCases:
    def test_orphaned_enter_no_exit(self) -> None:
        events = [
            PassBoundary(
                trace_id=TRACE_ID, timestamp=_ts(0),
                pass_number=1, direction="enter",
                metrics_snapshot={"x": 100},
            ),
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = convergence_report(trace)
        assert result.total_passes == 1
        assert result.passes[0].metrics_exit == {}
        assert result.passes[0].duration_ms is None

    def test_non_numeric_metrics_excluded_from_deltas(self) -> None:
        events = [
            PassBoundary(
                trace_id=TRACE_ID, timestamp=_ts(0),
                pass_number=1, direction="enter",
                metrics_snapshot={"status": "running", "count": 10},
            ),
            PassBoundary(
                trace_id=TRACE_ID, timestamp=_ts(5),
                pass_number=1, direction="exit",
                metrics_snapshot={"status": "done", "count": 5},
            ),
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = convergence_report(trace)
        assert "status" not in result.passes[0].metric_deltas
        assert "count" in result.passes[0].metric_deltas
        # status should not appear in trends
        assert "status" not in result.metric_trends

    def test_frozen_result(self, multipass_events: list) -> None:
        trace = Trace(events=multipass_events, trace_id=TRACE_ID)
        result = convergence_report(trace)
        with pytest.raises(ValidationError):
            result.total_passes = 0  # type: ignore[misc]
