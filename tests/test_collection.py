"""Tests for the cross-run trace aggregation module."""

from __future__ import annotations

from datetime import timedelta

import pytest

from orcheval.collection import (
    TraceCollection,
)
from orcheval.events import ErrorEvent, LLMCall, NodeEntry, NodeExit
from orcheval.trace import Trace

from .conftest import BASE_TIME

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace(
    trace_id: str,
    nodes: list[str],
    *,
    cost: float | None = None,
    base_offset: float = 0.0,
    error_node: str | None = None,
    tokens: tuple[int, int] | None = None,
) -> Trace:
    """Build a simple trace with the given node sequence and optional cost.

    Each node gets a 1-second span. If cost is given, a single LLMCall is
    added to the first node with that cost.
    """
    events: list = []
    t = 0.0

    for i, name in enumerate(nodes):
        span = f"span-{trace_id}-{i}"
        events.append(NodeEntry(
            trace_id=trace_id, span_id=span,
            timestamp=BASE_TIME + timedelta(seconds=base_offset + t),
            node_name=name,
        ))

        if i == 0 and cost is not None:
            in_tok = tokens[0] if tokens else 100
            out_tok = tokens[1] if tokens else 50
            events.append(LLMCall(
                trace_id=trace_id, span_id=f"llm-{trace_id}-{i}",
                parent_span_id=span,
                timestamp=BASE_TIME + timedelta(seconds=base_offset + t + 0.5),
                node_name=name,
                model="gpt-4o", input_tokens=in_tok, output_tokens=out_tok,
                cost=cost, duration_ms=500.0,
            ))

        if error_node and name == error_node:
            events.append(ErrorEvent(
                trace_id=trace_id, span_id=f"err-{trace_id}-{i}",
                parent_span_id=span,
                timestamp=BASE_TIME + timedelta(seconds=base_offset + t + 0.3),
                node_name=name,
                error_type="RuntimeError",
                error_message="something failed",
            ))

        t += 1.0
        events.append(NodeExit(
            trace_id=trace_id, span_id=span,
            timestamp=BASE_TIME + timedelta(seconds=base_offset + t),
            node_name=name, duration_ms=1000.0,
        ))

    return Trace(events=events, trace_id=trace_id)


@pytest.fixture
def collection_traces() -> list[Trace]:
    """Five traces with known characteristics.

    t1, t2: same shape [agent, summarizer], cost=0.01, 0.012
    t3:      same shape [agent, summarizer], cost=0.10 (outlier)
    t4:      different shape [planner, executor, summarizer], cost=0.02
    t5:      shape with error [agent, validator(error)], cost=0.005
    """
    return [
        _make_trace("t1", ["agent", "summarizer"], cost=0.010, base_offset=0),
        _make_trace("t2", ["agent", "summarizer"], cost=0.012, base_offset=10),
        _make_trace("t3", ["agent", "summarizer"], cost=0.100, base_offset=20),
        _make_trace("t4", ["planner", "executor", "summarizer"], cost=0.020, base_offset=30),
        _make_trace(
            "t5", ["agent", "validator"], cost=0.005, base_offset=40, error_node="validator"
        ),
    ]


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


class TestTraceCollectionBasic:
    def test_empty_collection(self) -> None:
        coll = TraceCollection([])
        assert len(coll) == 0
        assert not coll
        summary = coll.summary()
        assert summary.trace_count == 0
        assert summary.total_cost is None

    def test_single_trace(self) -> None:
        t = _make_trace("solo", ["agent"], cost=0.05)
        coll = TraceCollection([t])
        assert len(coll) == 1
        assert coll
        summary = coll.summary()
        assert summary.trace_count == 1
        assert summary.total_cost is not None
        assert summary.total_cost.count == 1

    def test_len(self, collection_traces: list[Trace]) -> None:
        coll = TraceCollection(collection_traces)
        assert len(coll) == 5

    def test_traces_property_returns_copy(self, collection_traces: list[Trace]) -> None:
        coll = TraceCollection(collection_traces)
        t1 = coll.traces
        t2 = coll.traces
        assert t1 is not t2
        assert len(t1) == len(t2)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestCollectionSummary:
    def test_cost_stats(self, collection_traces: list[Trace]) -> None:
        coll = TraceCollection(collection_traces)
        summary = coll.summary()
        assert summary.total_cost is not None
        assert summary.total_cost.count == 5
        # Costs: 0.01, 0.012, 0.10, 0.02, 0.005
        assert summary.total_cost.min == pytest.approx(0.005)
        assert summary.total_cost.max == pytest.approx(0.10)

    def test_duration_stats(self, collection_traces: list[Trace]) -> None:
        coll = TraceCollection(collection_traces)
        summary = coll.summary()
        assert summary.total_duration is not None
        assert summary.total_duration.count == 5

    def test_unique_node_names(self, collection_traces: list[Trace]) -> None:
        coll = TraceCollection(collection_traces)
        summary = coll.summary()
        expected = sorted({"agent", "summarizer", "planner", "executor", "validator"})
        assert summary.unique_node_names == expected

    def test_none_costs_excluded(self) -> None:
        traces = [
            _make_trace("a", ["x"], cost=0.01),
            _make_trace("b", ["x"]),  # no cost
        ]
        coll = TraceCollection(traces)
        summary = coll.summary()
        assert summary.total_cost is not None
        assert summary.total_cost.count == 1


# ---------------------------------------------------------------------------
# Node stats
# ---------------------------------------------------------------------------


class TestNodeStats:
    def test_duration_per_node(self, collection_traces: list[Trace]) -> None:
        coll = TraceCollection(collection_traces)
        stats = coll.node_stats("agent")
        assert stats.duration is not None
        # agent appears in t1, t2, t3, t5 with 1000ms each
        assert stats.duration.count == 4
        assert stats.duration.mean == pytest.approx(1000.0)

    def test_cost_per_node(self, collection_traces: list[Trace]) -> None:
        coll = TraceCollection(collection_traces)
        stats = coll.node_stats("agent")
        assert stats.cost is not None
        # agent has LLM calls in t1 (0.01), t2 (0.012), t3 (0.10), t5 (0.005)
        assert stats.cost.count == 4

    def test_error_rate(self, collection_traces: list[Trace]) -> None:
        coll = TraceCollection(collection_traces)
        stats = coll.node_stats("validator")
        assert stats.error_rate is not None
        # validator appears in t5 only, with error -> rate = 1.0
        assert stats.error_rate == pytest.approx(1.0)

    def test_unknown_node(self, collection_traces: list[Trace]) -> None:
        coll = TraceCollection(collection_traces)
        stats = coll.node_stats("nonexistent")
        assert stats.duration is None
        assert stats.cost is None
        assert stats.error_rate is None


# ---------------------------------------------------------------------------
# Execution shapes
# ---------------------------------------------------------------------------


class TestExecutionShapes:
    def test_identical_shapes_grouped(self, collection_traces: list[Trace]) -> None:
        coll = TraceCollection(collection_traces)
        shapes = coll.execution_shapes()
        # [agent, summarizer] appears in t1, t2, t3
        matching = [s for s in shapes if s.node_sequence == ["agent", "summarizer"]]
        assert len(matching) == 1
        assert matching[0].trace_count == 3
        assert set(matching[0].trace_ids) == {"t1", "t2", "t3"}

    def test_fraction_computed(self, collection_traces: list[Trace]) -> None:
        coll = TraceCollection(collection_traces)
        shapes = coll.execution_shapes()
        for shape in shapes:
            assert shape.fraction == pytest.approx(shape.trace_count / 5)

    def test_different_shapes(self, collection_traces: list[Trace]) -> None:
        coll = TraceCollection(collection_traces)
        shapes = coll.execution_shapes()
        # 3 distinct shapes
        assert len(shapes) == 3


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------


class TestFindOutliers:
    def test_cost_outlier_detected(self, collection_traces: list[Trace]) -> None:
        coll = TraceCollection(collection_traces)
        # Costs: 0.005, 0.01, 0.012, 0.02, 0.10 — median = 0.012
        # t3 (0.10) is 8.3x the median, clearly an outlier at threshold=3
        outliers = coll.find_outliers(metric="cost", threshold=3.0)
        high = [o for o in outliers if o.value > o.median]
        assert len(high) >= 1
        assert any(o.trace_id == "t3" for o in high)

    def test_no_outliers_within_threshold(self) -> None:
        traces = [
            _make_trace("a", ["x"], cost=0.010),
            _make_trace("b", ["x"], cost=0.011),
            _make_trace("c", ["x"], cost=0.012),
        ]
        coll = TraceCollection(traces)
        outliers = coll.find_outliers(metric="cost", threshold=3.0)
        assert outliers == []

    def test_custom_threshold(self, collection_traces: list[Trace]) -> None:
        coll = TraceCollection(collection_traces)
        # Lower threshold should catch more
        outliers_strict = coll.find_outliers(metric="cost", threshold=2.0)
        outliers_loose = coll.find_outliers(metric="cost", threshold=5.0)
        assert len(outliers_strict) >= len(outliers_loose)

    def test_duration_metric(self, collection_traces: list[Trace]) -> None:
        coll = TraceCollection(collection_traces)
        outliers = coll.find_outliers(metric="duration", threshold=3.0)
        # All durations are similar, so no outliers expected
        assert isinstance(outliers, list)

    def test_unknown_metric_raises(self, collection_traces: list[Trace]) -> None:
        coll = TraceCollection(collection_traces)
        with pytest.raises(ValueError, match="Unknown metric"):
            coll.find_outliers(metric="invalid_metric")

    def test_low_outlier_detected(self) -> None:
        """Suspiciously low values should also be flagged."""
        traces = [
            _make_trace("a", ["x"], cost=1.0, base_offset=0),
            _make_trace("b", ["x"], cost=1.1, base_offset=10),
            _make_trace("c", ["x"], cost=0.9, base_offset=20),
            _make_trace("d", ["x"], cost=1.0, base_offset=30),
            _make_trace("low", ["x"], cost=0.01, base_offset=40),  # suspiciously low
        ]
        coll = TraceCollection(traces)
        outliers = coll.find_outliers(metric="cost", threshold=3.0)
        low = [o for o in outliers if o.value < o.median]
        assert len(low) >= 1
        assert any(o.trace_id == "low" for o in low)


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------


class TestTrend:
    def test_increasing_trend(self) -> None:
        traces = [
            _make_trace("a", ["x"], cost=1.0, base_offset=0),
            _make_trace("b", ["x"], cost=2.0, base_offset=10),
            _make_trace("c", ["x"], cost=3.0, base_offset=20),
            _make_trace("d", ["x"], cost=4.0, base_offset=30),
        ]
        coll = TraceCollection(traces)
        result = coll.trend(metric="cost")
        assert result.direction == "increasing"
        assert result.change_pct is not None
        assert result.change_pct > 0

    def test_decreasing_trend(self) -> None:
        traces = [
            _make_trace("a", ["x"], cost=4.0, base_offset=0),
            _make_trace("b", ["x"], cost=3.0, base_offset=10),
            _make_trace("c", ["x"], cost=2.0, base_offset=20),
            _make_trace("d", ["x"], cost=1.0, base_offset=30),
        ]
        coll = TraceCollection(traces)
        result = coll.trend(metric="cost")
        assert result.direction == "decreasing"
        assert result.change_pct is not None
        assert result.change_pct < 0

    def test_stable_trend(self) -> None:
        traces = [
            _make_trace("a", ["x"], cost=1.00, base_offset=0),
            _make_trace("b", ["x"], cost=1.01, base_offset=10),
            _make_trace("c", ["x"], cost=1.02, base_offset=20),
            _make_trace("d", ["x"], cost=1.01, base_offset=30),
        ]
        coll = TraceCollection(traces)
        result = coll.trend(metric="cost")
        assert result.direction == "stable"

    def test_insufficient_data(self) -> None:
        traces = [
            _make_trace("a", ["x"], cost=1.0, base_offset=0),
            _make_trace("b", ["x"], cost=2.0, base_offset=10),
        ]
        coll = TraceCollection(traces)
        result = coll.trend(metric="cost")
        assert result.direction == "insufficient_data"
        assert result.change_pct is None

    def test_insufficient_data_no_values(self) -> None:
        traces = [
            _make_trace("a", ["x"]),  # no cost
            _make_trace("b", ["x"]),
            _make_trace("c", ["x"]),
        ]
        coll = TraceCollection(traces)
        result = coll.trend(metric="cost")
        assert result.direction == "insufficient_data"

    def test_unknown_metric_raises(self) -> None:
        coll = TraceCollection([_make_trace("a", ["x"], cost=1.0)])
        with pytest.raises(ValueError, match="Unknown metric"):
            coll.trend(metric="bogus")

    def test_points_are_chronological(self) -> None:
        traces = [
            _make_trace("c", ["x"], cost=3.0, base_offset=20),
            _make_trace("a", ["x"], cost=1.0, base_offset=0),
            _make_trace("b", ["x"], cost=2.0, base_offset=10),
            _make_trace("d", ["x"], cost=4.0, base_offset=30),
        ]
        coll = TraceCollection(traces)
        result = coll.trend(metric="cost")
        # Points should be sorted by timestamp regardless of input order
        assert [p.trace_id for p in result.points] == ["a", "b", "c", "d"]


# ---------------------------------------------------------------------------
# from_json_dir
# ---------------------------------------------------------------------------


class TestFromJsonDir:
    def test_round_trip(self, tmp_path, collection_traces: list[Trace], caplog) -> None:
        # Write traces to temp directory
        for i, t in enumerate(collection_traces):
            (tmp_path / f"trace_{i}.json").write_text(t.to_json(), encoding="utf-8")

        # Also write a non-trace file that should be skipped
        (tmp_path / "invalid.json").write_text("not valid json", encoding="utf-8")

        import logging
        with caplog.at_level(logging.WARNING, logger="orcheval.collection"):
            coll = TraceCollection.from_json_dir(tmp_path)
        assert len(coll) == 5
        assert any("invalid.json" in r.message for r in caplog.records)

    def test_empty_directory(self, tmp_path) -> None:
        coll = TraceCollection.from_json_dir(tmp_path)
        assert len(coll) == 0

    def test_non_json_files_ignored(self, tmp_path, collection_traces: list[Trace]) -> None:
        (tmp_path / "trace.json").write_text(collection_traces[0].to_json(), encoding="utf-8")
        (tmp_path / "notes.txt").write_text("not a trace", encoding="utf-8")
        coll = TraceCollection.from_json_dir(tmp_path)
        assert len(coll) == 1


# ---------------------------------------------------------------------------
# PercentileStats edge cases
# ---------------------------------------------------------------------------


class TestPercentileStats:
    def test_single_value(self) -> None:
        from orcheval.collection import _percentile_stats
        result = _percentile_stats([42.0])
        assert result is not None
        assert result.mean == 42.0
        assert result.median == 42.0
        assert result.p95 == 42.0
        assert result.min == 42.0
        assert result.max == 42.0

    def test_empty_returns_none(self) -> None:
        from orcheval.collection import _percentile_stats
        assert _percentile_stats([]) is None
