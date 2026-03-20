"""Tests for the cost report module."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orcheval.events import LLMCall
from orcheval.report.cost import UNKNOWN_MODEL, UNKNOWN_NODE, CostReport, cost_report
from orcheval.trace import Trace

from .conftest import TRACE_ID, _ts


class TestCostReportBasic:
    def test_empty_trace(self) -> None:
        trace = Trace(events=[], trace_id=TRACE_ID)
        result = cost_report(trace)
        assert isinstance(result, CostReport)
        assert result.nodes == []
        assert result.models == []
        assert result.total_cost is None
        assert result.most_expensive_node is None

    def test_totals_match_trace(self, sample_trace: Trace) -> None:
        result = cost_report(sample_trace)
        assert result.total_cost == sample_trace.total_cost()
        assert result.total_tokens == sample_trace.total_tokens()

    def test_node_breakdown(self, sample_trace: Trace) -> None:
        result = cost_report(sample_trace)
        node_names = [n.node_name for n in result.nodes]
        assert "agent" in node_names
        assert "summarizer" in node_names

        agent = next(n for n in result.nodes if n.node_name == "agent")
        assert agent.call_count == 1
        assert agent.total_cost == 0.005

        summarizer = next(n for n in result.nodes if n.node_name == "summarizer")
        assert summarizer.call_count == 1
        assert summarizer.total_cost == 0.002


class TestCostReportMultiModel:
    def test_per_model_breakdown(self, multi_model_events: list) -> None:
        trace = Trace(events=multi_model_events, trace_id=TRACE_ID)
        result = cost_report(trace)

        # Global model breakdown
        model_names = [m.model for m in result.models]
        assert "gpt-4o" in model_names
        assert "gpt-4o-mini" in model_names

        gpt4o = next(m for m in result.models if m.model == "gpt-4o")
        assert gpt4o.call_count == 2  # planner + executor
        assert gpt4o.input_tokens == 500  # 200 + 300
        assert gpt4o.output_tokens == 250  # 100 + 150
        assert gpt4o.total_cost == 0.020  # 0.008 + 0.012

        mini = next(m for m in result.models if m.model == "gpt-4o-mini")
        assert mini.call_count == 1
        assert mini.input_tokens == 50
        assert mini.total_cost == 0.001

    def test_per_node_model_detail(self, multi_model_events: list) -> None:
        trace = Trace(events=multi_model_events, trace_id=TRACE_ID)
        result = cost_report(trace)

        planner = next(n for n in result.nodes if n.node_name == "planner")
        assert planner.call_count == 2
        assert len(planner.models) == 2  # gpt-4o and gpt-4o-mini

    def test_most_expensive_node(self, multi_model_events: list) -> None:
        trace = Trace(events=multi_model_events, trace_id=TRACE_ID)
        result = cost_report(trace)
        # executor: 0.012, planner: 0.008 + 0.001 = 0.009
        assert result.most_expensive_node == "executor"

    def test_most_expensive_model(self, multi_model_events: list) -> None:
        trace = Trace(events=multi_model_events, trace_id=TRACE_ID)
        result = cost_report(trace)
        assert result.most_expensive_model == "gpt-4o"


class TestCostReportEdgeCases:
    def test_unknown_node(self) -> None:
        events = [
            LLMCall(
                trace_id=TRACE_ID, timestamp=_ts(0),
                model="gpt-4o", input_tokens=10, output_tokens=5, cost=0.001,
            ),
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = cost_report(trace)
        assert len(result.nodes) == 1
        assert result.nodes[0].node_name == UNKNOWN_NODE

    def test_unknown_model(self) -> None:
        events = [
            LLMCall(
                trace_id=TRACE_ID, timestamp=_ts(0), node_name="agent",
                input_tokens=10, output_tokens=5, cost=0.001,
            ),
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = cost_report(trace)
        assert result.nodes[0].models[0].model == UNKNOWN_MODEL

    def test_no_cost_data(self) -> None:
        events = [
            LLMCall(
                trace_id=TRACE_ID, timestamp=_ts(0), node_name="agent",
                model="gpt-4o", input_tokens=10, output_tokens=5,
            ),
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = cost_report(trace)
        assert result.nodes[0].total_cost is None
        assert result.nodes[0].models[0].total_cost is None
        assert result.nodes[0].models[0].avg_cost_per_call is None

    def test_avg_duration(self) -> None:
        events = [
            LLMCall(
                trace_id=TRACE_ID, timestamp=_ts(0), node_name="agent",
                model="gpt-4o", duration_ms=100.0,
            ),
            LLMCall(
                trace_id=TRACE_ID, timestamp=_ts(1), node_name="agent",
                model="gpt-4o", duration_ms=200.0,
            ),
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = cost_report(trace)
        assert result.nodes[0].models[0].avg_duration_ms == 150.0

    def test_frozen_result(self, sample_trace: Trace) -> None:
        result = cost_report(sample_trace)
        with pytest.raises(ValidationError):
            result.most_expensive_node = "hacked"  # type: ignore[misc]
