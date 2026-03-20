"""Tests for the unified report API."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orcheval.report import FullReport, report
from orcheval.report.convergence import ConvergenceReport
from orcheval.report.cost import CostReport, cost_report
from orcheval.report.retries import RetryReport
from orcheval.report.routing import RoutingReport
from orcheval.report.timeline import TimelineReport
from orcheval.trace import Trace

from .conftest import TRACE_ID


class TestFullReport:
    def test_all_sections_present(self, sample_trace: Trace) -> None:
        result = report(sample_trace)
        assert isinstance(result, FullReport)
        assert isinstance(result.cost, CostReport)
        assert isinstance(result.routing, RoutingReport)
        assert isinstance(result.convergence, ConvergenceReport)
        assert isinstance(result.timeline, TimelineReport)
        assert isinstance(result.retries, RetryReport)

    def test_empty_trace(self) -> None:
        trace = Trace(events=[], trace_id=TRACE_ID)
        result = report(trace)
        assert isinstance(result, FullReport)
        assert result.cost.nodes == []
        assert result.routing.decisions == []
        assert result.convergence.passes == []
        assert result.timeline.spans == []
        assert result.retries.error_clusters == []

    def test_cost_matches_individual(self, sample_trace: Trace) -> None:
        full = report(sample_trace)
        individual = cost_report(sample_trace)
        assert full.cost.total_cost == individual.total_cost
        assert full.cost.total_tokens == individual.total_tokens
        assert len(full.cost.nodes) == len(individual.nodes)


class TestTopLevelExports:
    def test_import_from_orcheval(self) -> None:
        from orcheval import FullReport as FR
        from orcheval import report as r
        assert FR is FullReport
        assert r is report

    def test_frozen(self, sample_trace: Trace) -> None:
        result = report(sample_trace)
        with pytest.raises(ValidationError):
            result.cost = CostReport()  # type: ignore[misc]
