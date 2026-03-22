"""Tests for the DataFrame export."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

import pytest

pd = pytest.importorskip("pandas")

from orcheval.export.dataframe import ALL_COLUMNS  # noqa: E402
from orcheval.events import (  # noqa: E402
    AgentMessage,
    ErrorEvent,
    LLMCall,
    NodeEntry,
    NodeExit,
    PassBoundary,
    RoutingDecision,
    ToolCall,
)
from orcheval.trace import Trace  # noqa: E402

from .conftest import BASE_TIME, TRACE_ID  # noqa: E402


def _ts(seconds: float):
    return BASE_TIME + timedelta(seconds=seconds)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def all_types_trace() -> Trace:
    """Trace with one event of every type for column coverage testing."""
    return Trace(events=[
        NodeEntry(
            trace_id=TRACE_ID, span_id="s-entry", timestamp=_ts(0), node_name="agent",
            input_state={"key": "value"},
        ),
        LLMCall(
            trace_id=TRACE_ID, span_id="s-llm", parent_span_id="s-entry",
            timestamp=_ts(1), node_name="agent",
            model="gpt-4o", input_tokens=100, output_tokens=50,
            cost=0.005, duration_ms=800.0,
        ),
        ToolCall(
            trace_id=TRACE_ID, span_id="s-tool", parent_span_id="s-entry",
            timestamp=_ts(2), node_name="agent",
            tool_name="search", duration_ms=300.0,
        ),
        ErrorEvent(
            trace_id=TRACE_ID, span_id="s-err", parent_span_id="s-entry",
            timestamp=_ts(3), node_name="agent",
            error_type="ValueError", error_message="bad input",
        ),
        NodeExit(
            trace_id=TRACE_ID, span_id="s-entry", timestamp=_ts(4),
            node_name="agent", duration_ms=4000.0,
            output_state={"key": "value", "result": "done"},
        ),
        RoutingDecision(
            trace_id=TRACE_ID, timestamp=_ts(5), node_name="router",
            source_node="agent", target_node="summarizer",
        ),
        AgentMessage(
            trace_id=TRACE_ID, timestamp=_ts(6), node_name="agent",
            sender="agent", receiver="summarizer", content_summary="results",
        ),
        PassBoundary(
            trace_id=TRACE_ID, timestamp=_ts(7),
            pass_number=1, direction="enter",
        ),
    ], trace_id=TRACE_ID)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


class TestDataFrameStructure:
    def test_row_count_matches_events(self, all_types_trace: Trace) -> None:
        df = all_types_trace.to_dataframe()
        assert len(df) == len(all_types_trace)

    def test_common_columns_present(self, all_types_trace: Trace) -> None:
        df = all_types_trace.to_dataframe()
        for col in ("event_type", "timestamp", "trace_id", "node_name", "span_id",
                     "parent_span_id"):
            assert col in df.columns, f"Missing common column: {col}"

    def test_type_specific_columns_present(self, all_types_trace: Trace) -> None:
        df = all_types_trace.to_dataframe()
        expected = [
            "duration_ms", "model", "input_tokens", "output_tokens", "cost",
            "tool_name", "error_type", "error_message", "source_node", "target_node",
            "sender", "receiver", "content_summary", "pass_number", "direction",
            "has_state_data",
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_column_order_matches_spec(self, all_types_trace: Trace) -> None:
        df = all_types_trace.to_dataframe()
        assert list(df.columns) == list(ALL_COLUMNS)


# ---------------------------------------------------------------------------
# Type-specific values
# ---------------------------------------------------------------------------


class TestDataFrameValues:
    def test_llm_call_values(self, all_types_trace: Trace) -> None:
        df = all_types_trace.to_dataframe()
        llm_rows = df[df["event_type"] == "llm_call"]
        assert len(llm_rows) == 1
        row = llm_rows.iloc[0]
        assert row["model"] == "gpt-4o"
        assert row["input_tokens"] == 100
        assert row["output_tokens"] == 50
        assert row["cost"] == pytest.approx(0.005)
        assert row["duration_ms"] == pytest.approx(800.0)

    def test_tool_call_values(self, all_types_trace: Trace) -> None:
        df = all_types_trace.to_dataframe()
        tool_rows = df[df["event_type"] == "tool_call"]
        assert len(tool_rows) == 1
        row = tool_rows.iloc[0]
        assert row["tool_name"] == "search"
        assert row["duration_ms"] == pytest.approx(300.0)

    def test_error_event_values(self, all_types_trace: Trace) -> None:
        df = all_types_trace.to_dataframe()
        err_rows = df[df["event_type"] == "error_event"]
        assert len(err_rows) == 1
        row = err_rows.iloc[0]
        assert row["error_type"] == "ValueError"
        assert row["error_message"] == "bad input"

    def test_routing_values(self, all_types_trace: Trace) -> None:
        df = all_types_trace.to_dataframe()
        rd_rows = df[df["event_type"] == "routing_decision"]
        assert len(rd_rows) == 1
        row = rd_rows.iloc[0]
        assert row["source_node"] == "agent"
        assert row["target_node"] == "summarizer"

    def test_agent_message_values(self, all_types_trace: Trace) -> None:
        df = all_types_trace.to_dataframe()
        msg_rows = df[df["event_type"] == "agent_message"]
        assert len(msg_rows) == 1
        row = msg_rows.iloc[0]
        assert row["sender"] == "agent"
        assert row["receiver"] == "summarizer"
        assert row["content_summary"] == "results"

    def test_pass_boundary_values(self, all_types_trace: Trace) -> None:
        df = all_types_trace.to_dataframe()
        pb_rows = df[df["event_type"] == "pass_boundary"]
        assert len(pb_rows) == 1
        row = pb_rows.iloc[0]
        assert row["pass_number"] == 1
        assert row["direction"] == "enter"

    def test_none_for_inapplicable(self, all_types_trace: Trace) -> None:
        df = all_types_trace.to_dataframe()
        entry_rows = df[df["event_type"] == "node_entry"]
        row = entry_rows.iloc[0]
        assert pd.isna(row["cost"])
        assert pd.isna(row["model"])
        assert pd.isna(row["tool_name"])
        assert pd.isna(row["error_type"])


# ---------------------------------------------------------------------------
# has_state_data
# ---------------------------------------------------------------------------


class TestHasStateData:
    def test_true_for_entry_with_state(self, all_types_trace: Trace) -> None:
        df = all_types_trace.to_dataframe()
        entry_rows = df[df["event_type"] == "node_entry"]
        # The NodeEntry in all_types_trace has input_state={"key": "value"}
        assert entry_rows.iloc[0]["has_state_data"] == True  # noqa: E712

    def test_true_for_exit_with_state(self, all_types_trace: Trace) -> None:
        df = all_types_trace.to_dataframe()
        exit_rows = df[df["event_type"] == "node_exit"]
        # The NodeExit has output_state={"key": "value", "result": "done"}
        assert exit_rows.iloc[0]["has_state_data"] == True  # noqa: E712

    def test_false_for_non_state_events(self, all_types_trace: Trace) -> None:
        df = all_types_trace.to_dataframe()
        llm_rows = df[df["event_type"] == "llm_call"]
        assert llm_rows.iloc[0]["has_state_data"] == False  # noqa: E712

    def test_false_for_entry_without_state(self) -> None:
        trace = Trace(events=[
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="x"),
            NodeExit(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(1),
                     node_name="x", duration_ms=1000.0),
        ], trace_id=TRACE_ID)
        df = trace.to_dataframe()
        entry_rows = df[df["event_type"] == "node_entry"]
        assert entry_rows.iloc[0]["has_state_data"] == False  # noqa: E712


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestDataFrameEdgeCases:
    def test_empty_trace(self) -> None:
        trace = Trace(events=[], trace_id=TRACE_ID)
        df = trace.to_dataframe()
        assert len(df) == 0
        assert list(df.columns) == list(ALL_COLUMNS)

    def test_import_error_message(self) -> None:
        with patch.dict("sys.modules", {"pandas": None}):
            # Force re-import to trigger the ImportError path
            import importlib

            import orcheval.export.dataframe as df_mod

            importlib.reload(df_mod)
            trace = Trace(events=[], trace_id=TRACE_ID)
            with pytest.raises(ImportError, match="pip install orcheval"):
                df_mod.build_dataframe(trace)
            # Restore
            importlib.reload(df_mod)


# ---------------------------------------------------------------------------
# Integration with sample_trace fixture
# ---------------------------------------------------------------------------


class TestDataFrameWithSampleTrace:
    def test_sample_trace(self, sample_trace: Trace) -> None:
        df = sample_trace.to_dataframe()
        assert len(df) == len(sample_trace)
        # sample_trace has: 2 NodeEntry, 2 NodeExit, 2 LLMCall, 1 ToolCall
        assert len(df[df["event_type"] == "node_entry"]) == 2
        assert len(df[df["event_type"] == "llm_call"]) == 2
        assert len(df[df["event_type"] == "tool_call"]) == 1
