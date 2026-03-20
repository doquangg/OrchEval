"""Tests for orcheval.adapters.manual — ManualAdapter."""

from __future__ import annotations

from orcheval.adapters.manual import ManualAdapter
from orcheval.events import (
    ErrorEvent,
    LLMCall,
    NodeEntry,
    NodeExit,
    ToolCall,
)

TRACE_ID = "test-manual-adapter"


class TestManualAdapterBasics:
    def test_get_callback_handler_returns_self(self) -> None:
        adapter = ManualAdapter(trace_id=TRACE_ID)
        assert adapter.get_callback_handler() is adapter

    def test_trace_id(self) -> None:
        adapter = ManualAdapter(trace_id=TRACE_ID)
        assert adapter.trace_id == TRACE_ID

    def test_get_events_empty(self) -> None:
        adapter = ManualAdapter(trace_id=TRACE_ID)
        assert adapter.get_events() == []

    def test_reset_clears_events(self) -> None:
        adapter = ManualAdapter(trace_id=TRACE_ID)
        adapter.node_entry("agent")
        assert len(adapter.get_events()) == 1
        adapter.reset()
        assert len(adapter.get_events()) == 0

    def test_get_events_returns_copy(self) -> None:
        adapter = ManualAdapter(trace_id=TRACE_ID)
        adapter.node_entry("agent")
        events = adapter.get_events()
        events.clear()
        assert len(adapter.get_events()) == 1


class TestManualAdapterEmit:
    def test_emit_generic(self) -> None:
        adapter = ManualAdapter(trace_id=TRACE_ID)
        event = adapter.emit(NodeEntry, node_name="agent")
        assert isinstance(event, NodeEntry)
        assert event.trace_id == TRACE_ID
        assert event.node_name == "agent"
        assert len(adapter.get_events()) == 1

    def test_emit_stamps_trace_id(self) -> None:
        adapter = ManualAdapter(trace_id=TRACE_ID)
        event = adapter.emit(LLMCall, model="gpt-4o")
        assert event.trace_id == TRACE_ID


class TestManualAdapterConvenience:
    def test_node_entry(self) -> None:
        adapter = ManualAdapter(trace_id=TRACE_ID)
        event = adapter.node_entry("agent")
        assert isinstance(event, NodeEntry)
        assert event.node_name == "agent"
        assert event.trace_id == TRACE_ID

    def test_node_exit(self) -> None:
        adapter = ManualAdapter(trace_id=TRACE_ID)
        event = adapter.node_exit("agent", duration_ms=1500.0)
        assert isinstance(event, NodeExit)
        assert event.duration_ms == 1500.0

    def test_llm_call(self) -> None:
        adapter = ManualAdapter(trace_id=TRACE_ID)
        event = adapter.llm_call(model="gpt-4o", input_tokens=100, output_tokens=50, cost=0.003)
        assert isinstance(event, LLMCall)
        assert event.model == "gpt-4o"
        assert event.input_tokens == 100

    def test_tool_call(self) -> None:
        adapter = ManualAdapter(trace_id=TRACE_ID)
        event = adapter.tool_call("search", tool_input={"q": "test"}, tool_output="results")
        assert isinstance(event, ToolCall)
        assert event.tool_name == "search"

    def test_error(self) -> None:
        adapter = ManualAdapter(trace_id=TRACE_ID)
        event = adapter.error("ValueError", "bad input")
        assert isinstance(event, ErrorEvent)
        assert event.error_type == "ValueError"
        assert event.error_message == "bad input"

    def test_multiple_events_ordered(self) -> None:
        adapter = ManualAdapter(trace_id=TRACE_ID)
        adapter.node_entry("agent")
        adapter.llm_call(model="gpt-4o")
        adapter.node_exit("agent")
        events = adapter.get_events()
        assert len(events) == 3
        assert isinstance(events[0], NodeEntry)
        assert isinstance(events[1], LLMCall)
        assert isinstance(events[2], NodeExit)
