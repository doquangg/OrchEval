"""Tests for orcheval.adapters.manual — ManualAdapter."""

from __future__ import annotations

from orcheval.adapters.manual import ManualAdapter
from orcheval.events import (
    AgentMessage,
    ErrorEvent,
    LLMCall,
    NodeEntry,
    NodeExit,
    PassBoundary,
    RoutingDecision,
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

    def test_llm_call_with_node_name(self) -> None:
        adapter = ManualAdapter(trace_id=TRACE_ID)
        event = adapter.llm_call(
            node_name="agent_1", model="gpt-4o", input_tokens=50, output_tokens=25
        )
        assert isinstance(event, LLMCall)
        assert event.node_name == "agent_1"
        assert event.model == "gpt-4o"

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

    def test_routing_decision(self) -> None:
        adapter = ManualAdapter(trace_id=TRACE_ID)
        event = adapter.routing_decision(
            "router", "target_a", decision_context={"score": 0.9}
        )
        assert isinstance(event, RoutingDecision)
        assert event.source_node == "router"
        assert event.target_node == "target_a"
        assert event.decision_context == {"score": 0.9}
        assert event.trace_id == TRACE_ID

    def test_agent_message(self) -> None:
        adapter = ManualAdapter(trace_id=TRACE_ID)
        event = adapter.agent_message("agent_1", "agent_2", content_summary="results")
        assert isinstance(event, AgentMessage)
        assert event.sender == "agent_1"
        assert event.receiver == "agent_2"
        assert event.content_summary == "results"
        assert event.trace_id == TRACE_ID

    def test_pass_boundary(self) -> None:
        adapter = ManualAdapter(trace_id=TRACE_ID)
        event = adapter.pass_boundary(1, "enter", metrics_snapshot={"count": 50})
        assert isinstance(event, PassBoundary)
        assert event.pass_number == 1
        assert event.direction == "enter"
        assert event.metrics_snapshot == {"count": 50}
        assert event.trace_id == TRACE_ID

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
