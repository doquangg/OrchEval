"""Tests for orcheval.events — event schema validation."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from orcheval.events import (
    EVENT_ADAPTER,
    AgentMessage,
    ErrorEvent,
    LLMCall,
    NodeEntry,
    NodeExit,
    PassBoundary,
    RoutingDecision,
    ToolCall,
)

TRACE_ID = "test-trace-events"


class TestEventConstruction:
    """Test that each event type can be constructed with valid data."""

    def test_node_entry(self) -> None:
        e = NodeEntry(trace_id=TRACE_ID, node_name="agent")
        assert e.event_type == "node_entry"
        assert e.node_name == "agent"
        assert e.trace_id == TRACE_ID

    def test_node_exit(self) -> None:
        e = NodeExit(trace_id=TRACE_ID, node_name="agent", duration_ms=1500.0)
        assert e.event_type == "node_exit"
        assert e.duration_ms == 1500.0

    def test_llm_call(self) -> None:
        e = LLMCall(
            trace_id=TRACE_ID,
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            cost=0.003,
            input_messages=[{"role": "user", "content": "hello"}],
            output_message={"role": "ai", "content": "hi"},
        )
        assert e.event_type == "llm_call"
        assert e.model == "gpt-4o"
        assert e.input_messages == [{"role": "user", "content": "hello"}]
        assert e.output_message == {"role": "ai", "content": "hi"}

    def test_tool_call(self) -> None:
        e = ToolCall(
            trace_id=TRACE_ID,
            tool_name="search",
            tool_input={"query": "test"},
            tool_output="results",
        )
        assert e.event_type == "tool_call"
        assert e.tool_name == "search"

    def test_routing_decision(self) -> None:
        e = RoutingDecision(
            trace_id=TRACE_ID,
            source_node="router",
            target_node="agent",
            decision_context={"score": 0.9},
        )
        assert e.event_type == "routing_decision"
        assert e.decision_context == {"score": 0.9}

    def test_agent_message(self) -> None:
        e = AgentMessage(
            trace_id=TRACE_ID,
            sender="agent_1",
            receiver="agent_2",
            content_summary="Passing results",
        )
        assert e.event_type == "agent_message"

    def test_error_event(self) -> None:
        e = ErrorEvent(
            trace_id=TRACE_ID,
            error_type="ValueError",
            error_message="Something went wrong",
            stacktrace="Traceback...",
        )
        assert e.event_type == "error_event"
        assert e.stacktrace == "Traceback..."

    def test_pass_boundary(self) -> None:
        e = PassBoundary(
            trace_id=TRACE_ID,
            pass_number=1,
            direction="enter",
            metrics_snapshot={"violation_count": 5},
        )
        assert e.event_type == "pass_boundary"
        assert e.direction == "enter"


class TestEventDefaults:
    """Test auto-generated fields."""

    def test_span_id_auto_generated(self) -> None:
        e = NodeEntry(trace_id=TRACE_ID, node_name="agent")
        assert isinstance(e.span_id, str)
        assert len(e.span_id) == 32  # uuid4 hex

    def test_span_id_unique(self) -> None:
        e1 = NodeEntry(trace_id=TRACE_ID, node_name="a")
        e2 = NodeEntry(trace_id=TRACE_ID, node_name="b")
        assert e1.span_id != e2.span_id

    def test_timestamp_auto_generated(self) -> None:
        e = NodeEntry(trace_id=TRACE_ID, node_name="agent")
        assert isinstance(e.timestamp, datetime)
        assert e.timestamp.tzinfo is not None

    def test_explicit_span_id_used(self) -> None:
        e = NodeEntry(trace_id=TRACE_ID, node_name="agent", span_id="my-span")
        assert e.span_id == "my-span"

    def test_parent_span_id_default_none(self) -> None:
        e = NodeEntry(trace_id=TRACE_ID, node_name="agent")
        assert e.parent_span_id is None

    def test_metadata_default_empty(self) -> None:
        e = NodeEntry(trace_id=TRACE_ID, node_name="agent")
        assert e.metadata == {}

    def test_node_name_default_none_on_base(self) -> None:
        e = LLMCall(trace_id=TRACE_ID)
        assert e.node_name is None


class TestEventFrozen:
    """Test that events are immutable after creation."""

    def test_cannot_mutate_field(self) -> None:
        e = NodeEntry(trace_id=TRACE_ID, node_name="agent")
        with pytest.raises(ValidationError):
            e.node_name = "other"  # type: ignore[misc]

    def test_cannot_mutate_trace_id(self) -> None:
        e = NodeEntry(trace_id=TRACE_ID, node_name="agent")
        with pytest.raises(ValidationError):
            e.trace_id = "new-id"  # type: ignore[misc]


class TestDiscriminatedUnion:
    """Test AnyEvent discriminated union parsing."""

    def test_parse_node_entry_dict(self) -> None:
        data = {"event_type": "node_entry", "trace_id": TRACE_ID, "node_name": "agent"}
        event = EVENT_ADAPTER.validate_python(data)
        assert isinstance(event, NodeEntry)
        assert event.node_name == "agent"

    def test_parse_llm_call_dict(self) -> None:
        data = {
            "event_type": "llm_call",
            "trace_id": TRACE_ID,
            "model": "gpt-4o",
            "input_messages": [{"role": "user", "content": "hello"}],
        }
        event = EVENT_ADAPTER.validate_python(data)
        assert isinstance(event, LLMCall)
        assert event.model == "gpt-4o"

    def test_parse_invalid_event_type_raises(self) -> None:
        data = {"event_type": "nonexistent", "trace_id": TRACE_ID}
        with pytest.raises(ValidationError):
            EVENT_ADAPTER.validate_python(data)

    def test_round_trip_serialization(self) -> None:
        original = ToolCall(
            trace_id=TRACE_ID,
            tool_name="search",
            tool_input={"query": "test"},
            tool_output="results",
            duration_ms=150.0,
        )
        data = original.model_dump()
        restored = EVENT_ADAPTER.validate_python(data)
        assert isinstance(restored, ToolCall)
        assert restored.tool_name == original.tool_name
        assert restored.tool_output == original.tool_output
        assert restored.duration_ms == original.duration_ms

    def test_all_event_types_parseable(self) -> None:
        """Ensure every concrete event type round-trips through the adapter."""
        events = [
            NodeEntry(trace_id=TRACE_ID, node_name="n"),
            NodeExit(trace_id=TRACE_ID, node_name="n"),
            LLMCall(trace_id=TRACE_ID),
            ToolCall(trace_id=TRACE_ID, tool_name="t"),
            RoutingDecision(trace_id=TRACE_ID, source_node="a", target_node="b"),
            AgentMessage(trace_id=TRACE_ID, sender="a", receiver="b"),
            ErrorEvent(trace_id=TRACE_ID, error_type="E", error_message="msg"),
            PassBoundary(trace_id=TRACE_ID, pass_number=1, direction="enter"),
        ]
        for event in events:
            data = event.model_dump()
            restored = EVENT_ADAPTER.validate_python(data)
            assert type(restored) is type(event)
