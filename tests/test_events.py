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

    @pytest.mark.parametrize(
        "cls,kwargs,expected_type",
        [
            (NodeEntry, {"node_name": "agent"}, "node_entry"),
            (NodeExit, {"node_name": "agent", "duration_ms": 1500.0}, "node_exit"),
            (
                LLMCall,
                {"model": "gpt-4o", "input_tokens": 100, "output_tokens": 50, "cost": 0.003},
                "llm_call",
            ),
            (
                ToolCall,
                {"tool_name": "search", "tool_input": {"query": "test"}, "tool_output": "results"},
                "tool_call",
            ),
            (
                RoutingDecision,
                {
                    "source_node": "router", "target_node": "agent",
                    "decision_context": {"score": 0.9},
                },
                "routing_decision",
            ),
            (
                AgentMessage,
                {"sender": "agent_1", "receiver": "agent_2", "content_summary": "Passing results"},
                "agent_message",
            ),
            (
                ErrorEvent,
                {
                    "error_type": "ValueError",
                    "error_message": "Something went wrong",
                    "stacktrace": "Traceback...",
                },
                "error_event",
            ),
            (
                PassBoundary,
                {
                    "pass_number": 1, "direction": "enter",
                    "metrics_snapshot": {"violation_count": 5},
                },
                "pass_boundary",
            ),
        ],
        ids=[
            "node_entry", "node_exit", "llm_call", "tool_call",
            "routing_decision", "agent_message", "error_event", "pass_boundary",
        ],
    )
    def test_event_construction(self, cls: type, kwargs: dict, expected_type: str) -> None:
        e = cls(trace_id=TRACE_ID, **kwargs)
        assert e.event_type == expected_type
        assert e.trace_id == TRACE_ID
        for key, value in kwargs.items():
            assert getattr(e, key) == value


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



class TestNewFields:
    """Test Tier 1 field additions."""

    def test_node_entry_input_state_default(self) -> None:
        e = NodeEntry(trace_id=TRACE_ID, node_name="agent")
        assert e.input_state == {}

    def test_node_entry_input_state_with_value(self) -> None:
        e = NodeEntry(trace_id=TRACE_ID, node_name="agent", input_state={"key": "val"})
        assert e.input_state == {"key": "val"}

    def test_node_exit_output_state_default(self) -> None:
        e = NodeExit(trace_id=TRACE_ID, node_name="agent")
        assert e.output_state == {}
        assert e.state_diff == {}

    def test_node_exit_output_state_with_value(self) -> None:
        e = NodeExit(
            trace_id=TRACE_ID,
            node_name="agent",
            output_state={"result": 42},
            state_diff={"added": ["result"], "removed": [], "modified": []},
        )
        assert e.output_state == {"result": 42}
        assert e.state_diff["added"] == ["result"]

    def test_llm_call_system_message_default(self) -> None:
        e = LLMCall(trace_id=TRACE_ID)
        assert e.system_message is None

    def test_llm_call_system_message_with_value(self) -> None:
        e = LLMCall(trace_id=TRACE_ID, system_message="You are a helpful assistant.")
        assert e.system_message == "You are a helpful assistant."

    def test_new_fields_round_trip(self) -> None:
        """New fields survive model_dump/validate round-trip."""
        entry = NodeEntry(
            trace_id=TRACE_ID, node_name="n", input_state={"x": 1}
        )
        data = entry.model_dump()
        restored = EVENT_ADAPTER.validate_python(data)
        assert isinstance(restored, NodeEntry)
        assert restored.input_state == {"x": 1}

        exit_ = NodeExit(
            trace_id=TRACE_ID,
            node_name="n",
            output_state={"y": 2},
            state_diff={"added": ["y"], "removed": [], "modified": []},
        )
        data = exit_.model_dump()
        restored = EVENT_ADAPTER.validate_python(data)
        assert isinstance(restored, NodeExit)
        assert restored.output_state == {"y": 2}
        assert restored.state_diff["added"] == ["y"]

        llm = LLMCall(trace_id=TRACE_ID, system_message="Be helpful")
        data = llm.model_dump()
        restored = EVENT_ADAPTER.validate_python(data)
        assert isinstance(restored, LLMCall)
        assert restored.system_message == "Be helpful"

    def test_backward_compat_missing_new_fields(self) -> None:
        """Old JSON without new fields deserializes with defaults."""
        old_entry = {
            "event_type": "node_entry",
            "trace_id": TRACE_ID,
            "node_name": "agent",
        }
        restored = EVENT_ADAPTER.validate_python(old_entry)
        assert isinstance(restored, NodeEntry)
        assert restored.input_state == {}

        old_exit = {
            "event_type": "node_exit",
            "trace_id": TRACE_ID,
            "node_name": "agent",
        }
        restored = EVENT_ADAPTER.validate_python(old_exit)
        assert isinstance(restored, NodeExit)
        assert restored.output_state == {}
        assert restored.state_diff == {}

        old_llm = {
            "event_type": "llm_call",
            "trace_id": TRACE_ID,
        }
        restored = EVENT_ADAPTER.validate_python(old_llm)
        assert isinstance(restored, LLMCall)
        assert restored.system_message is None


class TestEventFrozen:
    """Test that events are immutable after creation."""

    def test_cannot_mutate_field(self) -> None:
        e = NodeEntry(trace_id=TRACE_ID, node_name="agent")
        with pytest.raises(ValidationError):
            e.node_name = "other"  # type: ignore[misc]



class TestDiscriminatedUnion:
    """Test AnyEvent discriminated union parsing."""

    def test_parse_node_entry_dict(self) -> None:
        data = {"event_type": "node_entry", "trace_id": TRACE_ID, "node_name": "agent"}
        event = EVENT_ADAPTER.validate_python(data)
        assert isinstance(event, NodeEntry)
        assert event.node_name == "agent"

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
