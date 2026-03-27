"""Universal event schema for orchestration tracing.

All event types use only primitive Python types (str, int, float, dict, list, None).
No framework-specific imports or objects are allowed in this module.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, field_validator


class Event(BaseModel):
    """Base event model. All orchestration events inherit from this."""

    model_config = {"frozen": True}

    event_type: str
    trace_id: str
    span_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    parent_span_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    node_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class NodeEntry(Event):
    """Emitted when a graph node begins execution."""

    event_type: Literal["node_entry"] = "node_entry"
    node_name: str
    input_state: dict[str, Any] = Field(default_factory=dict)


class NodeExit(Event):
    """Emitted when a graph node finishes execution."""

    event_type: Literal["node_exit"] = "node_exit"
    node_name: str
    duration_ms: float | None = None
    output_state: dict[str, Any] = Field(default_factory=dict)
    state_diff: dict[str, Any] = Field(default_factory=dict)


class LLMCall(Event):
    """Emitted when an LLM call completes."""

    event_type: Literal["llm_call"] = "llm_call"
    model: str | None = None
    input_messages: list[dict[str, Any]] = Field(default_factory=list)
    output_message: dict[str, Any] | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None
    duration_ms: float | None = None
    prompt_summary: str | None = None
    response_summary: str | None = None
    system_message: str | None = None

    @field_validator("input_tokens", "output_tokens", mode="before")
    @classmethod
    def _coerce_tokens_to_int(cls, v: Any) -> int | None:
        if v is None:
            return None
        return int(v)


class ToolCall(Event):
    """Emitted when a tool call completes."""

    event_type: Literal["tool_call"] = "tool_call"
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_output: str | None = None
    duration_ms: float | None = None


class RoutingDecision(Event):
    """Emitted when a routing/conditional edge decision is made."""

    event_type: Literal["routing_decision"] = "routing_decision"
    source_node: str
    target_node: str
    decision_context: dict[str, Any] = Field(default_factory=dict)


class AgentMessage(Event):
    """Emitted when an agent sends a message (primarily for swarm systems)."""

    event_type: Literal["agent_message"] = "agent_message"
    sender: str
    receiver: str
    content_summary: str | None = None


class ErrorEvent(Event):
    """Emitted when an error occurs during execution."""

    event_type: Literal["error_event"] = "error_event"
    error_type: str
    error_message: str
    stacktrace: str | None = None


class PassBoundary(Event):
    """Emitted at the boundary of a multi-pass processing cycle."""

    event_type: Literal["pass_boundary"] = "pass_boundary"
    pass_number: int
    direction: Literal["enter", "exit"]
    metrics_snapshot: dict[str, Any] = Field(default_factory=dict)

    @field_validator("pass_number", mode="before")
    @classmethod
    def _coerce_pass_number_to_int(cls, v: Any) -> int:
        return int(v)

    @field_validator("metrics_snapshot", mode="before")
    @classmethod
    def _coerce_metric_ints(cls, v: Any) -> dict[str, Any]:
        if not isinstance(v, dict):
            return v
        _INT_KEYS = {"violations_found", "rows_remaining", "steps_executed"}
        out = dict(v)
        for key in _INT_KEYS:
            if key in out and out[key] is not None:
                out[key] = int(out[key])
        return out


AnyEvent = Annotated[
    NodeEntry
    | NodeExit
    | LLMCall
    | ToolCall
    | RoutingDecision
    | AgentMessage
    | ErrorEvent
    | PassBoundary,
    Field(discriminator="event_type"),
]
"""Discriminated union of all event types. Use EVENT_ADAPTER to parse dicts."""

EVENT_ADAPTER: TypeAdapter[AnyEvent] = TypeAdapter(AnyEvent)
"""TypeAdapter for parsing raw dicts into the correct Event subclass."""

ALL_EVENT_TYPES = (
    NodeEntry,
    NodeExit,
    LLMCall,
    ToolCall,
    RoutingDecision,
    AgentMessage,
    ErrorEvent,
    PassBoundary,
)
