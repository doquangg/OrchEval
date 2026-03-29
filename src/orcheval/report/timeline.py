"""End-to-end execution timeline with node spans and event markers."""

from __future__ import annotations

from datetime import datetime  # noqa: TCH003 — Pydantic needs this at runtime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

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

if TYPE_CHECKING:
    from orcheval.events import Event
    from orcheval.trace import Trace


class TimelineEvent(BaseModel):
    """A single event positioned on the timeline."""

    model_config = {"frozen": True}

    event_type: str
    offset_ms: float
    node_name: str | None = None
    summary: str
    duration_ms: float | None = None


class TimelineSpan(BaseModel):
    """A node execution span with nested child events."""

    model_config = {"frozen": True}

    node_name: str
    span_id: str
    start_ms: float
    end_ms: float | None = None
    duration_ms: float | None = None
    children: list[TimelineEvent] = Field(default_factory=list)


class TimelineReport(BaseModel):
    """Full execution timeline showing spans and events."""

    model_config = {"frozen": True}

    spans: list[TimelineSpan] = Field(default_factory=list)
    total_duration_ms: float | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None

    @property
    def events(self) -> list[TimelineEvent]:
        """Flat list of all events across spans, sorted by offset."""
        return sorted(
            (e for span in self.spans for e in span.children),
            key=lambda e: e.offset_ms,
        )


def _offset_ms(event_ts: datetime, start_ts: datetime) -> float:
    return (event_ts - start_ts).total_seconds() * 1000


def _event_summary(event: Event) -> str:
    """Generate a human-readable one-liner for an event."""
    if isinstance(event, NodeEntry):
        return f"Enter {event.node_name}"
    if isinstance(event, NodeExit):
        dur = f" ({event.duration_ms:.0f}ms)" if event.duration_ms is not None else ""
        return f"Exit {event.node_name}{dur}"
    if isinstance(event, LLMCall):
        model = event.model or "unknown"
        tok = ""
        if event.input_tokens is not None and event.output_tokens is not None:
            tok = f" ({event.input_tokens}+{event.output_tokens} tokens)"
        return f"LLM call to {model}{tok}"
    if isinstance(event, ToolCall):
        return f"Tool call: {event.tool_name}"
    if isinstance(event, RoutingDecision):
        return f"Route: {event.source_node} -> {event.target_node}"
    if isinstance(event, ErrorEvent):
        return f"Error: {event.error_type}: {event.error_message}"
    if isinstance(event, PassBoundary):
        return f"Pass {event.pass_number} {event.direction}"
    if isinstance(event, AgentMessage):
        return f"Message: {event.sender} -> {event.receiver}"
    return f"{event.event_type}"


def _event_duration(event: Event) -> float | None:
    """Extract duration_ms if the event type carries one."""
    if isinstance(event, (NodeExit, LLMCall, ToolCall)):
        return event.duration_ms
    return None


def timeline_report(trace: Trace) -> TimelineReport:
    """Generate an end-to-end execution timeline."""
    timeline = trace.get_timeline()
    if not timeline:
        return TimelineReport()

    start_ts = timeline[0].timestamp
    end_ts = timeline[-1].timestamp

    # Build spans from NodeEntry/NodeExit pairs (matched by span_id)
    entries: dict[str, tuple[NodeEntry, float]] = {}  # span_id -> (entry, start_ms)
    exits: dict[str, tuple[NodeExit, float]] = {}  # span_id -> (exit, end_ms)

    for event in timeline:
        if isinstance(event, NodeEntry):
            entries[event.span_id] = (event, _offset_ms(event.timestamp, start_ts))
        elif isinstance(event, NodeExit):
            exits[event.span_id] = (event, _offset_ms(event.timestamp, start_ts))

    # Collect child events by parent_span_id
    children_by_span: dict[str, list[TimelineEvent]] = {}
    for event in timeline:
        if event.parent_span_id and event.parent_span_id in entries:
            if event.parent_span_id not in children_by_span:
                children_by_span[event.parent_span_id] = []
            children_by_span[event.parent_span_id].append(TimelineEvent(
                event_type=event.event_type,
                offset_ms=_offset_ms(event.timestamp, start_ts),
                node_name=event.node_name,
                summary=_event_summary(event),
                duration_ms=_event_duration(event),
            ))

    # Assemble spans in chronological order of entry
    spans: list[TimelineSpan] = []
    for span_id, (entry, start_ms) in sorted(entries.items(), key=lambda x: x[1][1]):
        if span_id in exits:
            exit_event, end_ms = exits[span_id]
            duration = exit_event.duration_ms
            if duration is None:
                duration = end_ms - start_ms
        else:
            end_ms = None
            duration = None

        spans.append(TimelineSpan(
            node_name=entry.node_name,
            span_id=span_id,
            start_ms=start_ms,
            end_ms=end_ms,
            duration_ms=duration,
            children=children_by_span.get(span_id, []),
        ))

    return TimelineReport(
        spans=spans,
        total_duration_ms=trace.total_duration(),
        start_time=start_ts,
        end_time=end_ts,
    )
