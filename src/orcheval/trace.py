"""Trace container for collected orchestration events."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar, overload

if TYPE_CHECKING:
    from collections.abc import Iterator

from orcheval.events import (
    EVENT_ADAPTER,
    Event,
    LLMCall,
    NodeEntry,
    NodeExit,
    ToolCall,
)

E = TypeVar("E", bound=Event)


class NodeInvocation(NamedTuple):
    """A single node invocation with its span ID and duration."""

    node_name: str
    span_id: str
    duration_ms: float | None


class Trace:
    """Container holding a list of events from an orchestration run.

    Events are sorted by timestamp. Provides query and summary methods
    for analysis.
    """

    __slots__ = ("_events", "_trace_id")

    def __init__(self, events: list[Event], trace_id: str | None = None) -> None:
        self._events = sorted(events, key=lambda e: e.timestamp)
        self._trace_id = trace_id or (events[0].trace_id if events else "")

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    @property
    def trace_id(self) -> str:
        return self._trace_id

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    @overload
    def __getitem__(self, index: int) -> Event: ...

    @overload
    def __getitem__(self, index: slice) -> list[Event]: ...

    def __getitem__(self, index: int | slice) -> Event | list[Event]:
        return self._events[index]

    def __len__(self) -> int:
        return len(self._events)

    def __bool__(self) -> bool:
        return len(self._events) > 0

    # --- Query methods ---

    def get_events_by_type(self, event_type: type[E]) -> list[E]:
        """Return all events matching the given type."""
        return [e for e in self._events if isinstance(e, event_type)]

    def get_events_by_node(self, node_name: str) -> list[Event]:
        """Return all events associated with the given node name."""
        return [e for e in self._events if e.node_name == node_name]

    def get_llm_calls(self) -> list[LLMCall]:
        """Return all LLM call events."""
        return self.get_events_by_type(LLMCall)

    def get_tool_calls(self) -> list[ToolCall]:
        """Return all tool call events."""
        return self.get_events_by_type(ToolCall)

    def get_timeline(self) -> list[Event]:
        """Return all events in chronological order."""
        return list(self._events)

    # --- Summary methods ---

    def total_duration(self) -> float | None:
        """Total wall-clock duration in milliseconds (first to last event)."""
        if len(self._events) < 2:
            return None
        first = self._events[0].timestamp
        last = self._events[-1].timestamp
        delta = last - first
        return delta.total_seconds() * 1000

    def total_cost(self) -> float | None:
        """Sum of all LLM call costs. Returns None if no costs are recorded."""
        costs = [e.cost for e in self.get_llm_calls() if e.cost is not None]
        return sum(costs) if costs else None

    def total_tokens(self) -> dict[str, int]:
        """Sum of prompt and completion tokens across all LLM calls."""
        prompt = 0
        completion = 0
        for e in self.get_llm_calls():
            if e.input_tokens is not None:
                prompt += e.input_tokens
            if e.output_tokens is not None:
                completion += e.output_tokens
        return {"prompt": prompt, "completion": completion, "total": prompt + completion}

    def node_durations(self) -> dict[str, float]:
        """Total duration per node in ms, computed from NodeEntry/NodeExit pairs.

        Pairs are matched by node_name and span_id. If a NodeExit has
        duration_ms set, that value is used; otherwise duration is computed
        from the timestamp difference between entry and exit.
        """
        entries: dict[str, NodeEntry] = {}  # span_id -> NodeEntry
        durations: dict[str, float] = {}  # node_name -> total ms

        for event in self._events:
            if isinstance(event, NodeEntry):
                entries[event.span_id] = event
            elif isinstance(event, NodeExit):
                if event.duration_ms is not None:
                    ms = event.duration_ms
                elif event.span_id in entries:
                    entry = entries[event.span_id]
                    delta = event.timestamp - entry.timestamp
                    ms = delta.total_seconds() * 1000
                else:
                    continue
                durations[event.node_name] = durations.get(event.node_name, 0.0) + ms

        return durations

    def node_invocations(self) -> list[NodeInvocation]:
        """Per-invocation breakdown of node executions.

        Unlike node_durations() which aggregates by node name, this returns
        one entry per NodeEntry/NodeExit pair. Useful when nodes execute
        multiple times (retries, loops).
        """
        entries: dict[str, NodeEntry] = {}  # span_id -> NodeEntry
        result: list[NodeInvocation] = []

        for event in self._events:
            if isinstance(event, NodeEntry):
                entries[event.span_id] = event
            elif isinstance(event, NodeExit):
                if event.duration_ms is not None:
                    ms: float | None = event.duration_ms
                elif event.span_id in entries:
                    entry = entries[event.span_id]
                    delta = event.timestamp - entry.timestamp
                    ms = delta.total_seconds() * 1000
                else:
                    ms = None
                result.append(NodeInvocation(event.node_name, event.span_id, ms))

        return result

    def node_sequence(self) -> list[str]:
        """Ordered list of node names as they were entered."""
        return [e.node_name for e in self._events if isinstance(e, NodeEntry)]

    # --- Merge ---

    @classmethod
    def merge(cls, *traces: Trace) -> Trace:
        """Combine multiple traces into one, re-sorted by timestamp."""
        all_events: list[Event] = []
        for t in traces:
            all_events.extend(t._events)
        return cls(events=all_events)

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trace to a dict (with Python-native types)."""
        return {
            "trace_id": self._trace_id,
            "events": [e.model_dump() for e in self._events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trace:
        """Deserialize a trace from a dict produced by to_dict()."""
        trace_id = data["trace_id"]
        events: list[Event] = [EVENT_ADAPTER.validate_python(e) for e in data["events"]]
        return cls(events=events, trace_id=trace_id)

    def to_json(self) -> str:
        """Serialize the trace to a JSON string."""
        return json.dumps({
            "trace_id": self._trace_id,
            "events": [e.model_dump(mode="json") for e in self._events],
        })

    @classmethod
    def from_json(cls, data: str) -> Trace:
        """Deserialize a trace from a JSON string produced by to_json()."""
        return cls.from_dict(json.loads(data))
