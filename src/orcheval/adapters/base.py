"""Abstract base adapter for orchestration tracing."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from orcheval.events import Event

_log = logging.getLogger(__name__)


class BaseAdapter(ABC):
    """Base class for all framework-specific adapters.

    Subclasses translate framework callbacks/logs into universal Event objects.
    """

    def __init__(self, trace_id: str) -> None:
        self._trace_id = trace_id
        self._events: list[Event] = []

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @abstractmethod
    def get_callback_handler(self) -> Any:
        """Return a framework-specific callback/handler object.

        For frameworks with callback systems (e.g. LangGraph), this returns
        the handler to pass to the framework. For manual adapters, returns self.
        """
        ...

    def get_events(self) -> list[Event]:
        """Return a copy of all collected events."""
        return list(self._events)

    def _emit(self, event: Event) -> None:
        """Record an event. Called by subclasses."""
        _log.debug("%s event for node=%s", event.event_type, event.node_name)
        self._events.append(event)

    def reset(self) -> None:
        """Clear all collected events."""
        self._events.clear()
