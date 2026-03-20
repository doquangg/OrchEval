"""Tests for orcheval.Tracer — public API."""

from __future__ import annotations

from typing import Any

import pytest

from orcheval import Trace, Tracer
from orcheval.adapters.base import BaseAdapter
from orcheval.adapters.manual import ManualAdapter


class _DummyAdapter(BaseAdapter):
    """Minimal adapter for testing custom adapter support."""

    def get_callback_handler(self) -> Any:
        return self


class TestTracerCreation:
    def test_default_adapter_is_manual(self) -> None:
        tracer = Tracer()
        assert isinstance(tracer.adapter, ManualAdapter)

    def test_explicit_manual(self) -> None:
        tracer = Tracer(adapter="manual")
        assert isinstance(tracer.adapter, ManualAdapter)

    def test_custom_adapter_instance(self) -> None:
        custom = _DummyAdapter(trace_id="custom")
        tracer = Tracer(adapter=custom)
        assert tracer.adapter is custom

    def test_unknown_adapter_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown adapter"):
            Tracer(adapter="nonexistent")

    def test_invalid_adapter_type_raises(self) -> None:
        with pytest.raises(TypeError):
            Tracer(adapter=42)  # type: ignore[arg-type]

    def test_custom_trace_id(self) -> None:
        tracer = Tracer(trace_id="my-trace-123")
        assert tracer.trace_id == "my-trace-123"

    def test_auto_trace_id(self) -> None:
        tracer = Tracer()
        assert isinstance(tracer.trace_id, str)
        assert len(tracer.trace_id) == 32  # uuid4 hex


class TestTracerLangGraph:
    def test_langgraph_adapter_creation(self) -> None:
        pytest.importorskip("langchain_core")
        tracer = Tracer(adapter="langgraph")
        from orcheval.adapters.langgraph import LangGraphAdapter
        assert isinstance(tracer.adapter, LangGraphAdapter)


class TestTracerOperations:
    def test_handler_returns_adapter_handler(self) -> None:
        tracer = Tracer()
        # For ManualAdapter, handler is the adapter itself
        assert tracer.handler is tracer.adapter

    def test_collect_returns_trace(self) -> None:
        tracer = Tracer()
        trace = tracer.collect()
        assert isinstance(trace, Trace)
        assert len(trace) == 0

    def test_collect_includes_events(self) -> None:
        tracer = Tracer()
        adapter = tracer.adapter
        assert isinstance(adapter, ManualAdapter)
        adapter.node_entry("agent")
        adapter.llm_call(model="gpt-4o")
        adapter.node_exit("agent")

        trace = tracer.collect()
        assert len(trace) == 3

    def test_collect_uses_tracer_trace_id(self) -> None:
        tracer = Tracer(trace_id="my-id")
        trace = tracer.collect()
        assert trace.trace_id == "my-id"

    def test_reset_clears_events(self) -> None:
        tracer = Tracer()
        adapter = tracer.adapter
        assert isinstance(adapter, ManualAdapter)
        adapter.node_entry("agent")
        assert len(tracer.collect()) == 1

        tracer.reset()
        assert len(tracer.collect()) == 0

    def test_collect_multiple_times(self) -> None:
        """collect() should return the same events each time (until reset)."""
        tracer = Tracer()
        adapter = tracer.adapter
        assert isinstance(adapter, ManualAdapter)
        adapter.node_entry("agent")

        t1 = tracer.collect()
        t2 = tracer.collect()
        assert len(t1) == len(t2) == 1
