"""Tests for orcheval.adapters.openai_agents — OpenAI Agents SDK adapter.

These tests require the ``openai-agents`` package to be installed.
They are skipped otherwise.
"""

from __future__ import annotations

from typing import Any

import pytest

agents_mod = pytest.importorskip("agents")

from agents.tracing import (  # noqa: E402
    AgentSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
)
from agents.tracing.spans import SpanImpl  # noqa: E402

from orcheval.adapters.openai_agents import OpenAIAgentsAdapter  # noqa: E402
from orcheval.events import (  # noqa: E402
    AgentMessage,
    ErrorEvent,
    LLMCall,
    NodeEntry,
    NodeExit,
    RoutingDecision,
    ToolCall,
)

TRACE_ID = "test-openai-agents"

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _NoOpProcessor:
    """Minimal processor for constructing SpanImpl objects in tests."""

    def on_span_start(self, span: Any) -> None:
        pass

    def on_span_end(self, span: Any) -> None:
        pass

    def on_trace_start(self, trace: Any) -> None:
        pass

    def on_trace_end(self, trace: Any) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


_noop = _NoOpProcessor()


def _make_span(
    span_data: Any,
    *,
    parent_id: str | None = None,
    span_id: str | None = None,
    trace_id: str = "sdk-trace",
) -> SpanImpl:  # type: ignore[type-arg]
    """Create a real SDK ``SpanImpl`` for testing."""
    span = SpanImpl(
        trace_id=trace_id,
        span_id=span_id,
        parent_id=parent_id,
        processor=_noop,
        span_data=span_data,
        tracing_api_key=None,
    )
    span.start()
    span.finish()
    return span


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestOpenAIAgentsAdapterBasics:
    def test_creates_processor(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        handler = adapter.get_callback_handler()
        assert handler is not None

    def test_trace_id(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        assert adapter.trace_id == TRACE_ID

    def test_empty_events(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        assert adapter.get_events() == []

    def test_reset(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        proc = adapter.get_callback_handler()
        span = _make_span(AgentSpanData(name="agent"))
        proc.on_span_start(span)
        assert len(adapter.get_events()) == 1
        adapter.reset()
        assert len(adapter.get_events()) == 0


class TestAgentSpanTracking:
    """AgentSpan -> NodeEntry / NodeExit."""

    def test_agent_span_emits_node_entry(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        proc = adapter.get_callback_handler()
        span = _make_span(AgentSpanData(name="planner"))

        proc.on_span_start(span)

        events = adapter.get_events()
        assert len(events) == 1
        assert isinstance(events[0], NodeEntry)
        assert events[0].node_name == "planner"
        assert events[0].trace_id == TRACE_ID

    def test_agent_span_end_emits_node_exit(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        proc = adapter.get_callback_handler()
        span = _make_span(AgentSpanData(name="planner"))

        proc.on_span_start(span)
        proc.on_span_end(span)

        events = adapter.get_events()
        assert len(events) == 2
        assert isinstance(events[1], NodeExit)
        assert events[1].node_name == "planner"
        assert events[1].duration_ms is not None
        assert events[1].duration_ms >= 0

    def test_entry_exit_share_orcheval_span_id(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        proc = adapter.get_callback_handler()
        span = _make_span(AgentSpanData(name="agent"))

        proc.on_span_start(span)
        proc.on_span_end(span)

        events = adapter.get_events()
        assert events[0].span_id == events[1].span_id

    def test_unknown_agent_name(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        proc = adapter.get_callback_handler()
        span = _make_span(AgentSpanData(name=""))

        proc.on_span_start(span)

        events = adapter.get_events()
        assert events[0].node_name == "unknown_agent"


class TestGenerationSpanTracking:
    """GenerationSpan -> LLMCall."""

    def _run_generation(
        self,
        adapter: OpenAIAgentsAdapter,
        *,
        model: str | None = "gpt-4o",
        input_msgs: list[dict[str, Any]] | None = None,
        output_msgs: list[dict[str, Any]] | None = None,
        usage: dict[str, Any] | None = None,
        parent_id: str | None = None,
    ) -> None:
        proc = adapter.get_callback_handler()
        span_data = GenerationSpanData(
            model=model,
            input=input_msgs,
            output=output_msgs,
            usage=usage,
        )
        span = _make_span(span_data, parent_id=parent_id)
        proc.on_span_start(span)
        proc.on_span_end(span)

    def test_basic_generation(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        self._run_generation(
            adapter,
            model="gpt-4o",
            input_msgs=[{"role": "user", "content": "Hello"}],
            output_msgs=[{"role": "assistant", "content": "Hi there"}],
            usage={"input_tokens": 10, "output_tokens": 5},
        )

        events = adapter.get_events()
        assert len(events) == 1
        llm = events[0]
        assert isinstance(llm, LLMCall)
        assert llm.model == "gpt-4o"
        assert llm.input_tokens == 10
        assert llm.output_tokens == 5
        assert llm.duration_ms is not None
        assert llm.response_summary == "Hi there"

    def test_system_message_extracted(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        self._run_generation(
            adapter,
            input_msgs=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "What is 2+2?"},
            ],
            output_msgs=[{"role": "assistant", "content": "4"}],
        )

        llm = adapter.get_events()[0]
        assert isinstance(llm, LLMCall)
        assert llm.system_message == "You are helpful"

    def test_prompt_summary_from_first_non_system(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        self._run_generation(
            adapter,
            input_msgs=[
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "User question here"},
            ],
        )

        llm = adapter.get_events()[0]
        assert isinstance(llm, LLMCall)
        assert llm.prompt_summary == "User question here"

    def test_no_system_message_when_absent(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        self._run_generation(
            adapter,
            input_msgs=[{"role": "user", "content": "Just user"}],
        )

        llm = adapter.get_events()[0]
        assert isinstance(llm, LLMCall)
        assert llm.system_message is None

    def test_none_usage(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        self._run_generation(adapter, usage=None)

        llm = adapter.get_events()[0]
        assert isinstance(llm, LLMCall)
        assert llm.input_tokens is None
        assert llm.output_tokens is None


class TestFunctionSpanTracking:
    """FunctionSpan -> ToolCall."""

    def test_basic_function_span(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        proc = adapter.get_callback_handler()

        span_data = FunctionSpanData(name="search", input='{"query": "test"}', output="Found 3")
        span = _make_span(span_data)
        proc.on_span_start(span)
        proc.on_span_end(span)

        events = adapter.get_events()
        assert len(events) == 1
        tc = events[0]
        assert isinstance(tc, ToolCall)
        assert tc.tool_name == "search"
        assert tc.tool_input == {"raw": '{"query": "test"}'}
        assert tc.tool_output == "Found 3"
        assert tc.duration_ms is not None

    def test_output_truncated(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        proc = adapter.get_callback_handler()

        long_output = "x" * 1000
        span_data = FunctionSpanData(name="tool", input="inp", output=long_output)
        span = _make_span(span_data)
        proc.on_span_start(span)
        proc.on_span_end(span)

        tc = adapter.get_events()[0]
        assert isinstance(tc, ToolCall)
        assert len(tc.tool_output) == 500  # type: ignore[arg-type]


class TestHandoffTracking:
    """HandoffSpan -> RoutingDecision + AgentMessage."""

    def test_handoff_emits_routing_and_message(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        proc = adapter.get_callback_handler()

        span = _make_span(HandoffSpanData(from_agent="agent_a", to_agent="agent_b"))
        proc.on_span_start(span)
        proc.on_span_end(span)

        events = adapter.get_events()
        routing = [e for e in events if isinstance(e, RoutingDecision)]
        messages = [e for e in events if isinstance(e, AgentMessage)]

        assert len(routing) == 1
        assert routing[0].source_node == "agent_a"
        assert routing[0].target_node == "agent_b"
        assert routing[0].metadata["inferred"] is False
        assert routing[0].metadata["mechanism"] == "handoff"

        assert len(messages) == 1
        assert messages[0].sender == "agent_a"
        assert messages[0].receiver == "agent_b"

    def test_handoff_emitted_without_infer_routing(self) -> None:
        """Handoffs are explicit — emitted even when infer_routing=False."""
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID, infer_routing=False)
        proc = adapter.get_callback_handler()

        span = _make_span(HandoffSpanData(from_agent="a", to_agent="b"))
        proc.on_span_start(span)
        proc.on_span_end(span)

        routing = [e for e in adapter.get_events() if isinstance(e, RoutingDecision)]
        assert len(routing) == 1


class TestGuardrailTracking:
    """GuardrailSpan -> ToolCall with guardrail metadata."""

    def test_guardrail_emits_tool_call(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        proc = adapter.get_callback_handler()

        span = _make_span(GuardrailSpanData(name="content_filter", triggered=True))
        proc.on_span_start(span)
        proc.on_span_end(span)

        events = adapter.get_events()
        assert len(events) == 1
        tc = events[0]
        assert isinstance(tc, ToolCall)
        assert tc.tool_name == "guardrail:content_filter"
        assert tc.tool_output == "triggered=True"
        assert tc.metadata["guardrail"] is True
        assert tc.metadata["triggered"] is True

    def test_guardrail_not_triggered(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        proc = adapter.get_callback_handler()

        span = _make_span(GuardrailSpanData(name="safe_check", triggered=False))
        proc.on_span_start(span)
        proc.on_span_end(span)

        tc = adapter.get_events()[0]
        assert isinstance(tc, ToolCall)
        assert tc.tool_output == "triggered=False"
        assert tc.metadata["triggered"] is False


class TestErrorTracking:
    """Span errors -> ErrorEvent."""

    def test_span_error_emits_error_event(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        proc = adapter.get_callback_handler()

        span = _make_span(AgentSpanData(name="agent"))
        proc.on_span_start(span)

        # Manually set error on the span
        span.set_error({"message": "Something went wrong", "data": {"code": 500}})
        proc.on_span_end(span)

        events = adapter.get_events()
        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(error_events) == 1
        assert error_events[0].error_type == "SpanError"
        assert error_events[0].error_message == "Something went wrong"

    def test_error_emitted_before_exit(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        proc = adapter.get_callback_handler()

        span = _make_span(AgentSpanData(name="agent"))
        proc.on_span_start(span)
        span.set_error({"message": "fail", "data": None})
        proc.on_span_end(span)

        events = adapter.get_events()
        # NodeEntry, ErrorEvent, NodeExit
        assert isinstance(events[0], NodeEntry)
        assert isinstance(events[1], ErrorEvent)
        assert isinstance(events[2], NodeExit)


class TestRoutingInference:
    """Inferred RoutingDecisions from agent transitions."""

    def _run_two_agents(self, adapter: OpenAIAgentsAdapter) -> None:
        proc = adapter.get_callback_handler()
        span_a = _make_span(AgentSpanData(name="agent_a"))
        proc.on_span_start(span_a)
        proc.on_span_end(span_a)

        span_b = _make_span(AgentSpanData(name="agent_b"))
        proc.on_span_start(span_b)
        proc.on_span_end(span_b)

    def test_no_routing_by_default(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        self._run_two_agents(adapter)
        routing = [e for e in adapter.get_events() if isinstance(e, RoutingDecision)]
        assert len(routing) == 0

    def test_infer_routing_emits_decision(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID, infer_routing=True)
        self._run_two_agents(adapter)
        routing = [e for e in adapter.get_events() if isinstance(e, RoutingDecision)]
        assert len(routing) == 1
        assert routing[0].source_node == "agent_a"
        assert routing[0].target_node == "agent_b"
        assert routing[0].metadata["inferred"] is True

    def test_no_routing_on_first_agent(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID, infer_routing=True)
        proc = adapter.get_callback_handler()
        span = _make_span(AgentSpanData(name="first"))
        proc.on_span_start(span)
        proc.on_span_end(span)
        routing = [e for e in adapter.get_events() if isinstance(e, RoutingDecision)]
        assert len(routing) == 0

    def test_three_agent_sequence(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID, infer_routing=True)
        proc = adapter.get_callback_handler()
        for name in ["a", "b", "c"]:
            span = _make_span(AgentSpanData(name=name))
            proc.on_span_start(span)
            proc.on_span_end(span)
        routing = [e for e in adapter.get_events() if isinstance(e, RoutingDecision)]
        assert len(routing) == 2
        assert routing[0].source_node == "a"
        assert routing[0].target_node == "b"
        assert routing[1].source_node == "b"
        assert routing[1].target_node == "c"

    def test_routing_appears_before_node_entry(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID, infer_routing=True)
        self._run_two_agents(adapter)
        events = adapter.get_events()
        routing_idx = next(
            i for i, e in enumerate(events) if isinstance(e, RoutingDecision)
        )
        second_entry_idx = next(
            i for i, e in enumerate(events)
            if isinstance(e, NodeEntry) and e.node_name == "agent_b"
        )
        assert routing_idx < second_entry_idx

    def test_handoff_dedup_no_duplicate(self) -> None:
        """When a handoff A->B already emitted RoutingDecision, the inferred
        RoutingDecision for the A->B agent transition should be suppressed."""
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID, infer_routing=True)
        proc = adapter.get_callback_handler()

        # Agent A starts and runs
        span_a = _make_span(AgentSpanData(name="agent_a"))
        proc.on_span_start(span_a)

        # Handoff from A to B (child of A)
        handoff = _make_span(
            HandoffSpanData(from_agent="agent_a", to_agent="agent_b"),
            parent_id=span_a.span_id,
        )
        proc.on_span_start(handoff)
        proc.on_span_end(handoff)

        # Agent A ends
        proc.on_span_end(span_a)

        # Agent B starts — should NOT emit a duplicate inferred routing
        span_b = _make_span(AgentSpanData(name="agent_b"))
        proc.on_span_start(span_b)
        proc.on_span_end(span_b)

        routing = [e for e in adapter.get_events() if isinstance(e, RoutingDecision)]
        # Only one RoutingDecision: the explicit handoff
        assert len(routing) == 1
        assert routing[0].metadata["inferred"] is False
        assert routing[0].metadata["mechanism"] == "handoff"

    def test_handoff_dedup_different_target_still_infers(self) -> None:
        """Handoff A->B should not suppress an inferred A->C transition."""
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID, infer_routing=True)
        proc = adapter.get_callback_handler()

        span_a = _make_span(AgentSpanData(name="agent_a"))
        proc.on_span_start(span_a)

        handoff = _make_span(
            HandoffSpanData(from_agent="agent_a", to_agent="agent_b"),
            parent_id=span_a.span_id,
        )
        proc.on_span_start(handoff)
        proc.on_span_end(handoff)

        proc.on_span_end(span_a)

        # Agent C starts (NOT agent_b) — should still get an inferred routing
        span_c = _make_span(AgentSpanData(name="agent_c"))
        proc.on_span_start(span_c)
        proc.on_span_end(span_c)

        routing = [e for e in adapter.get_events() if isinstance(e, RoutingDecision)]
        assert len(routing) == 2
        # Explicit handoff A->B
        assert routing[0].source_node == "agent_a"
        assert routing[0].target_node == "agent_b"
        assert routing[0].metadata["inferred"] is False
        # Inferred A->C
        assert routing[1].source_node == "agent_a"
        assert routing[1].target_node == "agent_c"
        assert routing[1].metadata["inferred"] is True


class TestStateCapture:
    """capture_state flag captures agent metadata."""

    def test_state_capture_off_by_default(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        proc = adapter.get_callback_handler()
        span = _make_span(AgentSpanData(name="agent"))
        proc.on_span_start(span)
        proc.on_span_end(span)

        events = adapter.get_events()
        entry = [e for e in events if isinstance(e, NodeEntry)][0]
        exit_ = [e for e in events if isinstance(e, NodeExit)][0]
        assert entry.input_state == {}
        assert exit_.output_state == {}
        assert exit_.state_diff == {}

    def test_state_capture_on(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID, capture_state=True)
        proc = adapter.get_callback_handler()
        span = _make_span(AgentSpanData(name="agent"))
        proc.on_span_start(span)
        proc.on_span_end(span)

        events = adapter.get_events()
        entry = [e for e in events if isinstance(e, NodeEntry)][0]
        exit_ = [e for e in events if isinstance(e, NodeExit)][0]
        assert entry.input_state["agent_name"] == "agent"
        assert "tools" in entry.input_state
        assert exit_.output_state["agent_name"] == "agent"

    def test_capture_state_threaded_through_tracer(self) -> None:
        from orcheval import Tracer

        tracer = Tracer(adapter="openai_agents", capture_state=True)
        assert tracer.adapter._capture_state is True  # type: ignore[union-attr]


class TestParentChildHierarchy:
    """Child spans get correct parent_span_id and node_name."""

    def test_llm_within_agent_gets_parent_span(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        proc = adapter.get_callback_handler()

        agent_span = _make_span(AgentSpanData(name="planner"))
        proc.on_span_start(agent_span)

        gen_span = _make_span(
            GenerationSpanData(model="gpt-4o"),
            parent_id=agent_span.span_id,
        )
        proc.on_span_start(gen_span)
        proc.on_span_end(gen_span)

        proc.on_span_end(agent_span)

        events = adapter.get_events()
        llm_events = [e for e in events if isinstance(e, LLMCall)]
        node_entries = [e for e in events if isinstance(e, NodeEntry)]

        assert len(llm_events) == 1
        assert llm_events[0].parent_span_id == node_entries[0].span_id
        assert llm_events[0].node_name == "planner"

    def test_tool_within_agent_gets_parent_span(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        proc = adapter.get_callback_handler()

        agent_span = _make_span(AgentSpanData(name="executor"))
        proc.on_span_start(agent_span)

        tool_span = _make_span(
            FunctionSpanData(name="search", input="query", output="results"),
            parent_id=agent_span.span_id,
        )
        proc.on_span_start(tool_span)
        proc.on_span_end(tool_span)

        proc.on_span_end(agent_span)

        events = adapter.get_events()
        tool_events = [e for e in events if isinstance(e, ToolCall)]
        node_entries = [e for e in events if isinstance(e, NodeEntry)]

        assert len(tool_events) == 1
        assert tool_events[0].parent_span_id == node_entries[0].span_id
        assert tool_events[0].node_name == "executor"

    def test_nested_agent_spans(self) -> None:
        """Child agent gets parent agent's OrchEval span_id as parent_span_id."""
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        proc = adapter.get_callback_handler()

        parent_span = _make_span(AgentSpanData(name="orchestrator"))
        proc.on_span_start(parent_span)

        child_span = _make_span(
            AgentSpanData(name="worker"),
            parent_id=parent_span.span_id,
        )
        proc.on_span_start(child_span)
        proc.on_span_end(child_span)

        proc.on_span_end(parent_span)

        events = adapter.get_events()
        entries = [e for e in events if isinstance(e, NodeEntry)]
        assert len(entries) == 2
        # Worker's parent_span_id should be orchestrator's span_id
        assert entries[1].parent_span_id == entries[0].span_id


class TestConcurrentAgentSpans:
    """Two agent spans active simultaneously (agents-as-tools)."""

    def test_concurrent_agents_correct_parent(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID)
        proc = adapter.get_callback_handler()

        # Parent orchestrator
        orch = _make_span(AgentSpanData(name="orchestrator"))
        proc.on_span_start(orch)

        # Two child agents start concurrently
        worker_a = _make_span(
            AgentSpanData(name="worker_a"), parent_id=orch.span_id
        )
        worker_b = _make_span(
            AgentSpanData(name="worker_b"), parent_id=orch.span_id
        )
        proc.on_span_start(worker_a)
        proc.on_span_start(worker_b)

        # LLM call inside worker_a
        gen_a = _make_span(
            GenerationSpanData(model="gpt-4o"),
            parent_id=worker_a.span_id,
        )
        proc.on_span_start(gen_a)
        proc.on_span_end(gen_a)

        # LLM call inside worker_b
        gen_b = _make_span(
            GenerationSpanData(model="gpt-4o-mini"),
            parent_id=worker_b.span_id,
        )
        proc.on_span_start(gen_b)
        proc.on_span_end(gen_b)

        # End both workers, then orchestrator
        proc.on_span_end(worker_a)
        proc.on_span_end(worker_b)
        proc.on_span_end(orch)

        events = adapter.get_events()
        llm_calls = [e for e in events if isinstance(e, LLMCall)]
        entries = {e.node_name: e for e in events if isinstance(e, NodeEntry)}

        assert len(llm_calls) == 2
        # Each LLM call should be associated with its own worker
        llm_a = next(e for e in llm_calls if e.model == "gpt-4o")
        llm_b = next(e for e in llm_calls if e.model == "gpt-4o-mini")

        assert llm_a.node_name == "worker_a"
        assert llm_a.parent_span_id == entries["worker_a"].span_id
        assert llm_b.node_name == "worker_b"
        assert llm_b.parent_span_id == entries["worker_b"].span_id


class TestFullSequence:
    """Multi-agent flow: Agent A (LLM + tool) -> handoff -> Agent B (LLM)."""

    def test_full_sequence(self) -> None:
        adapter = OpenAIAgentsAdapter(trace_id=TRACE_ID, infer_routing=True)
        proc = adapter.get_callback_handler()

        # --- Agent A ---
        span_a = _make_span(AgentSpanData(name="agent_a"))
        proc.on_span_start(span_a)

        # LLM call in A
        gen = _make_span(
            GenerationSpanData(
                model="gpt-4o",
                input=[{"role": "user", "content": "Analyze"}],
                output=[{"role": "assistant", "content": "Done"}],
                usage={"input_tokens": 100, "output_tokens": 50},
            ),
            parent_id=span_a.span_id,
        )
        proc.on_span_start(gen)
        proc.on_span_end(gen)

        # Tool call in A
        tool = _make_span(
            FunctionSpanData(name="search", input="query", output="results"),
            parent_id=span_a.span_id,
        )
        proc.on_span_start(tool)
        proc.on_span_end(tool)

        # Handoff A -> B
        handoff = _make_span(
            HandoffSpanData(from_agent="agent_a", to_agent="agent_b"),
            parent_id=span_a.span_id,
        )
        proc.on_span_start(handoff)
        proc.on_span_end(handoff)

        proc.on_span_end(span_a)

        # --- Agent B ---
        span_b = _make_span(AgentSpanData(name="agent_b"))
        proc.on_span_start(span_b)

        gen_b = _make_span(
            GenerationSpanData(
                model="gpt-4o-mini",
                input=[{"role": "user", "content": "Summarize"}],
                output=[{"role": "assistant", "content": "Summary"}],
                usage={"input_tokens": 50, "output_tokens": 30},
            ),
            parent_id=span_b.span_id,
        )
        proc.on_span_start(gen_b)
        proc.on_span_end(gen_b)

        proc.on_span_end(span_b)

        # --- Verify ---
        events = adapter.get_events()

        node_entries = [e for e in events if isinstance(e, NodeEntry)]
        node_exits = [e for e in events if isinstance(e, NodeExit)]
        llm_calls = [e for e in events if isinstance(e, LLMCall)]
        tool_calls = [
            e for e in events
            if isinstance(e, ToolCall) and "guardrail" not in e.tool_name
        ]
        routing = [e for e in events if isinstance(e, RoutingDecision)]
        messages = [e for e in events if isinstance(e, AgentMessage)]

        assert len(node_entries) == 2
        assert len(node_exits) == 2
        assert len(llm_calls) == 2
        assert len(tool_calls) == 1
        assert len(routing) == 1  # Only the handoff, no inferred duplicate
        assert len(messages) == 1

        assert node_entries[0].node_name == "agent_a"
        assert node_entries[1].node_name == "agent_b"
        assert llm_calls[0].node_name == "agent_a"
        assert llm_calls[0].model == "gpt-4o"
        assert llm_calls[1].node_name == "agent_b"
        assert llm_calls[1].model == "gpt-4o-mini"
        assert tool_calls[0].tool_name == "search"
        assert routing[0].source_node == "agent_a"
        assert routing[0].target_node == "agent_b"
