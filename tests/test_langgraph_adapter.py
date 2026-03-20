"""Tests for orcheval.adapters.langgraph — LangGraph adapter.

These tests require langchain_core to be installed. They are skipped otherwise.
"""

from __future__ import annotations

import uuid

import pytest

langchain_core = pytest.importorskip("langchain_core")

from langchain_core.outputs import Generation, LLMResult  # noqa: E402

from orcheval.adapters.langgraph import LangGraphAdapter  # noqa: E402
from orcheval.events import (  # noqa: E402
    ErrorEvent,
    LLMCall,
    NodeEntry,
    NodeExit,
    RoutingDecision,
    ToolCall,
)

TRACE_ID = "test-langgraph"

pytestmark = pytest.mark.integration


def _make_run_id() -> str:
    return str(uuid.uuid4())


def _make_llm_result(
    text: str = "Hello",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> LLMResult:
    """Create a mock LLMResult."""
    generation = Generation(text=text)
    return LLMResult(
        generations=[[generation]],
        llm_output={"token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }},
    )


class TestLangGraphAdapterBasics:
    def test_creates_handler(self) -> None:
        adapter = LangGraphAdapter(trace_id=TRACE_ID)
        handler = adapter.get_callback_handler()
        assert handler is not None

    def test_trace_id(self) -> None:
        adapter = LangGraphAdapter(trace_id=TRACE_ID)
        assert adapter.trace_id == TRACE_ID

    def test_empty_events(self) -> None:
        adapter = LangGraphAdapter(trace_id=TRACE_ID)
        assert adapter.get_events() == []

    def test_reset(self) -> None:
        adapter = LangGraphAdapter(trace_id=TRACE_ID)
        handler = adapter.get_callback_handler()
        run_id = _make_run_id()
        handler.on_chain_start(
            {}, {}, run_id=run_id, metadata={"langgraph_node": "agent"}
        )
        assert len(adapter.get_events()) == 1
        adapter.reset()
        assert len(adapter.get_events()) == 0


class TestNodeTracking:
    """Layer 1: Direct callback invocation for node entry/exit."""

    def test_chain_start_with_node_emits_node_entry(self) -> None:
        adapter = LangGraphAdapter(trace_id=TRACE_ID)
        handler = adapter.get_callback_handler()
        run_id = _make_run_id()

        handler.on_chain_start(
            {}, {}, run_id=run_id, metadata={"langgraph_node": "agent"}
        )

        events = adapter.get_events()
        assert len(events) == 1
        assert isinstance(events[0], NodeEntry)
        assert events[0].node_name == "agent"
        assert events[0].trace_id == TRACE_ID

    def test_chain_start_without_node_no_node_entry(self) -> None:
        """Chains without langgraph_node metadata should NOT emit NodeEntry."""
        adapter = LangGraphAdapter(trace_id=TRACE_ID)
        handler = adapter.get_callback_handler()
        run_id = _make_run_id()

        handler.on_chain_start({}, {}, run_id=run_id, metadata={})

        events = adapter.get_events()
        node_entries = [e for e in events if isinstance(e, NodeEntry)]
        assert len(node_entries) == 0

    def test_chain_end_emits_node_exit(self) -> None:
        adapter = LangGraphAdapter(trace_id=TRACE_ID)
        handler = adapter.get_callback_handler()
        run_id = _make_run_id()

        handler.on_chain_start(
            {}, {}, run_id=run_id, metadata={"langgraph_node": "agent"}
        )
        handler.on_chain_end({}, run_id=run_id)

        events = adapter.get_events()
        assert len(events) == 2
        assert isinstance(events[1], NodeExit)
        assert events[1].node_name == "agent"
        assert events[1].duration_ms is not None
        assert events[1].duration_ms >= 0

    def test_node_entry_exit_share_span_id(self) -> None:
        adapter = LangGraphAdapter(trace_id=TRACE_ID)
        handler = adapter.get_callback_handler()
        run_id = _make_run_id()

        handler.on_chain_start(
            {}, {}, run_id=run_id, metadata={"langgraph_node": "agent"}
        )
        handler.on_chain_end({}, run_id=run_id)

        events = adapter.get_events()
        assert events[0].span_id == events[1].span_id


class TestLLMTracking:
    """Layer 1: Direct callback invocation for LLM calls."""

    def test_llm_start_end_emits_llm_call(self) -> None:
        adapter = LangGraphAdapter(trace_id=TRACE_ID)
        handler = adapter.get_callback_handler()
        llm_run_id = _make_run_id()

        handler.on_llm_start(
            {"kwargs": {"model_name": "gpt-4o"}},
            ["What is 2+2?"],
            run_id=llm_run_id,
        )
        handler.on_llm_end(
            _make_llm_result("4", prompt_tokens=15, completion_tokens=3),
            run_id=llm_run_id,
        )

        events = adapter.get_events()
        assert len(events) == 1
        assert isinstance(events[0], LLMCall)
        assert events[0].model == "gpt-4o"
        assert events[0].input_tokens == 15
        assert events[0].output_tokens == 3
        assert events[0].duration_ms is not None
        assert events[0].response_summary == "4"

    def test_llm_model_from_metadata(self) -> None:
        adapter = LangGraphAdapter(trace_id=TRACE_ID)
        handler = adapter.get_callback_handler()
        llm_run_id = _make_run_id()

        handler.on_llm_start(
            {},
            ["Hello"],
            run_id=llm_run_id,
            metadata={"ls_model_name": "claude-3.5-sonnet"},
        )
        handler.on_llm_end(_make_llm_result(), run_id=llm_run_id)

        events = adapter.get_events()
        assert events[0].model == "claude-3.5-sonnet"

    def test_llm_within_node_gets_parent_span(self) -> None:
        adapter = LangGraphAdapter(trace_id=TRACE_ID)
        handler = adapter.get_callback_handler()
        chain_run_id = _make_run_id()
        llm_run_id = _make_run_id()

        handler.on_chain_start(
            {}, {}, run_id=chain_run_id, metadata={"langgraph_node": "agent"}
        )
        handler.on_llm_start({}, ["Hi"], run_id=llm_run_id)
        handler.on_llm_end(_make_llm_result(), run_id=llm_run_id)
        handler.on_chain_end({}, run_id=chain_run_id)

        events = adapter.get_events()
        llm_event = [e for e in events if isinstance(e, LLMCall)][0]
        node_entry = [e for e in events if isinstance(e, NodeEntry)][0]
        assert llm_event.parent_span_id == node_entry.span_id
        assert llm_event.node_name == "agent"


class TestToolTracking:
    """Layer 1: Direct callback invocation for tool calls."""

    def test_tool_start_end_emits_tool_call(self) -> None:
        adapter = LangGraphAdapter(trace_id=TRACE_ID)
        handler = adapter.get_callback_handler()
        tool_run_id = _make_run_id()

        handler.on_tool_start(
            {"name": "search"},
            '{"query": "test"}',
            run_id=tool_run_id,
        )
        handler.on_tool_end("Found 3 results", run_id=tool_run_id)

        events = adapter.get_events()
        assert len(events) == 1
        assert isinstance(events[0], ToolCall)
        assert events[0].tool_name == "search"
        assert events[0].tool_output == "Found 3 results"
        assert events[0].duration_ms is not None


class TestErrorTracking:
    """Layer 1: Error callback handling."""

    def test_chain_error_emits_error_event(self) -> None:
        adapter = LangGraphAdapter(trace_id=TRACE_ID)
        handler = adapter.get_callback_handler()
        run_id = _make_run_id()

        handler.on_chain_start(
            {}, {}, run_id=run_id, metadata={"langgraph_node": "agent"}
        )
        handler.on_chain_error(ValueError("bad input"), run_id=run_id)

        events = adapter.get_events()
        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(error_events) == 1
        assert error_events[0].error_type == "ValueError"
        assert error_events[0].error_message == "bad input"

    def test_llm_error_emits_error_event(self) -> None:
        adapter = LangGraphAdapter(trace_id=TRACE_ID)
        handler = adapter.get_callback_handler()
        llm_run_id = _make_run_id()

        handler.on_llm_start({}, ["Hello"], run_id=llm_run_id)
        handler.on_llm_error(RuntimeError("API down"), run_id=llm_run_id)

        events = adapter.get_events()
        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(error_events) == 1
        assert error_events[0].error_type == "RuntimeError"

    def test_tool_error_emits_error_event(self) -> None:
        adapter = LangGraphAdapter(trace_id=TRACE_ID)
        handler = adapter.get_callback_handler()
        tool_run_id = _make_run_id()

        handler.on_tool_start({"name": "calc"}, "1+1", run_id=tool_run_id)
        handler.on_tool_error(TimeoutError("timed out"), run_id=tool_run_id)

        events = adapter.get_events()
        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(error_events) == 1
        assert error_events[0].error_type == "TimeoutError"


class TestFullNodeSequence:
    """Layer 1: Simulate a multi-node graph execution via direct callbacks."""

    def test_two_node_sequence(self) -> None:
        adapter = LangGraphAdapter(trace_id=TRACE_ID)
        handler = adapter.get_callback_handler()

        # Node 1: agent
        agent_run = _make_run_id()
        handler.on_chain_start(
            {}, {}, run_id=agent_run, metadata={"langgraph_node": "agent"}
        )
        llm_run = _make_run_id()
        handler.on_llm_start(
            {"kwargs": {"model_name": "gpt-4o"}}, ["Analyze"], run_id=llm_run
        )
        handler.on_llm_end(_make_llm_result("Analysis done"), run_id=llm_run)
        handler.on_chain_end({}, run_id=agent_run)

        # Node 2: summarizer
        summ_run = _make_run_id()
        handler.on_chain_start(
            {}, {}, run_id=summ_run, metadata={"langgraph_node": "summarizer"}
        )
        llm_run2 = _make_run_id()
        handler.on_llm_start({}, ["Summarize"], run_id=llm_run2)
        handler.on_llm_end(_make_llm_result("Summary"), run_id=llm_run2)
        handler.on_chain_end({}, run_id=summ_run)

        events = adapter.get_events()
        # Should have: NodeEntry, LLMCall, NodeExit, NodeEntry, LLMCall, NodeExit
        assert len(events) == 6

        node_entries = [e for e in events if isinstance(e, NodeEntry)]
        node_exits = [e for e in events if isinstance(e, NodeExit)]
        llm_calls = [e for e in events if isinstance(e, LLMCall)]

        assert len(node_entries) == 2
        assert len(node_exits) == 2
        assert len(llm_calls) == 2

        assert node_entries[0].node_name == "agent"
        assert node_entries[1].node_name == "summarizer"
        assert llm_calls[0].node_name == "agent"
        assert llm_calls[1].node_name == "summarizer"


class TestRoutingInference:
    """Test opt-in routing decision inference from node transitions."""

    def _run_two_nodes(
        self, adapter: LangGraphAdapter, node_a: str = "agent", node_b: str = "summarizer"
    ) -> None:
        handler = adapter.get_callback_handler()
        run_a = _make_run_id()
        handler.on_chain_start({}, {}, run_id=run_a, metadata={"langgraph_node": node_a})
        handler.on_chain_end({}, run_id=run_a)
        run_b = _make_run_id()
        handler.on_chain_start({}, {}, run_id=run_b, metadata={"langgraph_node": node_b})
        handler.on_chain_end({}, run_id=run_b)

    def test_no_routing_by_default(self) -> None:
        adapter = LangGraphAdapter(trace_id=TRACE_ID)
        self._run_two_nodes(adapter)
        routing = [e for e in adapter.get_events() if isinstance(e, RoutingDecision)]
        assert len(routing) == 0

    def test_infer_routing_emits_decision(self) -> None:
        adapter = LangGraphAdapter(trace_id=TRACE_ID, infer_routing=True)
        self._run_two_nodes(adapter)
        routing = [e for e in adapter.get_events() if isinstance(e, RoutingDecision)]
        assert len(routing) == 1
        assert routing[0].source_node == "agent"
        assert routing[0].target_node == "summarizer"

    def test_three_node_sequence(self) -> None:
        adapter = LangGraphAdapter(trace_id=TRACE_ID, infer_routing=True)
        handler = adapter.get_callback_handler()
        for name in ["a", "b", "c"]:
            run_id = _make_run_id()
            handler.on_chain_start({}, {}, run_id=run_id, metadata={"langgraph_node": name})
            handler.on_chain_end({}, run_id=run_id)
        routing = [e for e in adapter.get_events() if isinstance(e, RoutingDecision)]
        assert len(routing) == 2
        assert routing[0].source_node == "a"
        assert routing[0].target_node == "b"
        assert routing[1].source_node == "b"
        assert routing[1].target_node == "c"

    def test_routing_on_same_node_reentry(self) -> None:
        """Same-node re-entry should still emit a routing decision (self-loop)."""
        adapter = LangGraphAdapter(trace_id=TRACE_ID, infer_routing=True)
        self._run_two_nodes(adapter, node_a="agent", node_b="agent")
        routing = [e for e in adapter.get_events() if isinstance(e, RoutingDecision)]
        assert len(routing) == 1
        assert routing[0].source_node == "agent"
        assert routing[0].target_node == "agent"

    def test_no_routing_on_first_node(self) -> None:
        adapter = LangGraphAdapter(trace_id=TRACE_ID, infer_routing=True)
        handler = adapter.get_callback_handler()
        run_id = _make_run_id()
        handler.on_chain_start({}, {}, run_id=run_id, metadata={"langgraph_node": "first"})
        handler.on_chain_end({}, run_id=run_id)
        routing = [e for e in adapter.get_events() if isinstance(e, RoutingDecision)]
        assert len(routing) == 0

    def test_metadata_inferred_flag(self) -> None:
        adapter = LangGraphAdapter(trace_id=TRACE_ID, infer_routing=True)
        self._run_two_nodes(adapter)
        routing = [e for e in adapter.get_events() if isinstance(e, RoutingDecision)]
        assert routing[0].metadata["inferred"] is True

    def test_routing_appears_before_node_entry(self) -> None:
        """RoutingDecision should appear before the NodeEntry it caused."""
        adapter = LangGraphAdapter(trace_id=TRACE_ID, infer_routing=True)
        self._run_two_nodes(adapter)
        events = adapter.get_events()
        # Find the RoutingDecision and the second NodeEntry
        routing_idx = next(
            i for i, e in enumerate(events) if isinstance(e, RoutingDecision)
        )
        second_entry_idx = next(
            i for i, e in enumerate(events)
            if isinstance(e, NodeEntry) and e.node_name == "summarizer"
        )
        assert routing_idx < second_entry_idx
