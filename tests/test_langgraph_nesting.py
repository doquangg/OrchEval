"""Tests for nested chain deduplication in the LangGraph adapter.

LangGraph fires on_chain_start/on_chain_end at multiple nesting levels for a
single logical node execution. All levels carry
metadata={"langgraph_node": "<node_name>"}, which previously caused duplicate
NodeEntry/NodeExit events, ghost spans, and duplicate routing decisions.

These tests verify that the adapter correctly suppresses nested duplicates
while preserving correct parent-child relationships for LLM/tool calls.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

langchain_core = pytest.importorskip("langchain_core")

from langchain_core.outputs import Generation, LLMResult  # noqa: E402

from orcheval.adapters.langgraph import LangGraphAdapter  # noqa: E402
from orcheval.events import (  # noqa: E402
    LLMCall,
    NodeEntry,
    NodeExit,
    RoutingDecision,
    ToolCall,
)

TRACE_ID = "test-nesting"

pytestmark = pytest.mark.integration


def _make_run_id() -> str:
    return str(uuid.uuid4())


def _make_llm_result(
    text: str = "Hello",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> LLMResult:
    generation = Generation(text=text)
    return LLMResult(
        generations=[[generation]],
        llm_output={"token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }},
    )


class TestNestedChainDeduplication:
    """Verify that nested LangChain chains produce a single NodeEntry/NodeExit."""

    def test_nested_chains_produce_single_invocation(self) -> None:
        """Three nesting levels for one node should emit exactly 1 entry + 1 exit."""
        adapter = LangGraphAdapter(trace_id=TRACE_ID, infer_routing=True)
        handler = adapter.get_callback_handler()

        # Depth 0: graph-level dispatch
        run_d0 = _make_run_id()
        handler.on_chain_start(
            {}, {}, run_id=run_d0, metadata={"langgraph_node": "my_node"}
        )
        # Depth 1: inner chain wrapper
        run_d1 = _make_run_id()
        handler.on_chain_start(
            {}, {}, run_id=run_d1, metadata={"langgraph_node": "my_node"}
        )
        # Depth 2: with_structured_output wrapper
        run_d2 = _make_run_id()
        handler.on_chain_start(
            {}, {}, run_id=run_d2, metadata={"langgraph_node": "my_node"}
        )

        # LLM call inside the innermost chain
        llm_run = _make_run_id()
        handler.on_llm_start(
            {"kwargs": {"model_name": "gpt-4o"}}, ["test prompt"], run_id=llm_run
        )
        handler.on_llm_end(_make_llm_result(), run_id=llm_run)

        # Unwind in LIFO order
        handler.on_chain_end({}, run_id=run_d2)
        handler.on_chain_end({}, run_id=run_d1)
        handler.on_chain_end({}, run_id=run_d0)

        events = adapter.get_events()
        node_entries = [e for e in events if isinstance(e, NodeEntry)]
        node_exits = [e for e in events if isinstance(e, NodeExit)]
        llm_calls = [e for e in events if isinstance(e, LLMCall)]
        routing = [e for e in events if isinstance(e, RoutingDecision)]

        assert len(node_entries) == 1
        assert len(node_exits) == 1
        assert len(llm_calls) == 1
        assert node_entries[0].node_name == "my_node"
        assert node_exits[0].node_name == "my_node"
        # No routing for the first (and only) node
        assert len(routing) == 0

    def test_llm_parented_to_outermost_span(self) -> None:
        """LLM call inside nested chains should be parented to the outermost span."""
        adapter = LangGraphAdapter(trace_id=TRACE_ID)
        handler = adapter.get_callback_handler()

        run_d0 = _make_run_id()
        handler.on_chain_start(
            {}, {}, run_id=run_d0, metadata={"langgraph_node": "my_node"}
        )
        run_d1 = _make_run_id()
        handler.on_chain_start(
            {}, {}, run_id=run_d1, metadata={"langgraph_node": "my_node"}
        )

        llm_run = _make_run_id()
        handler.on_llm_start({}, ["test"], run_id=llm_run)
        handler.on_llm_end(_make_llm_result(), run_id=llm_run)

        handler.on_chain_end({}, run_id=run_d1)
        handler.on_chain_end({}, run_id=run_d0)

        events = adapter.get_events()
        entry = [e for e in events if isinstance(e, NodeEntry)][0]
        llm = [e for e in events if isinstance(e, LLMCall)][0]

        assert llm.parent_span_id == entry.span_id
        assert llm.node_name == "my_node"

    def test_tool_call_parented_to_outermost_span(self) -> None:
        """Tool call inside nested chains should be parented to the outermost span."""
        adapter = LangGraphAdapter(trace_id=TRACE_ID)
        handler = adapter.get_callback_handler()

        run_d0 = _make_run_id()
        handler.on_chain_start(
            {}, {}, run_id=run_d0, metadata={"langgraph_node": "my_node"}
        )
        run_d1 = _make_run_id()
        handler.on_chain_start(
            {}, {}, run_id=run_d1, metadata={"langgraph_node": "my_node"}
        )

        tool_run = _make_run_id()
        handler.on_tool_start({"name": "search"}, '{"q": "test"}', run_id=tool_run)
        handler.on_tool_end("results", run_id=tool_run)

        handler.on_chain_end({}, run_id=run_d1)
        handler.on_chain_end({}, run_id=run_d0)

        events = adapter.get_events()
        entry = [e for e in events if isinstance(e, NodeEntry)][0]
        tool = [e for e in events if isinstance(e, ToolCall)][0]

        assert tool.parent_span_id == entry.span_id
        assert tool.node_name == "my_node"

    def test_entry_exit_share_span_id(self) -> None:
        """NodeEntry and NodeExit should share the outermost span_id."""
        adapter = LangGraphAdapter(trace_id=TRACE_ID)
        handler = adapter.get_callback_handler()

        run_d0 = _make_run_id()
        handler.on_chain_start(
            {}, {}, run_id=run_d0, metadata={"langgraph_node": "my_node"}
        )
        run_d1 = _make_run_id()
        handler.on_chain_start(
            {}, {}, run_id=run_d1, metadata={"langgraph_node": "my_node"}
        )
        handler.on_chain_end({}, run_id=run_d1)
        handler.on_chain_end({}, run_id=run_d0)

        events = adapter.get_events()
        entry = [e for e in events if isinstance(e, NodeEntry)][0]
        exit_ = [e for e in events if isinstance(e, NodeExit)][0]
        assert entry.span_id == exit_.span_id


class TestToolLoopReentry:
    """Verify that tool-loop re-dispatches produce separate invocations."""

    def test_reentry_after_full_exit(self) -> None:
        """After a node fully exits, a new chain_start should create a new invocation."""
        adapter = LangGraphAdapter(trace_id=TRACE_ID)
        handler = adapter.get_callback_handler()

        # First invocation (with nesting)
        run_d0_a = _make_run_id()
        handler.on_chain_start(
            {}, {}, run_id=run_d0_a, metadata={"langgraph_node": "investigator"}
        )
        run_d1_a = _make_run_id()
        handler.on_chain_start(
            {}, {}, run_id=run_d1_a, metadata={"langgraph_node": "investigator"}
        )
        handler.on_chain_end({}, run_id=run_d1_a)
        handler.on_chain_end({}, run_id=run_d0_a)

        # Second invocation (tool-loop re-dispatch, also with nesting)
        run_d0_b = _make_run_id()
        handler.on_chain_start(
            {}, {}, run_id=run_d0_b, metadata={"langgraph_node": "investigator"}
        )
        run_d1_b = _make_run_id()
        handler.on_chain_start(
            {}, {}, run_id=run_d1_b, metadata={"langgraph_node": "investigator"}
        )
        handler.on_chain_end({}, run_id=run_d1_b)
        handler.on_chain_end({}, run_id=run_d0_b)

        events = adapter.get_events()
        entries = [e for e in events if isinstance(e, NodeEntry)]
        exits = [e for e in events if isinstance(e, NodeExit)]

        assert len(entries) == 2
        assert len(exits) == 2
        # Each invocation should have its own span_id
        assert entries[0].span_id != entries[1].span_id
        assert entries[0].span_id == exits[0].span_id
        assert entries[1].span_id == exits[1].span_id


class TestNestedRoutingDedup:
    """Verify that nested chains don't produce duplicate routing decisions."""

    def test_single_routing_decision_with_nesting(self) -> None:
        """Node A exits, node B enters with 2 nesting levels -> 1 RoutingDecision."""
        adapter = LangGraphAdapter(trace_id=TRACE_ID, infer_routing=True)
        handler = adapter.get_callback_handler()

        # Node A: simple entry/exit
        run_a = _make_run_id()
        handler.on_chain_start(
            {}, {}, run_id=run_a, metadata={"langgraph_node": "node_a"}
        )
        handler.on_chain_end({"result": "done"}, run_id=run_a)

        # Node B: enters with nested chains
        run_b_d0 = _make_run_id()
        handler.on_chain_start(
            {}, {}, run_id=run_b_d0, metadata={"langgraph_node": "node_b"}
        )
        run_b_d1 = _make_run_id()
        handler.on_chain_start(
            {}, {}, run_id=run_b_d1, metadata={"langgraph_node": "node_b"}
        )
        run_b_d2 = _make_run_id()
        handler.on_chain_start(
            {}, {}, run_id=run_b_d2, metadata={"langgraph_node": "node_b"}
        )
        handler.on_chain_end({}, run_id=run_b_d2)
        handler.on_chain_end({}, run_id=run_b_d1)
        handler.on_chain_end({}, run_id=run_b_d0)

        events = adapter.get_events()
        routing = [e for e in events if isinstance(e, RoutingDecision)]

        assert len(routing) == 1
        assert routing[0].source_node == "node_a"
        assert routing[0].target_node == "node_b"

    def test_three_node_sequence_with_nesting(self) -> None:
        """A -> B -> C with nesting at each node should produce exactly 2 routing decisions."""
        adapter = LangGraphAdapter(trace_id=TRACE_ID, infer_routing=True)
        handler = adapter.get_callback_handler()

        for name in ["a", "b", "c"]:
            # Each node has 2 nesting levels
            run_d0 = _make_run_id()
            handler.on_chain_start(
                {}, {}, run_id=run_d0, metadata={"langgraph_node": name}
            )
            run_d1 = _make_run_id()
            handler.on_chain_start(
                {}, {}, run_id=run_d1, metadata={"langgraph_node": name}
            )
            handler.on_chain_end({}, run_id=run_d1)
            handler.on_chain_end({}, run_id=run_d0)

        events = adapter.get_events()
        routing = [e for e in events if isinstance(e, RoutingDecision)]
        assert len(routing) == 2
        assert routing[0].source_node == "a"
        assert routing[0].target_node == "b"
        assert routing[1].source_node == "b"
        assert routing[1].target_node == "c"


class TestNestedErrorHandling:
    """Verify error handling works correctly with nested chain suppression."""

    def test_error_in_nested_chain_still_emits_error_event(self) -> None:
        """An error in a suppressed nested chain should still emit an ErrorEvent."""
        from orcheval.events import ErrorEvent

        adapter = LangGraphAdapter(trace_id=TRACE_ID)
        handler = adapter.get_callback_handler()

        run_d0 = _make_run_id()
        handler.on_chain_start(
            {}, {}, run_id=run_d0, metadata={"langgraph_node": "my_node"}
        )
        run_d1 = _make_run_id()
        handler.on_chain_start(
            {}, {}, run_id=run_d1, metadata={"langgraph_node": "my_node"}
        )

        # Error on the inner (suppressed) chain
        handler.on_chain_error(ValueError("inner error"), run_id=run_d1)
        # Outer chain still ends normally
        handler.on_chain_end({}, run_id=run_d0)

        events = adapter.get_events()
        errors = [e for e in events if isinstance(e, ErrorEvent)]
        entries = [e for e in events if isinstance(e, NodeEntry)]
        exits = [e for e in events if isinstance(e, NodeExit)]

        assert len(errors) == 1
        assert errors[0].error_message == "inner error"
        assert len(entries) == 1
        assert len(exits) == 1
