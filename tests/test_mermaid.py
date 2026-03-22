"""Tests for the Mermaid diagram export."""

from __future__ import annotations

from datetime import timedelta

from orcheval.events import ErrorEvent, LLMCall, NodeEntry, NodeExit, RoutingDecision
from orcheval.trace import Trace

from .conftest import BASE_TIME, TRACE_ID

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(seconds: float):
    return BASE_TIME + timedelta(seconds=seconds)


def _linear_trace(nodes: list[str], trace_id: str = TRACE_ID) -> Trace:
    """Build a trace with a simple linear node sequence (no loops)."""
    events = []
    for i, name in enumerate(nodes):
        span = f"span-{i}"
        events.append(NodeEntry(
            trace_id=trace_id, span_id=span, timestamp=_ts(i * 2), node_name=name,
        ))
        events.append(NodeExit(
            trace_id=trace_id, span_id=span, timestamp=_ts(i * 2 + 1),
            node_name=name, duration_ms=1000.0,
        ))
    return Trace(events=events, trace_id=trace_id)


def _loop_trace(sequence: list[str], trace_id: str = TRACE_ID) -> Trace:
    """Build a trace where nodes are entered in the given sequence order."""
    events = []
    for i, name in enumerate(sequence):
        span = f"span-{i}"
        events.append(NodeEntry(
            trace_id=trace_id, span_id=span, timestamp=_ts(i * 2), node_name=name,
        ))
        events.append(NodeExit(
            trace_id=trace_id, span_id=span, timestamp=_ts(i * 2 + 1),
            node_name=name, duration_ms=1000.0,
        ))
    return Trace(events=events, trace_id=trace_id)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


class TestMermaidStructure:
    def test_starts_with_graph(self) -> None:
        trace = _linear_trace(["agent", "summarizer"])
        result = trace.to_mermaid()
        assert result.startswith("graph LR")

    def test_ends_with_newline(self) -> None:
        trace = _linear_trace(["agent"])
        result = trace.to_mermaid()
        assert result.endswith("\n")

    def test_empty_trace(self) -> None:
        trace = Trace(events=[], trace_id=TRACE_ID)
        result = trace.to_mermaid()
        assert result == "graph LR\n"

    def test_single_node(self) -> None:
        trace = _linear_trace(["agent"])
        result = trace.to_mermaid()
        assert "agent" in result
        # Single node = no edges
        assert "-->" not in result


# ---------------------------------------------------------------------------
# Edge inference from node_sequence
# ---------------------------------------------------------------------------


class TestMermaidEdgeInference:
    def test_linear_edges_inferred(self) -> None:
        trace = _linear_trace(["A", "B", "C"])
        result = trace.to_mermaid()
        assert '-->|"1x"|' in result
        assert "A" in result
        assert "B" in result
        assert "C" in result
        # Should have exactly 2 edges: A->B, B->C
        assert result.count("-->") == 2

    def test_loop_edge_counts(self) -> None:
        # [A, B, A, B] -> consecutive pairs: (A,B), (B,A), (A,B)
        # Edges: A->B: 2x, B->A: 1x
        trace = _loop_trace(["A", "B", "A", "B"])
        result = trace.to_mermaid()
        assert 'A -->|"2x"| B' in result
        assert 'B -->|"1x"| A' in result

    def test_reconvergent_path_counts(self) -> None:
        # [A, B, C, B, C] -> consecutive pairs: (A,B), (B,C), (C,B), (B,C)
        # Edges: A->B: 1x, B->C: 2x, C->B: 1x
        trace = _loop_trace(["A", "B", "C", "B", "C"])
        result = trace.to_mermaid()
        assert 'A -->|"1x"| B' in result
        assert 'B -->|"2x"| C' in result
        assert 'C -->|"1x"| B' in result


# ---------------------------------------------------------------------------
# Routing decisions
# ---------------------------------------------------------------------------


class TestMermaidRoutingDecisions:
    def test_routing_decisions_used(self) -> None:
        """When RoutingDecision events exist, use them instead of node_sequence."""
        events = [
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="router"),
            RoutingDecision(
                trace_id=TRACE_ID, timestamp=_ts(1), node_name="router",
                source_node="router", target_node="handler_a",
            ),
            RoutingDecision(
                trace_id=TRACE_ID, timestamp=_ts(2), node_name="router",
                source_node="router", target_node="handler_a",
            ),
            RoutingDecision(
                trace_id=TRACE_ID, timestamp=_ts(3), node_name="router",
                source_node="router", target_node="handler_b",
            ),
            NodeExit(
                trace_id=TRACE_ID, span_id="s1", timestamp=_ts(4),
                node_name="router", duration_ms=4000.0,
            ),
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = trace.to_mermaid()
        # Routing decisions: router->handler_a: 2x, router->handler_b: 1x
        assert 'router -->|"2x"| handler_a' in result
        assert 'router -->|"1x"| handler_b' in result


# ---------------------------------------------------------------------------
# Node metadata
# ---------------------------------------------------------------------------


class TestMermaidNodeMetadata:
    def test_node_invocation_counts(self) -> None:
        trace = _loop_trace(["agent", "validator", "agent", "validator"])
        result = trace.to_mermaid()
        assert 'agent (2x)' in result
        assert 'validator (2x)' in result

    def test_error_node_styling(self) -> None:
        events = [
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="codegen"),
            ErrorEvent(
                trace_id=TRACE_ID, span_id="err1", parent_span_id="s1",
                timestamp=_ts(1), node_name="codegen",
                error_type="SyntaxError", error_message="bad code",
            ),
            NodeExit(
                trace_id=TRACE_ID, span_id="s1", timestamp=_ts(2),
                node_name="codegen", duration_ms=2000.0,
            ),
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = trace.to_mermaid()
        assert ":::error" in result
        assert "codegen" in result

    def test_classDef_present_when_errors(self) -> None:
        events = [
            NodeEntry(trace_id=TRACE_ID, span_id="s1", timestamp=_ts(0), node_name="x"),
            ErrorEvent(
                trace_id=TRACE_ID, span_id="err1", parent_span_id="s1",
                timestamp=_ts(1), node_name="x",
                error_type="RuntimeError", error_message="fail",
            ),
            NodeExit(
                trace_id=TRACE_ID, span_id="s1", timestamp=_ts(2),
                node_name="x", duration_ms=1000.0,
            ),
        ]
        trace = Trace(events=events, trace_id=TRACE_ID)
        result = trace.to_mermaid()
        assert "classDef error" in result

    def test_no_classDef_without_errors(self) -> None:
        trace = _linear_trace(["A", "B"])
        result = trace.to_mermaid()
        assert "classDef" not in result

    def test_deterministic_output(self) -> None:
        trace = _loop_trace(["C", "A", "B", "A"])
        r1 = trace.to_mermaid()
        r2 = trace.to_mermaid()
        assert r1 == r2


# ---------------------------------------------------------------------------
# Integration with sample_trace fixture
# ---------------------------------------------------------------------------


class TestMermaidWithSampleTrace:
    def test_sample_trace(self, sample_trace: Trace) -> None:
        result = sample_trace.to_mermaid()
        assert result.startswith("graph LR")
        assert "agent" in result
        assert "summarizer" in result
        # Linear: agent -> summarizer, so one edge
        assert result.count("-->") == 1
