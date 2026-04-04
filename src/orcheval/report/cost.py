"""Per-node, per-model cost and token breakdown."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from orcheval.trace import Trace

UNKNOWN_NODE = "<unknown>"
UNKNOWN_MODEL = "<unknown_model>"


class ModelUsage(BaseModel):
    """Token, cost, and call statistics for a single model."""

    model_config = {"frozen": True}

    model: str
    call_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_cost: float | None = None
    avg_cost_per_call: float | None = None
    avg_duration_ms: float | None = None


class NodeCostSummary(BaseModel):
    """Cost and token breakdown for a single node."""

    model_config = {"frozen": True}

    node_name: str
    models: list[ModelUsage] = Field(default_factory=list)
    total_cost: float | None = None
    total_tokens: int = 0
    call_count: int = 0


class CostReport(BaseModel):
    """Full cost and token analysis across all nodes and models."""

    model_config = {"frozen": True}

    nodes: list[NodeCostSummary] = Field(default_factory=list)
    models: list[ModelUsage] = Field(default_factory=list)
    total_cost: float | None = None
    total_tokens: dict[str, int] = Field(default_factory=dict)
    most_expensive_node: str | None = None
    most_expensive_model: str | None = None


def _build_model_usage(
    model_name: str,
    calls: list[tuple[int | None, int | None, float | None, float | None]],
) -> ModelUsage:
    """Build a ModelUsage from a list of (input_tokens, output_tokens, cost, duration_ms)."""
    call_count = len(calls)
    input_tok = sum(c[0] for c in calls if c[0] is not None)
    output_tok = sum(c[1] for c in calls if c[1] is not None)

    costs = [c[2] for c in calls if c[2] is not None]
    total_cost = sum(costs) if costs else None
    avg_cost = total_cost / len(costs) if total_cost is not None else None

    durations = [c[3] for c in calls if c[3] is not None]
    avg_duration = sum(durations) / len(durations) if durations else None

    return ModelUsage(
        model=model_name,
        call_count=call_count,
        input_tokens=input_tok,
        output_tokens=output_tok,
        total_tokens=input_tok + output_tok,
        total_cost=total_cost,
        avg_cost_per_call=avg_cost,
        avg_duration_ms=avg_duration,
    )


def cost_report(trace: Trace) -> CostReport:
    """Generate a per-node, per-model cost and token breakdown.

    Delegates total_cost and total_tokens to Trace summary methods.
    """
    llm_calls = trace.get_llm_calls()
    if not llm_calls:
        return CostReport(
            total_cost=trace.total_cost(),
            total_tokens=trace.total_tokens(),
        )

    # (input_tokens, output_tokens, cost, duration_ms) per call
    _CallData = tuple[int | None, int | None, float | None, float | None]

    # Group calls: node_name -> model -> list of call data
    node_model_calls: dict[str, dict[str, list[_CallData]]] = defaultdict(
        lambda: defaultdict(list),
    )
    global_model_calls: dict[str, list[_CallData]] = defaultdict(list)

    for call in llm_calls:
        node = call.node_name or UNKNOWN_NODE
        model = call.model or UNKNOWN_MODEL
        entry = (call.input_tokens, call.output_tokens, call.cost, call.duration_ms)
        node_model_calls[node][model].append(entry)
        global_model_calls[model].append(entry)

    # Build per-node summaries
    nodes: list[NodeCostSummary] = []
    for node_name, model_map in sorted(node_model_calls.items()):
        model_usages = [
            _build_model_usage(m, calls)
            for m, calls in sorted(model_map.items())
        ]
        node_costs = [mu.total_cost for mu in model_usages if mu.total_cost is not None]
        node_total_cost = sum(node_costs) if node_costs else None
        node_total_tokens = sum(mu.total_tokens for mu in model_usages)
        node_call_count = sum(mu.call_count for mu in model_usages)

        nodes.append(NodeCostSummary(
            node_name=node_name,
            models=model_usages,
            total_cost=node_total_cost,
            total_tokens=node_total_tokens,
            call_count=node_call_count,
        ))

    # Build global model summaries
    models = [
        _build_model_usage(m, calls)
        for m, calls in sorted(global_model_calls.items())
    ]

    # Find most expensive node and model
    nodes_with_cost = [(n.node_name, n.total_cost) for n in nodes if n.total_cost is not None]
    most_expensive_node = max(nodes_with_cost, key=lambda x: x[1])[0] if nodes_with_cost else None

    models_with_cost = [
        (m.model, m.total_cost) for m in models if m.total_cost is not None
    ]
    most_expensive_model = (
        max(models_with_cost, key=lambda x: x[1])[0] if models_with_cost else None
    )

    return CostReport(
        nodes=nodes,
        models=models,
        total_cost=trace.total_cost(),
        total_tokens=trace.total_tokens(),
        most_expensive_node=most_expensive_node,
        most_expensive_model=most_expensive_model,
    )
