# OrchEval

Evaluate, profile, and debug multi-agent LLM systems.

OrchEval captures structured trace events from multi-agent orchestrations and provides
query, analysis, and assertion tools for understanding system-level behavior — routing
correctness, inter-agent data flow, cost/latency per agent, and convergence across passes.

## Installation

```bash
pip install orcheval

# With LangGraph support:
pip install orcheval[langgraph]
```

## Quick Start

### With LangGraph

```python
from orcheval import Tracer

tracer = Tracer(adapter="langgraph")
result = graph.invoke(input, config={"callbacks": [tracer.handler]})
trace = tracer.collect()

# Inspect what happened
print(trace.node_sequence())        # ["agent", "summarizer"]
print(trace.total_cost())           # 0.007
print(trace.total_tokens())         # {"prompt": 350, "completion": 180, "total": 530}
print(trace.node_durations())       # {"agent": 3000.0, "summarizer": 2000.0}

for call in trace.get_llm_calls():
    print(f"{call.node_name}: {call.model} ({call.input_tokens}+{call.output_tokens} tokens)")
```

### Manual Tracing

For frameworks without a built-in adapter:

```python
from orcheval import Tracer

tracer = Tracer()  # defaults to manual adapter
tracer.adapter.node_entry("agent")
tracer.adapter.llm_call(model="gpt-4o", input_tokens=100, output_tokens=50, cost=0.003)
tracer.adapter.node_exit("agent", duration_ms=1500.0)

trace = tracer.collect()
```

## Reports

Generate structured analysis from a trace:

```python
from orcheval import Tracer, report

tracer = Tracer(adapter="langgraph")
result = graph.invoke(input, config={"callbacks": [tracer.handler]})
trace = tracer.collect()

full = report(trace)
print(full.cost.total_cost)           # Total $ across all LLM calls
print(full.routing.total_decisions)   # Number of routing decisions observed
print(full.timeline.total_duration_ms)
print(full.convergence.is_converging) # True if metrics improve across passes
print(full.retries.total_errors)      # Total error count
```

Individual reports can be generated separately:

```python
from orcheval.report import cost_report, routing_report

cost = cost_report(trace)
routing = routing_report(trace)
```

## Saving and Loading Traces

```python
# Serialize to JSON
json_str = trace.to_json()

# Deserialize from JSON
from orcheval import Trace
loaded = Trace.from_json(json_str)

# Dict form for programmatic use
d = trace.to_dict()
loaded = Trace.from_dict(d)
```

## Known Limitations

- **LangGraph routing detection**: LangGraph does not provide explicit routing/conditional-edge
  callbacks. Pass `infer_routing=True` to `Tracer` to emit inferred `RoutingDecision` events
  based on node transition sequences. These carry `metadata={"inferred": True}` and may not
  reflect actual conditional logic. For precise routing data, use the manual adapter's
  `routing_decision()` method.

- **LangGraph pass boundaries**: Multi-pass convergence tracking (`PassBoundary` events) is not
  automatically detected by the LangGraph adapter. Use the manual adapter's `pass_boundary()`
  method to record pass boundaries.

- **Cost data**: `LLMCall.cost` is a passthrough field — it is auto-populated when the LLM
  provider reports cost in the callback response. OrchEval does not include a built-in pricing
  table. When cost is unavailable, the cost report falls back to token counts and call counts.

## License

Apache 2.0
