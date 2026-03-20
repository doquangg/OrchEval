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

## License

Apache 2.0
