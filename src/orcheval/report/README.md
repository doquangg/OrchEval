# orcheval/report

Report modules analyze a `Trace` and return frozen Pydantic models. Every report is a pure function `f(Trace) -> FrozenModel` with no side effects.

## Report Modules

| Function | Returns | Analyzes |
|---|---|---|
| `cost_report(trace)` | `CostReport` | Token counts, costs, and model usage per node and per model |
| `routing_report(trace)` | `RoutingReport` | Routing edges, frequencies, and suspicious patterns |
| `convergence_report(trace)` | `ConvergenceReport` | Multi-pass metric trends and convergence status |
| `timeline_report(trace)` | `TimelineReport` | Chronological spans and events with offset/duration |
| `retry_report(trace)` | `RetryReport` | Error clusters, retry sequences, and success rates |
| `llm_patterns_report(trace)` | `LLMPatternsReport` | Behavioral patterns across LLM calls (prompt growth, redundancy, etc.) |

## FullReport and report()

`report(trace)` runs all six modules and returns a `FullReport`:

```python
full = report(trace)
full.cost          # CostReport
full.routing       # RoutingReport
full.convergence   # ConvergenceReport
full.timeline      # TimelineReport
full.retries       # RetryReport
full.llm_patterns  # LLMPatternsReport
```

For targeted analysis, call individual functions directly:

```python
from orcheval.report import cost_report, routing_report
cost = cost_report(trace)
```

## Comparison

`compare_runs()` diffs two traces across all dimensions:

```python
from orcheval import compare_runs
diff = compare_runs(baseline_trace, experiment_trace)
# Or via Trace method:
diff = baseline_trace.compare(experiment_trace)
```

Returns `RunComparison` with `.cost`, `.duration`, `.routing`, `.invocations`, `.errors`, `.convergence`, `.llm_patterns`, and a natural-language `.summary`.

## Pattern Detection

**Routing flags** (in `RoutingReport.flags`):
- `invariant_routing` — Source always routes to the same target
- `context_divergence` — Identical decision context maps to different targets
- `dominant_path` — Single target receives >= 95% of decisions
- `oscillation` — Alternates between two targets >= 3 times

**LLM patterns** (in `LLMPatternsReport.patterns`):
- `prompt_growth` (warning) — Input tokens grow > 50% across invocations
- `repeated_output` (warning) — Identical output despite different inputs
- `redundant_tool_call` (warning) — Same tool call repeated within one invocation
- `system_message_variance` (info) — System message differs across calls in a node
- `output_not_utilized` (info) — LLM output produced but state unchanged

## Adding a Report Module

1. Create `src/orcheval/report/your_module.py`
2. Define frozen Pydantic output models (`model_config = {"frozen": True}`)
3. Write `your_module_report(trace: Trace) -> YourModuleReport` as a pure function
4. Add the field to `FullReport` in `__init__.py`
5. Call it from the `report()` function
6. Add exports to `__all__`

## Files

- `__init__.py` — `FullReport`, `report()`, and all re-exports
- `cost.py` — `CostReport`, `ModelUsage`, `NodeCostSummary`, `cost_report()`
- `routing.py` — `RoutingReport`, `RoutingEdge`, `RoutingFlag`, `routing_report()`
- `convergence.py` — `ConvergenceReport`, `MetricConvergence`, `PassSummary`, `convergence_report()`
- `timeline.py` — `TimelineReport`, `TimelineSpan`, `TimelineEvent`, `timeline_report()`
- `retries.py` — `RetryReport`, `ErrorCluster`, `RetrySequence`, `retry_report()`
- `llm_patterns.py` — `LLMPatternsReport`, `LLMPattern`, `NodeLLMSummary`, `llm_patterns_report()`
- `comparison.py` — `RunComparison`, `compare_runs()`, and all delta/diff models
