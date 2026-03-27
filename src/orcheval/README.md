# orcheval package

OrchEval captures structured trace events from multi-agent orchestrations and provides analysis, comparison, and export tools.

## Architecture

```
Adapters (framework hooks)
    ↓ emit Event objects
Events (8 universal types)
    ↓ collected into
Traces (query + summary API)
    ↓ consumed by
Reports / Exports / Collection
```

Adapters produce `Event` objects from framework callbacks. Events are collected into a `Trace`, which feeds report modules, export formats, and cross-run aggregation.

## Public API

Everything in `__all__` (in `__init__.py`) is the stable public surface:

- **Core:** `Tracer`, `Trace`, `NodeInvocation`
- **Events:** `Event`, `AnyEvent`, `NodeEntry`, `NodeExit`, `LLMCall`, `ToolCall`, `RoutingDecision`, `AgentMessage`, `ErrorEvent`, `PassBoundary`
- **Adapters:** `BaseAdapter`, `ManualAdapter`
- **Report:** `FullReport`, `report()`, `RunComparison`, `compare_runs()`
- **Collection:** `TraceCollection`, `CollectionSummary`, `NodeStats`, `PercentileStats`, `ExecutionShape`, `TraceOutlier`, `TrendPoint`, `TrendResult`

Framework-specific adapters (`LangGraphAdapter`, `OpenAIAgentsAdapter`) are **not** re-exported and are considered internal.

## Module Map

| File/Directory | Contents |
|---|---|
| `__init__.py` | `Tracer` class, public API re-exports, `_resolve_adapter()` |
| `events.py` | 8 event types, `AnyEvent` discriminated union, `EVENT_ADAPTER` TypeAdapter |
| `trace.py` | `Trace` container (query, summary, serialization, export delegation), `NodeInvocation` |
| `collection.py` | `TraceCollection` with `summary()`, `find_outliers()`, `execution_shapes()`, `trend()` |
| `sanitize.py` | `sanitize_state()` and `compute_state_diff()` — state sanitization for adapters |
| `adapters/` | Framework-specific event producers ([README](adapters/README.md)) |
| `report/` | Analysis modules — 6 reports + comparison ([README](report/README.md)) |
| `export/` | Output formats: `dataframe.py`, `digest.py`, `mermaid.py`, `visualization.py` |

## Design Principles

- **Events are frozen Pydantic models.** Immutable after creation, JSON-serializable.
- **Trace is a plain class**, not a Pydantic model. It provides query methods (`get_llm_calls()`, `get_events_by_node()`), summary methods (`total_cost()`, `total_tokens()`), and export delegation (`to_mermaid()`, `to_html()`, etc.).
- **Export methods** on `Trace` (e.g., `to_dataframe()`, `to_digest()`) delegate to functions in `export/`.
- **Optional dependencies** (langgraph, openai-agents, pandas) are lazy-imported at point of use.
- **All report functions are pure** — `f(Trace) -> FrozenModel`, no side effects.

## Span Linking Model

Events are linked into a hierarchy using two matching strategies:

```
NodeEntry  (span_id="node-001")
  ├─ LLMCall    (span_id="llm-001",  parent_span_id="node-001")
  ├─ ToolCall   (span_id="tool-001", parent_span_id="node-001")
  ├─ ErrorEvent (span_id="err-001",  parent_span_id="node-001")
  └─ NodeExit   (span_id="node-001")   ← same span_id, NOT parent_span_id
```

1. **Entry/exit pairs** — `NodeEntry` and `NodeExit` share the same `span_id`. They define the boundaries of a node invocation.
2. **Child events** — `LLMCall`, `ToolCall`, and `ErrorEvent` set `parent_span_id` to the `span_id` of their enclosing node. They get their own unique `span_id`.

The canonical implementation of this logic is `_invocation_events()` in `export/digest.py`, which collects all events belonging to a single node invocation using both strategies.
