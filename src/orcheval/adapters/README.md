# orcheval/adapters

Adapters translate framework-specific callbacks into OrchEval's universal `Event` objects. Each adapter implements the `BaseAdapter` contract and emits events via `self._emit(event)`.

## Adapter Contract

```python
class BaseAdapter(ABC):
    def __init__(self, trace_id: str) -> None     # store trace_id, init empty event list
    def get_callback_handler(self) -> Any          # abstract — return framework-specific handler
    def get_events(self) -> list[Event]            # return copy of collected events
    def _emit(self, event: Event) -> None          # append event (called by subclass)
    def reset(self) -> None                        # clear collected events
```

Source: `base.py`

## Existing Adapters

| Adapter | Framework hook | `get_callback_handler()` returns | Constructor flags |
|---|---|---|---|
| `LangGraphAdapter` | `BaseCallbackHandler` (langchain-core) | Callback handler instance | `infer_routing`, `capture_state` |
| `OpenAIAgentsAdapter` | `TracingProcessor` (openai-agents) | Tracing processor instance | `infer_routing`, `capture_state` |
| `ManualAdapter` | None (direct API) | `self` | None |

**`infer_routing`** — Emit `RoutingDecision` events between consecutive nodes. These carry `metadata={"inferred": True}` and may not reflect actual conditional logic. Off by default.

**`capture_state`** — Capture input/output state on `NodeEntry`/`NodeExit`. For LangGraph this is the full graph state dict. For OpenAI Agents SDK this is agent metadata (name, tools, handoffs, output_type). Off by default.

Both framework adapters use `threading.Lock` around `_emit` for thread safety. `ManualAdapter` is not thread-safe.

## Universal Event Types

Adapters must only emit these types from `orcheval.events`:

- **`NodeEntry`** — Node/agent begins execution
- **`NodeExit`** — Node/agent finishes execution (with optional `duration_ms`, `state_diff`)
- **`LLMCall`** — LLM API call completes (model, tokens, cost, messages)
- **`ToolCall`** — Tool/function executes (tool_name, input, output)
- **`RoutingDecision`** — Flow routes from source to target node
- **`AgentMessage`** — Agent-to-agent communication (sender, receiver)
- **`ErrorEvent`** — Error occurred (error_type, error_message)
- **`PassBoundary`** — Multi-pass processing boundary (pass_number, direction)

**Critical rule:** Framework types must never leak into events. Adapters convert all framework objects to primitive Python types (str, int, float, dict, list, None) at the boundary. The `sanitize_state()` utility in `orcheval/sanitize.py` handles Pydantic models, DataFrames, numpy arrays, and circular references.

## Adding a New Adapter

1. Create `src/orcheval/adapters/your_framework.py`
2. Subclass `BaseAdapter`
3. Implement `get_callback_handler()` to return the framework's hook object
4. In the hook, translate framework events into `Event` types and call `self._emit()`
5. Register the adapter name in `Tracer._resolve_adapter()` in `src/orcheval/__init__.py`
6. Add an optional-dependency group in `pyproject.toml`
7. Use the `_ensure_*()` lazy-import guard pattern (see `langgraph.py` line 22)

Skeleton:

```python
from orcheval.adapters.base import BaseAdapter
from orcheval.events import NodeEntry, NodeExit, LLMCall

def _ensure_crewai() -> None:
    try:
        import crewai  # noqa: F401
    except ImportError:
        raise ImportError(
            "crewai is required for the CrewAI adapter. "
            "Install it with: pip install orcheval[crewai]"
        ) from None

class CrewAIAdapter(BaseAdapter):
    def __init__(self, trace_id: str, **kwargs) -> None:
        super().__init__(trace_id)
        _ensure_crewai()
        # ... framework-specific setup

    def get_callback_handler(self):
        # Return whatever hook object CrewAI expects
        return self._handler
```

## Files

- `base.py` — `BaseAdapter` abstract base class
- `manual.py` — `ManualAdapter` with convenience methods for each event type
- `langgraph.py` — `LangGraphAdapter` with LangChain `BaseCallbackHandler`
- `openai_agents.py` — `OpenAIAgentsAdapter` with SDK `TracingProcessor`
- `__init__.py` — Re-exports `BaseAdapter` and `ManualAdapter`
