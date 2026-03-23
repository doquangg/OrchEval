# tests

## Running Tests

```bash
pytest                      # unit tests (no optional deps required)
pytest -m integration       # adapter tests (requires langgraph or openai-agents)
pytest --cov                # with coverage
```

The project uses `--strict-markers` and `-ra -q` by default (see `pyproject.toml`).

## Test Layers

- **Unit** — Direct function calls with manually constructed events. Covers events, trace, reports, exports, and collection.
- **Integration** — Real framework adapters (`LangGraphAdapter`, `OpenAIAgentsAdapter`) with synthetic graphs/agents. Marked with `@pytest.mark.integration`, skipped if optional deps are missing.
- **Round-trip** — Serialization tests (JSON, `model_dump`/`model_validate`) ensuring events and traces survive encoding cycles.

## Fixtures (conftest.py)

All fixtures use deterministic timestamps via the `_ts(seconds)` helper, which returns `BASE_TIME + timedelta(seconds=seconds)` where `BASE_TIME = 2025-01-15T10:00:00Z` and `TRACE_ID = "test-trace-0001"`.

Key fixtures:

| Fixture | Description |
|---|---|
| `sample_events` / `sample_trace` | 2-node pipeline (agent + summarizer) with LLM calls and a tool call |
| `routing_events` | Invariant routing, context divergence, and dominant path patterns |
| `oscillation_events` | Alternating routing targets (8 decisions) |
| `multipass_events` | 3 converging passes with improving metrics |
| `error_retry_events` | Retry pattern (codegen retries once, validator fails) |
| `multi_model_events` | Multiple models (gpt-4o, gpt-4o-mini) across nodes |
| `stateful_trace` | State capture with input/output state and state diffs |
| `llm_pattern_events` / `llm_pattern_trace` | Triggers all 5 LLM pattern detectors |

## Adding Tests

- Name files `test_<module>.py`
- Use `ManualAdapter` to construct test traces — no framework deps needed
- Use `_ts(seconds)` for deterministic timestamps
- Mark adapter tests with `@pytest.mark.integration`
- No tests require real LLM API calls or cost money
