# examples

Self-contained scripts demonstrating OrchEval's features. No LLM API keys or framework dependencies required — all examples use the ManualAdapter to simulate traces.

| Script | What it shows |
|---|---|
| `quickstart_manual.py` | Build a trace, generate a report, export HTML and JSON |
| `diagnose_bad_architecture.py` | Detect prompt growth, redundant tool calls, oscillating routing, retries |
| `compare_two_runs.py` | Diff two traces: cost, duration, routing, errors, LLM patterns |
| `cross_run_analysis.py` | Aggregate stats, outlier detection, trend analysis, execution shape clustering |

## Running

```bash
pip install orcheval
python examples/quickstart_manual.py
```

Each script prints analysis to stdout and writes HTML visualizations to `orcheval_outputs/`. Open the HTML files in any browser.
