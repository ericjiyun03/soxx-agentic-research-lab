# SOXX/SOXL Step 5 LangGraph Wrapper

Status: implemented.

## Goal

Introduce LangGraph orchestration without changing deterministic model, feature, validation, leakage, or artifact logic.

Step 5 answers:

```text
Can the existing deterministic SOXX/SOXL workflow run as structured graph nodes and produce the same artifacts as the script runner?
```

This step does not add LangSmith tracing, LLM agents, MCP tools, new market data sources, or new alpha features.

## What This Step Adds

- A shared deterministic workflow module: `soxx_mvp/pipeline.py`
- A LangGraph wrapper under `soxx_mvp/graph/`
- A graph CLI: `scripts/run_soxx_graph.py`
- A `--run-id` option for deterministic artifact comparisons
- Step 5 tests for graph artifacts, metrics, and failure routing
- `langgraph>=1.0,<2.0` in `requirements.txt`

## Key Architecture

The existing domain logic stays in its original modules:

```text
soxx_mvp/data.py
soxx_mvp/features.py
soxx_mvp/feature_registry.py
soxx_mvp/backtest.py
soxx_mvp/leakage.py
soxx_mvp/reports.py
soxx_mvp/artifacts.py
```

`pipeline.py` consolidates the orchestration logic that used to live inside `scripts/run_soxx_mvp.py`. It calls the existing functions in order and passes outputs through a shared `state` dictionary.

```text
Original modules implement behavior
        |
        v
pipeline.py defines named workflow steps
        |
        v
run_soxx_mvp.py runs steps directly
run_soxx_graph.py runs the same steps through LangGraph
```

## Pipeline Steps

Current named steps:

```text
LoadConfig
PullMarketData
MaterializeFeatures
ValidatePointInTimeData
RunForecastModels
RunBacktest
WriteArtifacts
```

Each step receives the current `state` dictionary and returns an updated state.

Example:

```text
initial state:
config_path, output_dir, sample_data

LoadConfig adds:
config, horizons, run_id

PullMarketData adds:
tickers, prices

MaterializeFeatures adds:
rows

ValidatePointInTimeData adds:
feature_selection, validation reports

RunForecastModels adds:
horizon_results, predictions_by_horizon, forecasts

RunBacktest adds:
all_metrics, leakage_report

WriteArtifacts adds:
artifact_paths
```

## Deterministic Runner vs LangGraph Runner

The deterministic runner uses a plain Python loop:

```text
scripts/run_soxx_mvp.py
  -> run_deterministic_pipeline()
  -> loop over PIPELINE_STEPS
```

The LangGraph runner uses graph nodes and guarded edges:

```text
scripts/run_soxx_graph.py
  -> invoke_soxx_graph()
  -> StateGraph
  -> graph node wrappers
  -> same pipeline step functions
```

Both runners call the same underlying step functions from `pipeline.py`. The difference is execution mechanism:

```text
deterministic runner: direct for-loop
LangGraph runner: StateGraph nodes with conditional routing
```

## Failure Behavior

In the deterministic runner, exceptions fail the process normally.

In the LangGraph runner, each node is wrapped by `_run_node()` in `soxx_mvp/graph/nodes.py`. If a step raises, the graph records:

```text
status = failed
failed_node = node name
error_type = exception class
error_message = exception message
```

Routing then stops the graph before later nodes run. In particular, failed validation or model training should not reach `WriteArtifacts`.

## Files

```text
requirements.txt
scripts/run_soxx_mvp.py
scripts/run_soxx_graph.py
soxx_mvp/pipeline.py
soxx_mvp/graph/__init__.py
soxx_mvp/graph/state.py
soxx_mvp/graph/nodes.py
soxx_mvp/graph/routing.py
soxx_mvp/graph/main_graph.py
tests/test_graph_step5.py
```

## Outputs

The graph runner preserves the same artifact contract as the deterministic runner:

```text
features.csv
predictions_h1.csv
predictions_h5.csv
baseline_predictions_h1.csv
baseline_predictions_h5.csv
strategy_comparison_h1.csv
strategy_comparison_h5.csv
cost_sensitivity_h1.csv
cost_sensitivity_h5.csv
metrics.json
latest_forecast.json
run_config.json
backtest_report.md
leakage_report.json
validation_report.md
artifact_manifest.json
```

## Run

From `SOXX_agentic/`:

```bash
./.venv/bin/python -m pytest
```

Sample-data graph run:

```bash
./.venv/bin/python scripts/run_soxx_graph.py \
  --config configs/backtest_full.json \
  --sample-data \
  --output-dir artifacts/soxx_step5_sample
```

Live or cached market-data graph run:

```bash
./.venv/bin/python scripts/run_soxx_graph.py \
  --config configs/backtest_full.json \
  --output-dir artifacts/soxx_step5_live
```

If local certificates fail during live fetches, add:

```bash
--insecure-ssl
```

## Acceptance Criteria

Step 5 is complete when:

```text
pytest passes
graph sample-data run succeeds
graph run writes the expected artifact set
graph metrics match the deterministic runner for the same config and run_id
invalid feature set stops at ValidatePointInTimeData
invalid model config stops at RunForecastModels
WriteArtifacts is not reached after graph node failures
existing deterministic script runner still works
```
