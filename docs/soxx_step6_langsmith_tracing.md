# SOXX/SOXL Step 6 LangSmith Tracing

Status: implemented.

## Goal

Add soft opt-in LangSmith tracing around the Step 5 LangGraph runner without changing deterministic feature, model, backtest, leakage, or artifact behavior.

Step 6 answers:

```text
Can every graph run be inspected and reproduced from compact trace metadata when LangSmith is configured?
```

This step does not add LLM agents, new market data sources, new features, or model changes.

## What This Step Adds

- Optional `langsmith` dependency
- A tracing helper module: `soxx_mvp/tracing.py`
- A traced graph invocation path for `scripts/run_soxx_graph.py`
- CLI options for LangSmith project override and custom trace tags
- Step 6 tests for metadata summaries, no-credential behavior, mocked LangSmith tracing, and failed graph traces

## Soft Opt-In Behavior

LangSmith tracing is enabled only when the environment includes:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=...
```

Optionally set:

```bash
export LANGSMITH_PROJECT=soxx-agentic-research
```

If tracing is not configured, graph runs continue locally and produce the same artifacts as Step 5.

## Run

From `SOXX_agentic/`:

```bash
./.venv/bin/python scripts/run_soxx_graph.py \
  --config configs/backtest_full.json \
  --sample-data \
  --output-dir artifacts/soxx_step6_sample
```

With an explicit LangSmith project and tag:

```bash
LANGSMITH_TRACING=true LANGSMITH_API_KEY=... \
./.venv/bin/python scripts/run_soxx_graph.py \
  --config configs/backtest_full.json \
  --sample-data \
  --output-dir artifacts/soxx_step6_sample \
  --langsmith-project soxx-agentic-research \
  --trace-tag sample-run
```

## Trace Contents

The root trace is named:

```text
SOXX/SOXL Graph Run
```

Default tags:

```text
soxx
soxl
langgraph
step6
```

Trace metadata includes:

```text
run_id
config_path
config_hash
sample_data
as_of_date
start_date
end_date
tickers
horizons
model_config
feature_set name/hash/count/columns
feature_matrix_hash
output_dir
artifact_names
artifact_paths
artifact_hashes
per-horizon model metrics
point-in-time validation status
leakage audit status/error count
graph status/completed_nodes/failed_node/error
```

Trace metadata intentionally excludes full price data, feature rows, prediction rows, raw environment dumps, and credentials.

## Failure Behavior

Graph node failures still route through the Step 5 failure state:

```text
status = failed
failed_node = node name
error_type = exception class
error_message = exception message
```

When LangSmith tracing is enabled, the trace records that failure summary even if `WriteArtifacts` is not reached and no artifact hashes exist.

## Acceptance Criteria

Step 6 is complete when:

```text
pytest passes
graph run succeeds without LangSmith credentials
graph run uses LangSmith trace wrapper when configured
trace metadata includes config hash, feature set, artifact paths/hashes, metrics, and leakage status
trace metadata excludes large raw state fields
failed graph runs record failed node and error metadata
existing deterministic script runner still works unchanged
```
