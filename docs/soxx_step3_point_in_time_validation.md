# SOXX/SOXL Step 3 Point-in-Time Validation and Leakage Audit

Status: implemented.

## Goal

Make the deterministic backtest credible before adding agents, MCP, LangGraph, or LangSmith.

This step answers:

```text
Can the system prove that each prediction only uses data and labels that would have been available as of that prediction date?
```

Step 3 is not a model-improvement step. It is an auditability step.

## What This Step Adds

- `as_of_date` validation
- temporal metadata on feature rows
- feature availability checks
- target-label availability checks
- walk-forward training-window checks
- future-data leakage checks
- deterministic artifact hashes
- a validation/leakage report written with each run

## What "Temporal Features" Means

In Step 3, temporal fields are mostly audit metadata, not new predictive alpha features.

The current feature row already has a market date:

```text
date
```

Step 3 adds fields that describe when each part of that row is allowed to be used:

```text
feature_available_date
target_available_date_1d
target_available_date_5d
target_end_date_1d
target_end_date_5d
```

For example, for a row dated `2024-01-02`:

```text
SOXX ret_5d, vol_20d, drawdown_20d, volume_z20
```

are based only on prices and volume up through `2024-01-02`, so their feature availability date is `2024-01-02`.

But the 5-day forward target for that same row is not known on `2024-01-02`. It is only known after the future 5-day return has completed. So:

```text
target_available_date_5d > 2024-01-02
```

The leakage guard then enforces:

```text
prediction_date >= feature_available_date
prediction_date >= training_row_target_available_date
```

This means the model can use past feature rows and past labels only after those labels would have existed in real time.

## What This Step Does Not Add

- LLM agents
- LangGraph orchestration
- LangSmith tracing
- MCP servers
- SEC/fundamental data
- macro data
- news/event extraction
- a new trading strategy
- new claims that the model is profitable

Those come after the deterministic backtest is auditable.

## New Files

Paths are relative to `SOXX_agentic/`.

```text
soxx_mvp/temporal.py
soxx_mvp/leakage.py
soxx_mvp/artifacts.py
tests/test_temporal_guard.py
tests/test_leakage_audit.py
docs/soxx_step3_point_in_time_validation.md
```

## Updated Files

```text
scripts/run_soxx_mvp.py
soxx_mvp/features.py
soxx_mvp/reports.py
tests/test_backtest_step2.py
```

## Runtime Flow

The main command should remain:

```text
scripts/run_soxx_mvp.py
```

Step 3 inserts validation around the existing deterministic flow:

```text
configs/backtest_full.json
        |
        v
scripts/run_soxx_mvp.py
        |
        v
load prices
        |
        v
build feature rows with temporal metadata
        |
        v
validate point-in-time feature availability
        |
        v
run walk-forward backtest
        |
        v
audit prediction/training windows for leakage
        |
        v
write artifacts, validation report, and artifact manifest
```

## Validation Rules

### 1. Config Date Validation

The runner should fail if:

```text
start_date > end_date
as_of_date < end_date
validation/test split dates are malformed
test period overlaps validation period
```

### 2. Feature Availability Validation

The runner should fail if:

```text
feature_available_date > date
feature_available_date > as_of_date
date > as_of_date
dates are unsorted
dates are duplicated
```

### 3. Target Availability Validation

The runner should fail if a training row uses a target before that target is available.

For a 5-day target:

```text
target_available_date_5d must be <= prediction_date
```

This is the core point-in-time rule.

### 4. Walk-Forward Window Validation

For horizon `h`, each prediction should satisfy:

```text
train_end_date <= prediction_date shifted back by h trading rows
n_train >= min_train_rows
train_start_date <= train_end_date
```

The current Step 2 backtest already does this implicitly with row indexing. Step 3 makes it explicit and auditable.

### 5. Feature Column Leakage Validation

The model feature set must reject columns such as:

```text
target_return_1d
target_direction_1d
target_return_5d
target_direction_5d
actual_return
actual_direction
buy_hold_return
strategy_return
```

The model can train on approved market features only.

## New Outputs

Step 3 should preserve all Step 2 outputs:

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
```

Step 3 should add:

```text
leakage_report.json
validation_report.md
artifact_manifest.json
```

### `leakage_report.json`

Machine-readable audit result:

```json
{
  "status": "passed",
  "as_of_date": "2026-05-07",
  "checks": [
    {
      "name": "feature_availability",
      "status": "passed"
    },
    {
      "name": "target_availability",
      "status": "passed"
    }
  ]
}
```

### `validation_report.md`

Human-readable summary:

```text
Point-in-time validation passed.
No feature rows were dated after as_of_date.
No training labels were used before target availability.
No disallowed target/actual columns entered the model feature matrix.
```

### `artifact_manifest.json`

Hash manifest for reproducibility:

```json
{
  "run_id": "20260518T214856Z",
  "config_hash": "sha256:...",
  "artifacts": {
    "features.csv": "sha256:...",
    "metrics.json": "sha256:..."
  }
}
```

## Environment

From `SOXX_agentic/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

If the venv already exists:

```bash
cd SOXX_agentic
source .venv/bin/activate
```

## Run

Run tests:

```bash
python -m pytest
```

Run a fast deterministic sample-data check:

```bash
python scripts/run_soxx_mvp.py \
  --config configs/backtest_full.json \
  --sample-data \
  --output-dir artifacts/soxx_step3_sample
```

Run the cached/live full backtest:

```bash
python scripts/run_soxx_mvp.py --config configs/backtest_full.json
```

If the local Python install has certificate issues:

```bash
python scripts/run_soxx_mvp.py --config configs/backtest_full.json --insecure-ssl
```

## Check Outputs

After a Step 3 run, check:

```bash
ls artifacts/soxx_step3_sample
cat artifacts/soxx_step3_sample/validation_report.md
cat artifacts/soxx_step3_sample/leakage_report.json
cat artifacts/soxx_step3_sample/artifact_manifest.json
```

For a timestamped full run:

```bash
ls artifacts/soxx_mvp
```

Then open the newest directory under:

```text
artifacts/soxx_mvp/{run_id}/
```

## Tests

Expected test coverage:

```text
temporal config validation
feature row date ordering
feature availability cannot be after prediction date
training target availability cannot be after prediction date
target columns cannot enter model feature columns
synthetic leaked feature data fails loudly
synthetic leaked training-window data fails loudly
runner writes Step 3 audit artifacts
```

## Acceptance Criteria

Step 3 is complete when:

```text
python -m pytest passes
sample-data run succeeds
full cached/live run succeeds
leakage_report.json is written
validation_report.md is written
artifact_manifest.json is written
intentional leakage tests fail loudly
existing Step 2 metrics and baseline outputs are still produced
```

## Why This Matters

The next project layers will add agents that propose experiments, summarize results, and eventually orchestrate research workflows.

Before that, deterministic code needs to prove:

```text
the backtest did not cheat
the data timeline is auditable
model inputs are separated from future labels
artifacts can be hashed and traced
```

That is the point of Step 3.
