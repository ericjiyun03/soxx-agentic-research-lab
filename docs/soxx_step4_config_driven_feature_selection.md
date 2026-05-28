# SOXX/SOXL Step 4 Config-Driven Feature Selection

Status: implemented.

## Goal

Prepare the deterministic backtest for later agent-proposed experiments without letting agents bypass point-in-time or leakage rules.

Step 4 answers:

```text
Can each run choose a named, validated subset of approved model features and prove exactly which feature set was used?
```

This step does not add agents, LangGraph, LangSmith, MCP, news extraction, SEC data, or new alpha features.

## What This Step Adds

- An approved model-input feature registry
- Named feature sets in config
- Feature-set validation before model training
- Explicit selected feature columns passed into backtest and latest forecast
- Feature-set hash recorded in artifacts
- Tests for invalid, unknown, target, and missing feature columns

## Runtime Flow

```text
configs/backtest_full.json
        |
        v
configs/feature_sets.json
        |
        v
soxx_mvp/feature_registry.py validates selected feature set
        |
        v
scripts/run_soxx_mvp.py passes selected columns into backtest
        |
        v
soxx_mvp/backtest.py trains only on selected columns
        |
        v
leakage audit checks the selected model columns
        |
        v
artifacts record feature set name, columns, and hash
```

## Files

```text
configs/feature_sets.json
soxx_mvp/feature_registry.py
soxx_mvp/features.py
soxx_mvp/backtest.py
soxx_mvp/artifacts.py
soxx_mvp/reports.py
scripts/run_soxx_mvp.py
tests/test_feature_schema.py
```

## Feature Registry

The registry is the allow-list of model inputs. If a column is not registered, the model cannot train on it.

Examples of allowed current features:

```text
soxx_ret_1d
soxx_ret_5d
soxx_vol_20d
soxl_leverage_realized_5d
soxx_drawdown_20d
```

Examples of rejected columns:

```text
target_return_1d
actual_return
strategy_return
future_magic_signal
```

## Feature Sets

Feature sets are named subsets of the registry.

Current sets:

```text
all_market_v1
primary_only_v1
cross_asset_v1
```

The config chooses one:

```json
{
  "feature_set": "all_market_v1",
  "feature_sets_path": "configs/feature_sets.json"
}
```

If those fields are omitted, the runner defaults to `all_market_v1` so older configs keep using the full current feature list.

## Validation Rules

The run fails before training if:

```text
the feature set name is unknown
the feature set is empty
a feature appears twice
a selected feature is not registered
a selected feature is a target, actual, strategy, date, or metadata column
a selected feature is missing from generated feature rows
a selected feature row value is not finite numeric data
```

## Outputs

Step 4 preserves all Step 3 outputs and adds feature-set lineage inside:

```text
metrics.json
run_config.json
artifact_manifest.json
backtest_report.md
validation_report.md
```

Recorded fields include:

```text
feature_set.name
feature_set.hash
feature_set.selected_feature_count
feature_set.selected_feature_columns
feature_set.validation.status
```

## Agent Usage Later

Agents may propose:

```text
use all_market_v1
use primary_only_v1
compare cross_asset_v1 against all_market_v1
```

Agents may not invent executable model inputs.

If an agent proposes a new feature idea, the safe workflow is:

```text
agent proposes feature idea
developer reviews point-in-time validity
deterministic feature code is implemented
tests prove no leakage
feature is added to registry
future agents may select it
```

## Run

From `SOXX_agentic/`:

```bash
python -m pytest
```

Fast sample-data check:

```bash
python scripts/run_soxx_mvp.py \
  --config configs/backtest_full.json \
  --sample-data \
  --output-dir artifacts/soxx_step4_sample
```

Full cached/live backtest:

```bash
python scripts/run_soxx_mvp.py --config configs/backtest_full.json
```

## Acceptance Criteria

Step 4 is complete when:

```text
pytest passes
sample-data run succeeds
full cached/live run succeeds
config can choose feature subsets
unknown feature names fail validation
target and actual columns fail validation
the model trains only on selected approved columns
feature_set hash is recorded in artifacts
Step 3 leakage validation still passes
```
