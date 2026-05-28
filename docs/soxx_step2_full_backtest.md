# SOXX/SOXL Step 2 Full Walk-Forward Backtest

## Goal

Turn the Step 1 smoke-test MVP into a fuller deterministic backtest with sklearn models, baseline comparisons, validation/test windows, and transaction-cost sensitivity.

This step still does not add agents, MCP, LangGraph, LangSmith, SEC data, macro data, or news extraction.

## Environment

From `SOXX_agentic/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

## Run

Use deterministic sample data:

```bash
python scripts/run_soxx_mvp.py --config configs/backtest_full.json --sample-data --output-dir artifacts/soxx_step2_sample
```

Use cached/live Yahoo chart API data:

```bash
python scripts/run_soxx_mvp.py --config configs/backtest_full.json
```

If the local Python install has certificate issues:

```bash
python scripts/run_soxx_mvp.py --config configs/backtest_full.json --insecure-ssl
```

## What Changed From Step 1

- Models now use `scikit-learn`:
  - `StandardScaler` + `LogisticRegression` for direction
  - `StandardScaler` + `Ridge` for forward returns
- Walk-forward data handling now uses `pandas`.
- The full config runs uncapped historical predictions.
- The 5-day horizon uses stride `5` by default to avoid overlapping 5-day return rows in headline metrics.
- The backtest compares the model against deterministic baselines:
  - buy and hold
  - momentum
  - mean reversion
  - volatility-regime-filtered momentum
- Metrics are reported by split:
  - validation
  - test
  - all

## Outputs

Default output directory:

```text
artifacts/soxx_mvp/{run_id}/
```

Step 2 writes:

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

## Tests

```bash
python -m pytest
```

The tests cover sklearn wrapper determinism, single-class logistic fallback behavior, baseline/date alignment, split labeling, transaction-cost sensitivity, volatility-regime baseline behavior, and runner artifact creation.
