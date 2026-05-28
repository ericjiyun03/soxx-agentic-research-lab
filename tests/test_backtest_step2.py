from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from soxx_mvp.backtest import run_walkforward
from soxx_mvp.baselines import baseline_positions
from soxx_mvp.data import generate_sample_prices
from soxx_mvp.features import build_feature_rows


def _sample_rows() -> list[dict[str, float | int | str]]:
    prices = generate_sample_prices(["SOXX", "SOXL", "SMH"], "2019-01-01", "2020-12-31")
    return build_feature_rows(
        prices,
        primary_ticker="SOXX",
        leveraged_ticker="SOXL",
        comparison_ticker="SMH",
        horizons=[1, 5],
    )


def _run_small_backtest(**overrides: object) -> dict[str, object]:
    params = {
        "rows": _sample_rows(),
        "horizon": 1,
        "train_window_rows": 60,
        "min_train_rows": 60,
        "prediction_stride": 5,
        "max_backtest_predictions": 20,
        "long_threshold": 0.55,
        "short_threshold": 0.45,
        "transaction_cost_bps": 5.0,
        "logistic_config": {"max_iter": 100, "l2": 0.1},
        "ridge_config": {"l2": 0.1},
        "cost_sensitivity_bps": [0.0, 25.0],
        "primary_ticker": "SOXX",
    }
    params.update(overrides)
    return run_walkforward(**params)


def test_baseline_predictions_align_to_model_prediction_dates() -> None:
    result = _run_small_backtest()
    model_dates = [row["date"] for row in result["model_predictions"]]
    for baseline in ["buy_hold", "momentum", "mean_reversion", "volatility_regime_momentum"]:
        baseline_dates = [
            row["date"] for row in result["baseline_predictions"] if row["baseline"] == baseline
        ]
        assert baseline_dates == model_dates


def test_transaction_cost_sensitivity_reduces_or_preserves_returns() -> None:
    result = _run_small_backtest()
    model_rows = [
        row
        for row in result["cost_sensitivity"]
        if row["strategy"] == "model" and row["split"] == "all"
    ]
    by_cost = {row["transaction_cost_bps"]: row["mean_strategy_return"] for row in model_rows}
    assert by_cost[25.0] <= by_cost[0.0]


def test_validation_and_test_split_labels_are_applied() -> None:
    result = _run_small_backtest(
        max_backtest_predictions=None,
        validation_start_date="2019-06-03",
        validation_end_date="2019-12-31",
        test_start_date="2020-01-02",
        test_end_date="2020-12-31",
    )
    splits = {row["split"] for row in result["model_predictions"]}
    assert splits == {"validation", "test"}


def test_volatility_regime_baseline_uses_training_window_median() -> None:
    row = pd.Series({"soxx_ret_5d": 0.02, "soxx_vol_20d": 0.30})
    train_rows = pd.DataFrame({"soxx_vol_20d": [0.10, 0.12, 0.14]})
    assert baseline_positions(row, train_rows)["volatility_regime_momentum"] == 0

    row = pd.Series({"soxx_ret_5d": 0.02, "soxx_vol_20d": 0.11})
    assert baseline_positions(row, train_rows)["volatility_regime_momentum"] == 1


def test_runner_writes_step2_artifacts_with_sample_data(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = {
        "project": "soxx-soxl-agentic-research-lab",
        "as_of_date": "2020-12-31",
        "start_date": "2019-01-01",
        "end_date": "2020-12-31",
        "tickers": ["SOXX", "SOXL", "SMH"],
        "primary_ticker": "SOXX",
        "leveraged_ticker": "SOXL",
        "comparison_ticker": "SMH",
        "horizons": [1, 5],
        "train_window_rows": 60,
        "min_train_rows": 60,
        "prediction_stride_by_horizon": {"1": 5, "5": 5},
        "max_backtest_predictions": None,
        "validation_start_date": "2019-06-03",
        "validation_end_date": "2019-12-31",
        "test_start_date": "2020-01-02",
        "test_end_date": "2020-12-31",
        "long_threshold": 0.55,
        "short_threshold": 0.45,
        "transaction_cost_bps": 5.0,
        "cost_sensitivity_bps": [0.0, 5.0, 25.0],
        "logistic": {"max_iter": 100, "l2": 0.1},
        "ridge": {"l2": 0.1},
    }
    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "artifacts"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/run_soxx_mvp.py",
            "--config",
            str(config_path),
            "--sample-data",
            "--output-dir",
            str(output_dir),
        ],
        cwd=project_root,
        check=True,
    )

    expected_files = [
        "features.csv",
        "predictions_h1.csv",
        "predictions_h5.csv",
        "baseline_predictions_h1.csv",
        "baseline_predictions_h5.csv",
        "strategy_comparison_h1.csv",
        "strategy_comparison_h5.csv",
        "cost_sensitivity_h1.csv",
        "cost_sensitivity_h5.csv",
        "metrics.json",
        "latest_forecast.json",
        "run_config.json",
        "backtest_report.md",
        "leakage_report.json",
        "validation_report.md",
        "artifact_manifest.json",
    ]
    for filename in expected_files:
        assert (output_dir / filename).exists()

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "artifact_manifest.json").read_text(encoding="utf-8"))

    assert metrics["feature_set"]["name"] == "all_market_v1"
    assert metrics["feature_set"]["selected_feature_count"] > 0
    assert metrics["point_in_time_validation"]["feature_set"]["status"] == "passed"
    assert run_config["feature_set"]["hash"] == metrics["feature_set"]["hash"]
    assert manifest["feature_set"]["hash"] == metrics["feature_set"]["hash"]
