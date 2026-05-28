from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from soxx_mvp.graph import invoke_soxx_graph
from soxx_mvp.pipeline import (
    NODE_LOAD_CONFIG,
    NODE_MATERIALIZE_FEATURES,
    NODE_PULL_MARKET_DATA,
    NODE_RUN_FORECAST_MODELS,
    NODE_VALIDATE_POINT_IN_TIME_DATA,
    NODE_WRITE_ARTIFACTS,
    RunOptions,
    initial_pipeline_state,
)


EXPECTED_ARTIFACTS = {
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
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _base_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "project": "soxx-soxl-agentic-research-lab",
        "as_of_date": "2020-12-31",
        "start_date": "2019-01-01",
        "end_date": "2020-12-31",
        "tickers": ["SOXX", "SOXL", "SMH"],
        "primary_ticker": "SOXX",
        "leveraged_ticker": "SOXL",
        "comparison_ticker": "SMH",
        "feature_set": "all_market_v1",
        "feature_sets_path": "configs/feature_sets.json",
        "horizons": [1, 5],
        "train_window_rows": 60,
        "min_train_rows": 60,
        "prediction_stride_by_horizon": {"1": 5, "5": 5},
        "max_backtest_predictions": 10,
        "validation_start_date": "2019-06-03",
        "validation_end_date": "2019-12-31",
        "test_start_date": "2020-01-02",
        "test_end_date": "2020-12-31",
        "long_threshold": 0.55,
        "short_threshold": 0.45,
        "transaction_cost_bps": 5.0,
        "cost_sensitivity_bps": [0.0, 5.0],
        "logistic": {"max_iter": 100, "l2": 0.1},
        "ridge": {"l2": 0.1},
    }
    config.update(overrides)
    return config


def _write_config(tmp_path: Path, **overrides: object) -> Path:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_base_config(**overrides)), encoding="utf-8")
    return config_path


def _invoke_graph(config_path: Path, output_dir: Path) -> dict[str, object]:
    return invoke_soxx_graph(
        initial_pipeline_state(
            RunOptions(
                config_path=config_path,
                output_dir=output_dir,
                sample_data=True,
                run_id="step5-test",
            )
        )
    )


def test_graph_sample_data_run_matches_script_artifact_contract(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    script_output = tmp_path / "script"
    graph_output = tmp_path / "graph"
    project_root = _project_root()

    subprocess.run(
        [
            sys.executable,
            "scripts/run_soxx_mvp.py",
            "--config",
            str(config_path),
            "--sample-data",
            "--output-dir",
            str(script_output),
            "--run-id",
            "step5-test",
        ],
        cwd=project_root,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/run_soxx_graph.py",
            "--config",
            str(config_path),
            "--sample-data",
            "--output-dir",
            str(graph_output),
            "--run-id",
            "step5-test",
        ],
        cwd=project_root,
        check=True,
    )

    assert {path.name for path in script_output.iterdir()} == EXPECTED_ARTIFACTS
    assert {path.name for path in graph_output.iterdir()} == EXPECTED_ARTIFACTS

    script_metrics = json.loads((script_output / "metrics.json").read_text(encoding="utf-8"))
    graph_metrics = json.loads((graph_output / "metrics.json").read_text(encoding="utf-8"))
    assert graph_metrics == script_metrics
    assert graph_metrics["feature_set"]["name"] == "all_market_v1"
    assert graph_metrics["feature_set"]["hash"].startswith("sha256:")
    assert sorted(graph_metrics["horizons"]) == ["1d", "5d"]
    assert graph_metrics["point_in_time_validation"]["feature_set"]["status"] == "passed"
    assert graph_metrics["point_in_time_validation"]["leakage"]["status"] == "passed"
    assert graph_metrics["horizons"]["1d"]["selected_feature_columns"]


def test_invalid_feature_set_stops_at_point_in_time_validation(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, feature_set="unknown_feature_set")
    output_dir = tmp_path / "failed_feature_set"

    final_state = _invoke_graph(config_path, output_dir)

    assert final_state["status"] == "failed"
    assert final_state["failed_node"] == NODE_VALIDATE_POINT_IN_TIME_DATA
    assert final_state["completed_nodes"] == [
        NODE_LOAD_CONFIG,
        NODE_PULL_MARKET_DATA,
        NODE_MATERIALIZE_FEATURES,
    ]
    assert NODE_WRITE_ARTIFACTS not in final_state["completed_nodes"]
    assert not output_dir.exists()


def test_invalid_model_config_stops_at_run_forecast_models(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, ridge={"alpha": -1.0})
    output_dir = tmp_path / "failed_model"

    final_state = _invoke_graph(config_path, output_dir)

    assert final_state["status"] == "failed"
    assert final_state["failed_node"] == NODE_RUN_FORECAST_MODELS
    assert NODE_VALIDATE_POINT_IN_TIME_DATA in final_state["completed_nodes"]
    assert NODE_WRITE_ARTIFACTS not in final_state["completed_nodes"]
    assert not output_dir.exists()
