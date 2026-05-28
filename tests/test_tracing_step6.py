from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from soxx_mvp.artifacts import sha256_json
from soxx_mvp.graph import invoke_soxx_graph
from soxx_mvp.pipeline import (
    NODE_VALIDATE_POINT_IN_TIME_DATA,
    RunOptions,
    initial_pipeline_state,
)
from soxx_mvp import tracing


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


def _write_config(tmp_path: Path, **overrides: object) -> tuple[Path, dict[str, object]]:
    config = _base_config(**overrides)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, config


def _initial_state(config_path: Path, output_dir: Path, run_id: str = "step6-test") -> dict[str, Any]:
    return initial_pipeline_state(
        RunOptions(
            config_path=config_path,
            output_dir=output_dir,
            sample_data=True,
            run_id=run_id,
        )
    )


def test_trace_metadata_summarizes_successful_graph_run(tmp_path: Path) -> None:
    config_path, config = _write_config(tmp_path)
    final_state = invoke_soxx_graph(_initial_state(config_path, tmp_path / "artifacts"))

    metadata = tracing.build_trace_metadata(final_state)

    assert metadata["run_id"] == "step6-test"
    assert metadata["config_hash"] == sha256_json(config)
    assert metadata["sample_data"] is True
    assert metadata["as_of_date"] == "2020-12-31"
    assert metadata["feature_set"]["name"] == "all_market_v1"
    assert metadata["feature_set"]["hash"].startswith("sha256:")
    assert metadata["feature_matrix_hash"].startswith("sha256:")
    assert metadata["artifact_paths"]["features.csv"].endswith("features.csv")
    assert metadata["artifact_hashes"]["features.csv"].startswith("sha256:")
    assert metadata["metrics"]["1d"]["prediction_count"] == 10
    assert metadata["metrics"]["1d"]["model"]["test"]["brier_score"] >= 0.0
    assert metadata["point_in_time_validation"]["leakage_status"] == "passed"
    assert metadata["graph"]["status"] == "completed"
    assert "rows" not in metadata
    assert "prices" not in metadata
    assert "predictions_by_horizon" not in metadata


def test_traced_wrapper_is_noop_without_langsmith_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _ = _write_config(tmp_path)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    def fail_if_loaded() -> Any:
        raise AssertionError("LangSmith should not be loaded without credentials")

    monkeypatch.setattr(tracing, "_load_langsmith", fail_if_loaded)

    final_state = tracing.invoke_soxx_graph_with_tracing(
        _initial_state(config_path, tmp_path / "artifacts")
    )

    assert final_state["status"] == "completed"
    assert final_state["artifact_paths"]["metrics.json"].endswith("metrics.json")


def test_traced_wrapper_uses_langsmith_trace_and_flushes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _ = _write_config(tmp_path)
    fake_langsmith = _FakeLangSmith()
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setattr(tracing, "_load_langsmith", lambda: fake_langsmith)

    final_state = tracing.invoke_soxx_graph_with_tracing(
        _initial_state(config_path, tmp_path / "artifacts"),
        langsmith_project="soxx-test-project",
        trace_tags=["unit-test"],
    )

    assert final_state["status"] == "completed"
    assert fake_langsmith.trace_call["name"] == tracing.DEFAULT_TRACE_NAME
    assert fake_langsmith.trace_call["project_name"] == "soxx-test-project"
    assert "unit-test" in fake_langsmith.trace_call["tags"]
    assert fake_langsmith.client.flushed is True
    assert fake_langsmith.run.end_calls[-1]["outputs"]["status"] == "completed"
    assert fake_langsmith.run.end_calls[-1]["metadata"]["artifact_hashes"]["features.csv"].startswith(
        "sha256:"
    )


def test_failed_graph_run_is_summarized_in_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _ = _write_config(tmp_path, feature_set="unknown_feature_set")
    fake_langsmith = _FakeLangSmith()
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setattr(tracing, "_load_langsmith", lambda: fake_langsmith)

    final_state = tracing.invoke_soxx_graph_with_tracing(
        _initial_state(config_path, tmp_path / "failed_artifacts")
    )
    metadata = fake_langsmith.run.end_calls[-1]["metadata"]

    assert final_state["status"] == "failed"
    assert metadata["graph"]["failed_node"] == NODE_VALIDATE_POINT_IN_TIME_DATA
    assert metadata["artifact_hashes"] == {}
    assert metadata["artifact_paths"] == {}


class _FakeClient:
    def __init__(self) -> None:
        self.flushed = False

    def flush(self) -> None:
        self.flushed = True


class _FakeRun:
    def __init__(self) -> None:
        self.end_calls: list[dict[str, Any]] = []

    def __enter__(self) -> "_FakeRun":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def end(
        self,
        *,
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.end_calls.append(
            {
                "outputs": outputs,
                "error": error,
                "metadata": metadata,
            }
        )


class _FakeLangSmith:
    def __init__(self) -> None:
        self.client = _FakeClient()
        self.run = _FakeRun()
        self.trace_call: dict[str, Any] = {}

    def Client(self) -> _FakeClient:
        return self.client

    def trace(
        self,
        name: str,
        run_type: str,
        *,
        inputs: dict[str, Any],
        project_name: str | None,
        tags: list[str],
        metadata: dict[str, Any],
        client: _FakeClient,
    ) -> _FakeRun:
        self.trace_call = {
            "name": name,
            "run_type": run_type,
            "inputs": inputs,
            "project_name": project_name,
            "tags": tags,
            "metadata": metadata,
            "client": client,
        }
        return self.run
