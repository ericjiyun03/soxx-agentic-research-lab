from __future__ import annotations

import json
from pathlib import Path

import pytest

from soxx_mvp.backtest import run_walkforward
from soxx_mvp.data import generate_sample_prices
from soxx_mvp.feature_registry import FeatureRegistryError, resolve_feature_set
from soxx_mvp.features import build_feature_rows, feature_columns


def _sample_rows() -> list[dict[str, float | int | str]]:
    prices = generate_sample_prices(["SOXX", "SOXL", "SMH"], "2019-01-01", "2020-12-31")
    return build_feature_rows(
        prices,
        primary_ticker="SOXX",
        leveraged_ticker="SOXL",
        comparison_ticker="SMH",
        horizons=[1, 5],
    )


def _write_feature_sets(tmp_path: Path, columns: list[str], name: str = "test_set") -> Path:
    path = tmp_path / "feature_sets.json"
    path.write_text(json.dumps({"feature_sets": {name: columns}}), encoding="utf-8")
    return path


def test_default_feature_set_matches_current_full_feature_columns() -> None:
    project_root = Path(__file__).resolve().parents[1]
    selection = resolve_feature_set(
        feature_set_name="all_market_v1",
        feature_sets_path=project_root / "configs" / "feature_sets.json",
        rows=_sample_rows(),
    )
    assert selection.columns == feature_columns()
    assert selection.validation_report["status"] == "passed"
    assert selection.hash.startswith("sha256:")


def test_unknown_feature_name_fails_validation(tmp_path: Path) -> None:
    path = _write_feature_sets(tmp_path, ["soxx_ret_1d", "future_magic_signal"])
    with pytest.raises(FeatureRegistryError):
        resolve_feature_set(feature_set_name="test_set", feature_sets_path=path, rows=_sample_rows())


def test_target_column_fails_validation(tmp_path: Path) -> None:
    path = _write_feature_sets(tmp_path, ["soxx_ret_1d", "target_return_1d"])
    with pytest.raises(FeatureRegistryError):
        resolve_feature_set(feature_set_name="test_set", feature_sets_path=path, rows=_sample_rows())


def test_missing_feature_row_column_fails_validation(tmp_path: Path) -> None:
    path = _write_feature_sets(tmp_path, ["soxx_ret_1d", "soxx_ret_5d"])
    rows = [{"date": "2020-01-01", "feature_available_date": "2020-01-01", "soxx_ret_1d": 0.01}]
    with pytest.raises(FeatureRegistryError):
        resolve_feature_set(feature_set_name="test_set", feature_sets_path=path, rows=rows)


def test_walkforward_uses_selected_feature_columns_only() -> None:
    rows = _sample_rows()
    columns = ["soxx_ret_1d", "soxx_ret_5d", "soxx_vol_20d"]
    result = run_walkforward(
        rows,
        horizon=1,
        train_window_rows=60,
        min_train_rows=60,
        prediction_stride=5,
        max_backtest_predictions=10,
        long_threshold=0.55,
        short_threshold=0.45,
        transaction_cost_bps=5.0,
        logistic_config={"max_iter": 100, "l2": 0.1},
        ridge_config={"l2": 0.1},
        model_feature_columns=columns,
        primary_ticker="SOXX",
    )

    assert result["model_feature_columns"] == columns
    assert result["metrics"]["selected_feature_columns"] == columns
    assert result["metrics"]["selected_feature_count"] == len(columns)
