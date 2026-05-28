from __future__ import annotations

import pytest

from soxx_mvp.backtest import run_walkforward
from soxx_mvp.data import generate_sample_prices
from soxx_mvp.features import build_feature_rows, feature_columns
from soxx_mvp.leakage import (
    LeakageAuditError,
    assert_leakage_report_passed,
    build_leakage_report,
)


def _sample_rows() -> list[dict[str, float | int | str]]:
    prices = generate_sample_prices(["SOXX", "SOXL", "SMH"], "2019-01-01", "2020-12-31")
    return build_feature_rows(
        prices,
        primary_ticker="SOXX",
        leveraged_ticker="SOXL",
        comparison_ticker="SMH",
        horizons=[1, 5],
    )


def _run_small_backtest(
    rows: list[dict[str, float | int | str]],
    *,
    horizon: int = 1,
) -> dict[str, object]:
    return run_walkforward(
        rows,
        horizon=horizon,
        train_window_rows=60,
        min_train_rows=60,
        prediction_stride=5,
        max_backtest_predictions=10,
        long_threshold=0.55,
        short_threshold=0.45,
        transaction_cost_bps=5.0,
        logistic_config={"max_iter": 100, "l2": 0.1},
        ridge_config={"l2": 0.1},
        primary_ticker="SOXX",
    )


def test_leakage_audit_accepts_walkforward_predictions() -> None:
    rows = _sample_rows()
    result = _run_small_backtest(rows)
    report = build_leakage_report(
        rows=rows,
        predictions_by_horizon={1: result["model_predictions"]},
        model_feature_columns=feature_columns(),
        as_of_date="2020-12-31",
    )
    assert report["status"] == "passed"
    assert report["error_count"] == 0


def test_leakage_audit_rejects_target_columns_as_model_inputs() -> None:
    rows = _sample_rows()
    result = _run_small_backtest(rows)
    report = build_leakage_report(
        rows=rows,
        predictions_by_horizon={1: result["model_predictions"]},
        model_feature_columns=feature_columns() + ["target_return_1d"],
        as_of_date="2020-12-31",
    )
    assert report["status"] == "failed"
    with pytest.raises(LeakageAuditError):
        assert_leakage_report_passed(report)


def test_leakage_audit_rejects_unavailable_training_label() -> None:
    rows = _sample_rows()
    result = _run_small_backtest(rows)
    first_prediction = result["model_predictions"][0]
    for row in rows:
        if first_prediction["train_start_date"] <= row["date"] <= first_prediction["train_end_date"]:
            row["target_available_date_1d"] = "2099-01-01"
            break

    report = build_leakage_report(
        rows=rows,
        predictions_by_horizon={1: result["model_predictions"]},
        model_feature_columns=feature_columns(),
        as_of_date="2020-12-31",
    )
    assert report["status"] == "failed"
    assert any("target label is unavailable" in error for error in report["errors"])
