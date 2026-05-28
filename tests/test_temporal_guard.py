from __future__ import annotations

import pytest

from soxx_mvp.data import generate_sample_prices
from soxx_mvp.features import build_feature_rows
from soxx_mvp.temporal import (
    TemporalValidationError,
    validate_config_dates,
    validate_feature_rows,
)


def _config() -> dict[str, object]:
    return {
        "as_of_date": "2020-12-31",
        "start_date": "2019-01-01",
        "end_date": "2020-12-31",
        "validation_start_date": "2019-06-03",
        "validation_end_date": "2019-12-31",
        "test_start_date": "2020-01-02",
        "test_end_date": "2020-12-31",
    }


def _sample_rows() -> list[dict[str, float | int | str]]:
    prices = generate_sample_prices(["SOXX", "SOXL", "SMH"], "2019-01-01", "2020-12-31")
    return build_feature_rows(
        prices,
        primary_ticker="SOXX",
        leveraged_ticker="SOXL",
        comparison_ticker="SMH",
        horizons=[1, 5],
    )


def test_feature_rows_include_point_in_time_metadata() -> None:
    rows = _sample_rows()
    first = rows[0]
    assert first["feature_available_date"] == first["date"]
    assert first["target_end_date_1d"] >= first["date"]
    assert first["target_available_date_1d"] == first["target_end_date_1d"]
    assert first["target_available_date_5d"] == first["target_end_date_5d"]


def test_temporal_validation_accepts_sample_rows() -> None:
    config_report = validate_config_dates(_config())
    feature_report = validate_feature_rows(_sample_rows(), as_of_date="2020-12-31", horizons=[1, 5])
    assert config_report["status"] == "passed"
    assert feature_report["status"] == "passed"


def test_temporal_validation_rejects_future_feature_availability() -> None:
    rows = _sample_rows()
    rows[0]["feature_available_date"] = "2099-01-01"
    with pytest.raises(TemporalValidationError):
        validate_feature_rows(rows, as_of_date="2020-12-31", horizons=[1, 5])


def test_config_validation_rejects_overlapping_splits() -> None:
    config = _config()
    config["validation_end_date"] = "2020-03-01"
    config["test_start_date"] = "2020-02-01"
    with pytest.raises(TemporalValidationError):
        validate_config_dates(config)
