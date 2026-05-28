from __future__ import annotations

from typing import Any

from .data import parse_date


class LeakageAuditError(ValueError):
    """Raised when the leakage audit detects an invalid backtest timeline."""


DISALLOWED_FEATURE_PREFIXES = (
    "target_",
    "actual_",
    "strategy_",
    "gross_strategy_",
)

DISALLOWED_FEATURE_COLUMNS = {
    "buy_hold_return",
    "feature_available_date",
    "row_idx",
    "date",
    "horizon",
    "position",
    "prob_up",
    "pred_direction",
    "pred_return",
    "split",
    "trading_cost",
    "train_start_date",
    "train_end_date",
    "transaction_cost_bps",
    "turnover",
}


def build_leakage_report(
    *,
    rows: list[dict[str, float | int | str]],
    predictions_by_horizon: dict[int, list[dict[str, Any]]],
    model_feature_columns: list[str],
    as_of_date: str,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    feature_column_errors = _feature_column_errors(model_feature_columns)
    checks.append(_check("model_feature_columns", feature_column_errors))

    window_errors, window_summary = _prediction_window_errors(
        rows=rows,
        predictions_by_horizon=predictions_by_horizon,
        as_of_date=as_of_date,
    )
    checks.append(_check("prediction_training_windows", window_errors, summary=window_summary))

    errors = [error for check in checks for error in check["errors"]]
    return {
        "status": "failed" if errors else "passed",
        "as_of_date": as_of_date,
        "error_count": len(errors),
        "errors": errors,
        "checks": checks,
    }


def assert_leakage_report_passed(report: dict[str, Any]) -> None:
    if report.get("status") == "passed":
        return
    errors = report.get("errors") or []
    preview = "; ".join(str(error) for error in errors[:5])
    raise LeakageAuditError(f"leakage audit failed: {preview}")


def _feature_column_errors(columns: list[str]) -> list[str]:
    errors: list[str] = []
    for column in columns:
        lowered = column.lower()
        if lowered in DISALLOWED_FEATURE_COLUMNS:
            errors.append(f"model feature column is not point-in-time input: {column}")
            continue
        if any(lowered.startswith(prefix) for prefix in DISALLOWED_FEATURE_PREFIXES):
            errors.append(f"model feature column uses disallowed prefix: {column}")
            continue
        if lowered.startswith("target_available_date_") or lowered.startswith("target_end_date_"):
            errors.append(f"model feature column uses target temporal metadata: {column}")
    return errors


def _prediction_window_errors(
    *,
    rows: list[dict[str, float | int | str]],
    predictions_by_horizon: dict[int, list[dict[str, Any]]],
    as_of_date: str,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    sorted_rows = sorted(rows, key=lambda row: str(row["date"]))
    date_to_row = {str(row["date"]): row for row in sorted_rows}
    sorted_dates = [str(row["date"]) for row in sorted_rows]
    date_to_index = {date_value: idx for idx, date_value in enumerate(sorted_dates)}
    as_of = _parse_or_record("as_of_date", as_of_date, errors)
    prediction_count = 0

    for horizon, predictions in predictions_by_horizon.items():
        target_return_col = f"target_return_{horizon}d"
        target_direction_col = f"target_direction_{horizon}d"
        target_available_col = f"target_available_date_{horizon}d"
        prediction_count += len(predictions)

        for prediction_idx, prediction in enumerate(predictions):
            prefix = f"horizon {horizon} prediction {prediction_idx}"
            prediction_date_value = str(prediction.get("date", ""))
            train_start_value = str(prediction.get("train_start_date", ""))
            train_end_value = str(prediction.get("train_end_date", ""))

            prediction_date = _parse_or_record(f"{prefix} date", prediction_date_value, errors)
            if as_of and prediction_date and prediction_date > as_of:
                errors.append(f"{prefix} prediction date is after as_of_date: {prediction_date_value}")

            if prediction_date_value not in date_to_index:
                errors.append(f"{prefix} date is missing from feature rows: {prediction_date_value}")
                continue
            if train_start_value not in date_to_index:
                errors.append(f"{prefix} train_start_date is missing from feature rows: {train_start_value}")
                continue
            if train_end_value not in date_to_index:
                errors.append(f"{prefix} train_end_date is missing from feature rows: {train_end_value}")
                continue

            prediction_row = date_to_row[prediction_date_value]
            feature_available_value = str(prediction_row.get("feature_available_date", ""))
            feature_available_date = _parse_or_record(
                f"{prefix} feature_available_date",
                feature_available_value,
                errors,
            )
            if feature_available_date and prediction_date and feature_available_date > prediction_date:
                errors.append(
                    f"{prefix} feature_available_date is after prediction date: {feature_available_value}"
                )

            prediction_row_idx = date_to_index[prediction_date_value]
            train_start_idx = date_to_index[train_start_value]
            train_end_idx = date_to_index[train_end_value]
            if train_start_idx > train_end_idx:
                errors.append(f"{prefix} train_start_date is after train_end_date")
                continue
            if train_end_idx > prediction_row_idx - int(horizon):
                errors.append(
                    f"{prefix} train_end_date is too close to prediction date for {horizon}d horizon"
                )

            eligible_train_count = 0
            for train_date_value in sorted_dates[train_start_idx : train_end_idx + 1]:
                train_row = date_to_row[train_date_value]
                if _is_blank(train_row.get(target_return_col)) or _is_blank(
                    train_row.get(target_direction_col)
                ):
                    errors.append(
                        f"{prefix} training row {train_date_value} lacks {horizon}d target label"
                    )
                    continue

                train_feature_available = _parse_or_record(
                    f"{prefix} training row {train_date_value} feature_available_date",
                    str(train_row.get("feature_available_date", "")),
                    errors,
                )
                if (
                    train_feature_available
                    and prediction_date
                    and train_feature_available > prediction_date
                ):
                    errors.append(
                        f"{prefix} training row {train_date_value} feature is unavailable at prediction date"
                    )

                target_available_value = str(train_row.get(target_available_col, ""))
                target_available_date = _parse_or_record(
                    f"{prefix} training row {train_date_value} {target_available_col}",
                    target_available_value,
                    errors,
                )
                if target_available_date is None:
                    continue
                if target_available_date and prediction_date and target_available_date > prediction_date:
                    errors.append(
                        f"{prefix} training row {train_date_value} target label is unavailable at prediction date"
                    )
                else:
                    eligible_train_count += 1

            expected_train_count = int(prediction.get("n_train", 0))
            if eligible_train_count != expected_train_count:
                errors.append(
                    f"{prefix} n_train mismatch: prediction has {expected_train_count}, "
                    f"audit found {eligible_train_count}"
                )

    return errors, {"prediction_count": prediction_count, "horizons": sorted(predictions_by_horizon)}


def _check(name: str, errors: list[str], summary: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "status": "failed" if errors else "passed",
        "error_count": len(errors),
        "errors": errors,
        "summary": summary or {},
    }


def _parse_or_record(field: str, value: str, errors: list[str]):
    try:
        return parse_date(value)
    except (TypeError, ValueError):
        errors.append(f"{field} is not YYYY-MM-DD: {value}")
        return None


def _is_blank(value: object) -> bool:
    return value is None or value == ""
