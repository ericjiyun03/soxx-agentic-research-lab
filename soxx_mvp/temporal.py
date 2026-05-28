from __future__ import annotations

from datetime import date
from typing import Any

from .data import parse_date


class TemporalValidationError(ValueError):
    """Raised when point-in-time date validation fails."""


def validate_config_dates(config: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    required_fields = ["as_of_date", "start_date", "end_date"]
    parsed: dict[str, date] = {}

    for field in required_fields:
        value = config.get(field)
        if not isinstance(value, str) or not value:
            errors.append(f"{field} is required")
            continue
        try:
            parsed[field] = parse_date(value)
        except ValueError:
            errors.append(f"{field} is not YYYY-MM-DD: {value}")

    if not errors:
        if parsed["start_date"] > parsed["end_date"]:
            errors.append("start_date must be <= end_date")
        if parsed["end_date"] > parsed["as_of_date"]:
            errors.append("end_date must be <= as_of_date")

    split_ranges = [
        ("validation", config.get("validation_start_date"), config.get("validation_end_date")),
        ("test", config.get("test_start_date"), config.get("test_end_date")),
    ]
    normalized_ranges: dict[str, tuple[date, date]] = {}
    for name, start_value, end_value in split_ranges:
        if start_value is None and end_value is None:
            continue
        if not isinstance(start_value, str) or not isinstance(end_value, str):
            errors.append(f"{name} split requires both start and end dates")
            continue
        try:
            start = parse_date(start_value)
            end = parse_date(end_value)
        except ValueError:
            errors.append(f"{name} split dates must be YYYY-MM-DD")
            continue
        if start > end:
            errors.append(f"{name} split start date must be <= end date")
        if "as_of_date" in parsed and end > parsed["as_of_date"]:
            errors.append(f"{name} split end date must be <= as_of_date")
        normalized_ranges[name] = (start, end)

    if "validation" in normalized_ranges and "test" in normalized_ranges:
        validation = normalized_ranges["validation"]
        test = normalized_ranges["test"]
        if max(validation[0], test[0]) <= min(validation[1], test[1]):
            errors.append("validation and test date ranges must not overlap")

    return _report_or_raise(
        name="config_dates",
        errors=errors,
        summary={
            "as_of_date": config.get("as_of_date"),
            "start_date": config.get("start_date"),
            "end_date": config.get("end_date"),
        },
    )


def validate_feature_rows(
    rows: list[dict[str, float | int | str]],
    *,
    as_of_date: str,
    horizons: list[int],
) -> dict[str, Any]:
    errors: list[str] = []
    if not rows:
        errors.append("feature rows are empty")
        return _report_or_raise("feature_rows", errors=errors, summary={})

    as_of = _parse_or_error("as_of_date", as_of_date, errors)
    dates: list[str] = []
    seen: set[str] = set()

    for idx, row in enumerate(rows):
        row_date_value = str(row.get("date", ""))
        dates.append(row_date_value)
        row_date = _parse_or_error(f"rows[{idx}].date", row_date_value, errors)
        feature_available_value = str(row.get("feature_available_date", ""))
        feature_available_date = _parse_or_error(
            f"rows[{idx}].feature_available_date",
            feature_available_value,
            errors,
        )

        if row_date_value in seen:
            errors.append(f"duplicate feature row date: {row_date_value}")
        seen.add(row_date_value)

        if row_date and as_of and row_date > as_of:
            errors.append(f"feature row date after as_of_date: {row_date_value}")
        if feature_available_date and row_date and feature_available_date > row_date:
            errors.append(
                f"feature_available_date after row date for {row_date_value}: {feature_available_value}"
            )
        if feature_available_date and as_of and feature_available_date > as_of:
            errors.append(
                f"feature_available_date after as_of_date for {row_date_value}: {feature_available_value}"
            )

        for horizon in horizons:
            _validate_target_temporal_fields(row, idx, horizon, row_date, as_of, errors)

    if dates != sorted(dates):
        errors.append("feature row dates must be sorted ascending")

    return _report_or_raise(
        name="feature_rows",
        errors=errors,
        summary={
            "as_of_date": as_of_date,
            "row_count": len(rows),
            "first_feature_date": rows[0].get("date"),
            "last_feature_date": rows[-1].get("date"),
            "horizons": horizons,
        },
    )


def _validate_target_temporal_fields(
    row: dict[str, float | int | str],
    idx: int,
    horizon: int,
    row_date: date | None,
    as_of: date | None,
    errors: list[str],
) -> None:
    target_return = row.get(f"target_return_{horizon}d")
    target_direction = row.get(f"target_direction_{horizon}d")
    target_end_value = row.get(f"target_end_date_{horizon}d")
    target_available_value = row.get(f"target_available_date_{horizon}d")

    has_target = not _is_blank(target_return) and not _is_blank(target_direction)
    has_temporal = not _is_blank(target_end_value) and not _is_blank(target_available_value)

    if has_target and not has_temporal:
        errors.append(f"rows[{idx}] has target_{horizon}d without target temporal metadata")
        return
    if not has_target and has_temporal:
        errors.append(f"rows[{idx}] has target temporal metadata without target_{horizon}d")
        return
    if not has_target:
        return

    target_end_date = _parse_or_error(
        f"rows[{idx}].target_end_date_{horizon}d",
        str(target_end_value),
        errors,
    )
    target_available_date = _parse_or_error(
        f"rows[{idx}].target_available_date_{horizon}d",
        str(target_available_value),
        errors,
    )

    if row_date and target_end_date and target_end_date < row_date:
        errors.append(f"target_end_date_{horizon}d is before row date at rows[{idx}]")
    if target_end_date and target_available_date and target_available_date < target_end_date:
        errors.append(f"target_available_date_{horizon}d is before target_end_date at rows[{idx}]")
    if as_of and target_available_date and target_available_date > as_of:
        errors.append(f"target_available_date_{horizon}d after as_of_date at rows[{idx}]")


def _report_or_raise(name: str, *, errors: list[str], summary: dict[str, Any]) -> dict[str, Any]:
    report = {
        "name": name,
        "status": "failed" if errors else "passed",
        "error_count": len(errors),
        "errors": errors,
        "summary": summary,
    }
    if errors:
        preview = "; ".join(errors[:5])
        raise TemporalValidationError(f"{name} validation failed: {preview}")
    return report


def _parse_or_error(field: str, value: str, errors: list[str]) -> date | None:
    try:
        return parse_date(value)
    except (TypeError, ValueError):
        errors.append(f"{field} is not YYYY-MM-DD: {value}")
        return None


def _is_blank(value: object) -> bool:
    return value is None or value == ""
