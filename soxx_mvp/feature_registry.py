from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_FEATURE_SET_NAME = "all_market_v1"


class FeatureRegistryError(ValueError):
    """Raised when a requested feature set is not allowed for model training."""


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    group: str
    description: str


@dataclass(frozen=True)
class FeatureSelection:
    name: str
    columns: list[str]
    hash: str
    source_path: str
    validation_report: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hash": self.hash,
            "source_path": self.source_path,
            "selected_feature_count": len(self.columns),
            "selected_feature_columns": self.columns,
            "validation": self.validation_report,
        }


FEATURE_DEFINITIONS: tuple[FeatureDefinition, ...] = (
    FeatureDefinition("soxx_ret_1d", "market", "SOXX 1-day trailing return."),
    FeatureDefinition("soxx_ret_5d", "market", "SOXX 5-day trailing return."),
    FeatureDefinition("soxx_ret_20d", "market", "SOXX 20-day trailing return."),
    FeatureDefinition("soxx_vol_20d", "market", "SOXX 20-day realized volatility."),
    FeatureDefinition("soxx_volume_z20", "market", "SOXX 20-day volume z-score."),
    FeatureDefinition("soxl_ret_1d", "instrument", "SOXL 1-day trailing return."),
    FeatureDefinition("soxl_ret_5d", "instrument", "SOXL 5-day trailing return."),
    FeatureDefinition("soxl_ret_20d", "instrument", "SOXL 20-day trailing return."),
    FeatureDefinition("soxl_vol_20d", "instrument", "SOXL 20-day realized volatility."),
    FeatureDefinition("soxl_volume_z20", "instrument", "SOXL 20-day volume z-score."),
    FeatureDefinition("smh_ret_1d", "cross_asset", "SMH 1-day trailing return."),
    FeatureDefinition("smh_ret_5d", "cross_asset", "SMH 5-day trailing return."),
    FeatureDefinition("smh_ret_20d", "cross_asset", "SMH 20-day trailing return."),
    FeatureDefinition("smh_vol_20d", "cross_asset", "SMH 20-day realized volatility."),
    FeatureDefinition("smh_volume_z20", "cross_asset", "SMH 20-day volume z-score."),
    FeatureDefinition(
        "soxx_smh_ret_spread_5d",
        "cross_asset",
        "SOXX minus SMH 5-day return spread.",
    ),
    FeatureDefinition(
        "soxl_leverage_realized_5d",
        "instrument",
        "SOXL 5-day return divided by SOXX 5-day return.",
    ),
    FeatureDefinition("soxx_drawdown_20d", "risk", "SOXX 20-day rolling drawdown."),
)

DISALLOWED_FEATURE_PREFIXES = (
    "target_",
    "actual_",
    "strategy_",
    "gross_strategy_",
)

DISALLOWED_FEATURE_COLUMNS = {
    "buy_hold_return",
    "date",
    "feature_available_date",
    "horizon",
    "position",
    "primary_close",
    "leveraged_close",
    "prob_up",
    "pred_direction",
    "pred_return",
    "row_idx",
    "split",
    "trading_cost",
    "train_start_date",
    "train_end_date",
    "transaction_cost_bps",
    "turnover",
}


def feature_registry() -> dict[str, FeatureDefinition]:
    return {definition.name: definition for definition in FEATURE_DEFINITIONS}


def registered_feature_columns() -> list[str]:
    return [definition.name for definition in FEATURE_DEFINITIONS]


def load_feature_sets(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_feature_sets = payload.get("feature_sets", payload)
    if not isinstance(raw_feature_sets, dict):
        raise FeatureRegistryError("feature_sets config must be a JSON object")

    feature_sets: dict[str, list[str]] = {}
    for name, columns in raw_feature_sets.items():
        if not isinstance(name, str) or not name:
            raise FeatureRegistryError("feature set names must be non-empty strings")
        if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns):
            raise FeatureRegistryError(f"feature set {name} must be a list of strings")
        feature_sets[name] = list(columns)
    return feature_sets


def resolve_feature_set(
    *,
    feature_set_name: str | None,
    feature_sets_path: Path,
    rows: list[dict[str, float | int | str]] | None = None,
) -> FeatureSelection:
    name = feature_set_name or DEFAULT_FEATURE_SET_NAME
    feature_sets = load_feature_sets(feature_sets_path)
    if name not in feature_sets:
        raise FeatureRegistryError(f"unknown feature set: {name}")

    columns = list(feature_sets[name])
    validation_report = validate_feature_columns(columns, rows=rows)
    return FeatureSelection(
        name=name,
        columns=columns,
        hash=feature_set_hash(name=name, columns=columns),
        source_path=str(feature_sets_path),
        validation_report=validation_report,
    )


def validate_feature_columns(
    columns: list[str],
    *,
    rows: list[dict[str, float | int | str]] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    registry = feature_registry()

    if not columns:
        errors.append("feature set must contain at least one feature")

    seen: set[str] = set()
    for column in columns:
        lowered = column.lower()
        if column in seen:
            errors.append(f"duplicate feature column: {column}")
            continue
        seen.add(column)

        if lowered in DISALLOWED_FEATURE_COLUMNS or any(
            lowered.startswith(prefix) for prefix in DISALLOWED_FEATURE_PREFIXES
        ):
            errors.append(f"feature column is not allowed as a model input: {column}")
            continue
        if column not in registry:
            errors.append(f"unknown feature column: {column}")

    if rows and not errors:
        row_columns = set(rows[0].keys())
        for column in columns:
            if column not in row_columns:
                errors.append(f"feature column is missing from feature rows: {column}")

        for row_idx, row in enumerate(rows):
            for column in columns:
                value = row.get(column)
                if not isinstance(value, (float, int)) or math.isnan(float(value)) or math.isinf(float(value)):
                    errors.append(f"rows[{row_idx}].{column} is not a finite numeric value")
                    break
            if errors:
                break

    report = {
        "name": "feature_set",
        "status": "failed" if errors else "passed",
        "error_count": len(errors),
        "errors": errors,
        "summary": {
            "selected_feature_count": len(columns),
            "selected_feature_columns": columns,
        },
    }
    if errors:
        preview = "; ".join(errors[:5])
        raise FeatureRegistryError(f"feature set validation failed: {preview}")
    return report


def feature_set_hash(*, name: str, columns: list[str]) -> str:
    encoded = json.dumps(
        {"name": name, "columns": columns},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
