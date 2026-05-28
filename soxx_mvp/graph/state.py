from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict


class SoxxGraphState(TypedDict, total=False):
    config_path: Path
    output_dir: Path | None
    cache_dir: Path
    run_id: str | None
    sample_data: bool
    refresh: bool
    insecure_ssl: bool
    config: dict[str, Any]
    horizons: list[int]
    tickers: list[str]
    prices: dict[str, Any]
    rows: list[dict[str, float | int | str]]
    feature_selection: Any
    config_validation_report: dict[str, Any]
    feature_validation_report: dict[str, Any]
    horizon_results: dict[int, dict[str, Any]]
    predictions_by_horizon: dict[int, list[dict[str, Any]]]
    forecasts: list[dict[str, Any]]
    all_metrics: dict[str, Any]
    leakage_report: dict[str, Any]
    artifact_paths: dict[str, str]
    completed_nodes: list[str]
    status: str
    failed_node: str
    error_type: str
    error_message: str
