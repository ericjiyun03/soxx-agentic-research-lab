from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .artifacts import build_artifact_manifest
from .backtest import latest_forecast, run_walkforward
from .data import generate_sample_prices, load_or_fetch_prices
from .feature_registry import FeatureSelection, resolve_feature_set
from .features import build_feature_rows
from .io_utils import project_root, read_json, write_csv, write_json, write_text
from .leakage import assert_leakage_report_passed, build_leakage_report
from .reports import render_backtest_report, render_validation_report
from .temporal import validate_config_dates, validate_feature_rows


NODE_LOAD_CONFIG = "LoadConfig"
NODE_PULL_MARKET_DATA = "PullMarketData"
NODE_MATERIALIZE_FEATURES = "MaterializeFeatures"
NODE_VALIDATE_POINT_IN_TIME_DATA = "ValidatePointInTimeData"
NODE_RUN_FORECAST_MODELS = "RunForecastModels"
NODE_RUN_BACKTEST = "RunBacktest"
NODE_WRITE_ARTIFACTS = "WriteArtifacts"

PIPELINE_STEPS: tuple[tuple[str, Callable[[dict[str, Any]], dict[str, Any]]], ...] = (
    (NODE_LOAD_CONFIG, lambda state: load_config_step(state)),
    (NODE_PULL_MARKET_DATA, lambda state: pull_market_data_step(state)),
    (NODE_MATERIALIZE_FEATURES, lambda state: materialize_features_step(state)),
    (NODE_VALIDATE_POINT_IN_TIME_DATA, lambda state: validate_point_in_time_data_step(state)),
    (NODE_RUN_FORECAST_MODELS, lambda state: run_forecast_models_step(state)),
    (NODE_RUN_BACKTEST, lambda state: run_backtest_step(state)),
    (NODE_WRITE_ARTIFACTS, lambda state: write_artifacts_step(state)),
)


@dataclass(frozen=True)
class RunOptions:
    config_path: Path
    output_dir: Path | None = None
    refresh: bool = False
    sample_data: bool = False
    insecure_ssl: bool = False
    run_id: str | None = None


def initial_pipeline_state(options: RunOptions) -> dict[str, Any]:
    return {
        "config_path": Path(options.config_path),
        "output_dir": Path(options.output_dir) if options.output_dir is not None else None,
        "cache_dir": project_root() / "data" / "market_cache",
        "run_id": options.run_id,
        "sample_data": bool(options.sample_data),
        "refresh": bool(options.refresh),
        "insecure_ssl": bool(options.insecure_ssl),
        "completed_nodes": [],
        "status": "running",
        "failed_node": "",
        "error_type": "",
        "error_message": "",
        "artifact_paths": {},
    }


def run_deterministic_pipeline(options: RunOptions) -> dict[str, Any]:
    state = initial_pipeline_state(options)
    for node_name, step in PIPELINE_STEPS:
        state = step(state)
        state = mark_node_completed(state, node_name)
    state["status"] = "completed"
    return state


def mark_node_completed(state: dict[str, Any], node_name: str) -> dict[str, Any]:
    next_state = dict(state)
    next_state["completed_nodes"] = [*next_state.get("completed_nodes", []), node_name]
    return next_state


def load_config_step(state: dict[str, Any]) -> dict[str, Any]:
    config_path = Path(state["config_path"])
    config = read_json(config_path)
    horizons = [int(value) for value in config["horizons"]]
    config_validation_report = validate_config_dates(config)
    run_id = str(state.get("run_id") or _utc_run_id())
    output_dir = _resolve_output_dir(state.get("output_dir"), run_id)

    return {
        **state,
        "config_path": config_path,
        "config": config,
        "horizons": horizons,
        "config_validation_report": config_validation_report,
        "run_id": run_id,
        "output_dir": output_dir,
        "cache_dir": Path(state.get("cache_dir") or project_root() / "data" / "market_cache"),
    }


def pull_market_data_step(state: dict[str, Any]) -> dict[str, Any]:
    config = state["config"]
    tickers = list(config["tickers"])
    if state.get("sample_data"):
        prices = generate_sample_prices(tickers, config["start_date"], config["end_date"])
    else:
        prices = load_or_fetch_prices(
            tickers,
            config["start_date"],
            config["end_date"],
            Path(state["cache_dir"]),
            refresh=bool(state.get("refresh")),
            insecure_ssl=bool(state.get("insecure_ssl")),
        )

    return {
        **state,
        "tickers": tickers,
        "prices": prices,
    }


def materialize_features_step(state: dict[str, Any]) -> dict[str, Any]:
    config = state["config"]
    rows = build_feature_rows(
        state["prices"],
        primary_ticker=config["primary_ticker"],
        leveraged_ticker=config["leveraged_ticker"],
        comparison_ticker=config["comparison_ticker"],
        horizons=state["horizons"],
    )
    if not rows:
        raise ValueError("No feature rows generated")

    return {
        **state,
        "rows": rows,
    }


def validate_point_in_time_data_step(state: dict[str, Any]) -> dict[str, Any]:
    config = state["config"]
    rows = state["rows"]
    feature_validation_report = validate_feature_rows(
        rows,
        as_of_date=config["as_of_date"],
        horizons=state["horizons"],
    )
    feature_selection = resolve_feature_set(
        feature_set_name=config.get("feature_set"),
        feature_sets_path=_feature_sets_path(config, Path(state["config_path"])),
        rows=rows,
    )

    return {
        **state,
        "feature_validation_report": feature_validation_report,
        "feature_selection": feature_selection,
    }


def run_forecast_models_step(state: dict[str, Any]) -> dict[str, Any]:
    config = state["config"]
    rows = state["rows"]
    feature_selection: FeatureSelection = state["feature_selection"]
    horizon_results: dict[int, dict[str, Any]] = {}
    predictions_by_horizon: dict[int, list[dict[str, Any]]] = {}
    forecasts: list[dict[str, Any]] = []

    for horizon in state["horizons"]:
        prediction_stride = _prediction_stride(config, horizon)
        result = run_walkforward(
            rows,
            horizon=horizon,
            train_window_rows=int(config["train_window_rows"]),
            min_train_rows=int(config["min_train_rows"]),
            prediction_stride=prediction_stride,
            max_backtest_predictions=(
                int(config["max_backtest_predictions"])
                if config.get("max_backtest_predictions") is not None
                else None
            ),
            long_threshold=float(config["long_threshold"]),
            short_threshold=float(config["short_threshold"]),
            transaction_cost_bps=float(config["transaction_cost_bps"]),
            logistic_config=config["logistic"],
            ridge_config=config["ridge"],
            validation_start_date=config.get("validation_start_date"),
            validation_end_date=config.get("validation_end_date"),
            test_start_date=config.get("test_start_date"),
            test_end_date=config.get("test_end_date"),
            cost_sensitivity_bps=[float(value) for value in config.get("cost_sensitivity_bps", [])],
            model_feature_columns=feature_selection.columns,
            primary_ticker=config["primary_ticker"],
        )
        horizon_results[horizon] = result
        predictions_by_horizon[horizon] = result["model_predictions"]

        try:
            forecasts.append(
                latest_forecast(
                    rows,
                    horizon=horizon,
                    train_window_rows=int(config["train_window_rows"]),
                    min_train_rows=int(config["min_train_rows"]),
                    logistic_config=config["logistic"],
                    ridge_config=config["ridge"],
                    model_feature_columns=feature_selection.columns,
                )
            )
        except ValueError as exc:
            forecasts.append({"horizon": horizon, "error": str(exc)})

    return {
        **state,
        "horizon_results": horizon_results,
        "predictions_by_horizon": predictions_by_horizon,
        "forecasts": forecasts,
    }


def run_backtest_step(state: dict[str, Any]) -> dict[str, Any]:
    config = state["config"]
    rows = state["rows"]
    feature_selection: FeatureSelection = state["feature_selection"]

    all_metrics: dict[str, Any] = {
        "run_id": state["run_id"],
        "sample_data": bool(state.get("sample_data")),
        "feature_row_count": len(rows),
        "first_feature_date": rows[0]["date"],
        "last_feature_date": rows[-1]["date"],
        "feature_set": feature_selection.summary(),
        "point_in_time_validation": {
            "config_dates": state["config_validation_report"],
            "feature_rows": state["feature_validation_report"],
            "feature_set": feature_selection.validation_report,
        },
        "horizons": {},
    }

    for horizon in state["horizons"]:
        result = state["horizon_results"][horizon]
        all_metrics["horizons"][f"{horizon}d"] = result["metrics"]

    leakage_report = build_leakage_report(
        rows=rows,
        predictions_by_horizon=state["predictions_by_horizon"],
        model_feature_columns=feature_selection.columns,
        as_of_date=config["as_of_date"],
    )
    assert_leakage_report_passed(leakage_report)
    all_metrics["point_in_time_validation"]["leakage"] = {
        "status": leakage_report["status"],
        "error_count": leakage_report["error_count"],
    }

    return {
        **state,
        "all_metrics": all_metrics,
        "leakage_report": leakage_report,
    }


def write_artifacts_step(state: dict[str, Any]) -> dict[str, Any]:
    config = state["config"]
    output_dir = Path(state["output_dir"])
    feature_selection: FeatureSelection = state["feature_selection"]
    artifact_paths: dict[str, str] = {}

    def record(path: Path) -> Path:
        artifact_paths[path.name] = str(path)
        return path

    for horizon in state["horizons"]:
        result = state["horizon_results"][horizon]
        write_csv(record(output_dir / f"predictions_h{horizon}.csv"), result["model_predictions"])
        write_csv(record(output_dir / f"baseline_predictions_h{horizon}.csv"), result["baseline_predictions"])
        write_csv(record(output_dir / f"strategy_comparison_h{horizon}.csv"), result["strategy_comparison"])
        write_csv(record(output_dir / f"cost_sensitivity_h{horizon}.csv"), result["cost_sensitivity"])

    write_csv(record(output_dir / "features.csv"), state["rows"])
    write_json(record(output_dir / "metrics.json"), state["all_metrics"])
    write_json(record(output_dir / "latest_forecast.json"), {"forecasts": state["forecasts"]})
    write_text(record(output_dir / "backtest_report.md"), render_backtest_report(state["all_metrics"]))
    write_json(record(output_dir / "leakage_report.json"), state["leakage_report"])
    write_text(
        record(output_dir / "validation_report.md"),
        render_validation_report(
            config_report=state["config_validation_report"],
            feature_report=state["feature_validation_report"],
            leakage_report=state["leakage_report"],
            feature_set_report=feature_selection.validation_report,
        ),
    )
    write_json(
        record(output_dir / "run_config.json"),
        {
            "config": config,
            "sample_data": bool(state.get("sample_data")),
            "output_dir": str(output_dir),
            "feature_set": feature_selection.summary(),
        },
    )
    manifest_path = output_dir / "artifact_manifest.json"
    write_json(
        record(manifest_path),
        build_artifact_manifest(
            output_dir=output_dir,
            run_id=str(state["run_id"]),
            config=config,
            feature_set=feature_selection.summary(),
            source_files=_source_files(
                bool(state.get("sample_data")),
                state["tickers"],
                config,
                Path(state["cache_dir"]),
            ),
        ),
    )

    return {
        **state,
        "artifact_paths": artifact_paths,
    }


def print_run_summary(state: dict[str, Any]) -> None:
    feature_selection: FeatureSelection = state["feature_selection"]
    all_metrics = state["all_metrics"]
    print(f"Wrote artifacts to {state['output_dir']}")
    print(
        "feature_set="
        f"{feature_selection.name} "
        f"features={len(feature_selection.columns)} "
        f"hash={feature_selection.hash}"
    )
    for horizon, metrics in all_metrics["horizons"].items():
        if not metrics:
            print(f"{horizon}: no predictions")
            continue
        model_scores = metrics.get("model", {})
        model_summary = model_scores.get("test") or model_scores.get("all") or {}
        print(
            f"{horizon}: n={metrics['prediction_count']} "
            f"accuracy={model_summary.get('accuracy', 0.0):.3f} "
            f"brier={model_summary.get('brier_score', 0.0):.3f} "
            f"strategy_sharpe={model_summary.get('sharpe', 0.0):.3f}"
        )


def _utc_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _resolve_output_dir(output_dir: Path | str | None, run_id: str) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    return project_root() / "artifacts" / "soxx_mvp" / run_id


def _prediction_stride(config: dict[str, Any], horizon: int) -> int:
    by_horizon = config.get("prediction_stride_by_horizon")
    if isinstance(by_horizon, dict):
        value = by_horizon.get(str(horizon), by_horizon.get(horizon))
        if value is not None:
            return int(value)
    return int(config.get("prediction_stride", 1))


def _feature_sets_path(config: dict[str, Any], config_path: Path) -> Path:
    raw_path = str(config.get("feature_sets_path", "configs/feature_sets.json"))
    path = Path(raw_path)
    if path.is_absolute():
        return path

    project_path = project_root() / path
    if project_path.exists():
        return project_path
    return config_path.parent / path


def _source_files(
    sample_data: bool,
    tickers: list[str],
    config: dict[str, Any],
    cache_dir: Path,
) -> list[Path]:
    if sample_data:
        return []
    return [
        cache_dir / f"{ticker}_{config['start_date']}_{config['end_date']}.csv"
        for ticker in tickers
    ]
