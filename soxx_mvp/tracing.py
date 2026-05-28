from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Mapping

from .artifacts import sha256_json
from .graph import invoke_soxx_graph
from .io_utils import project_root, read_json


DEFAULT_TRACE_NAME = "SOXX/SOXL Graph Run"
DEFAULT_TRACE_TAGS = ("soxx", "soxl", "langgraph", "step6")
TRUTHY_ENV_VALUES = {"1", "true", "t", "yes", "y", "on"}


def langsmith_tracing_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = env if env is not None else os.environ
    tracing_value = values.get("LANGSMITH_TRACING") or values.get("LANGSMITH_TRACING_V2") or ""
    return tracing_value.strip().lower() in TRUTHY_ENV_VALUES and bool(
        values.get("LANGSMITH_API_KEY")
    )


def invoke_soxx_graph_with_tracing(
    initial_state: dict[str, Any],
    *,
    langsmith_project: str | None = None,
    trace_tags: list[str] | None = None,
) -> dict[str, Any]:
    if not langsmith_tracing_enabled():
        return invoke_soxx_graph(initial_state)

    try:
        langsmith = _load_langsmith()
        client = langsmith.Client()
    except Exception:
        return invoke_soxx_graph(initial_state)

    tags = _trace_tags(trace_tags)
    initial_metadata = build_trace_metadata(initial_state)
    runnable_config = {
        "run_name": DEFAULT_TRACE_NAME,
        "tags": tags,
        "metadata": initial_metadata,
    }

    try:
        with langsmith.trace(
            DEFAULT_TRACE_NAME,
            "chain",
            inputs=_trace_inputs(initial_state),
            project_name=langsmith_project,
            tags=tags,
            metadata=initial_metadata,
            client=client,
        ) as run_tree:
            try:
                final_state = invoke_soxx_graph(initial_state, config=runnable_config)
            except Exception as exc:
                failure_metadata = build_trace_metadata(
                    {
                        **initial_state,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                run_tree.end(error=f"{type(exc).__name__}: {exc}", metadata=failure_metadata)
                raise

            final_metadata = build_trace_metadata(final_state)
            run_tree.end(outputs=_trace_outputs(final_state), metadata=final_metadata)
    finally:
        flush_langsmith_client(client)
    return final_state


def flush_langsmith_client(client: Any) -> None:
    flush = getattr(client, "flush", None)
    if callable(flush):
        try:
            flush()
        except Exception:
            pass


def build_trace_metadata(state: Mapping[str, Any]) -> dict[str, Any]:
    config = _config_from_state(state)
    manifest = _artifact_manifest_from_state(state)
    feature_set = _feature_set_summary(state, manifest)
    artifact_paths = _artifact_paths(state)
    artifact_hashes = dict(manifest.get("artifacts", {})) if manifest else {}

    metadata: dict[str, Any] = {
        "run_id": state.get("run_id") or manifest.get("run_id"),
        "config_path": _path_value(state.get("config_path")),
        "config_hash": manifest.get("config_hash") or (sha256_json(config) if config else None),
        "sample_data": bool(state.get("sample_data")),
        "as_of_date": config.get("as_of_date"),
        "start_date": config.get("start_date"),
        "end_date": config.get("end_date"),
        "tickers": list(state.get("tickers") or config.get("tickers") or []),
        "horizons": list(state.get("horizons") or config.get("horizons") or []),
        "model_config": _model_config_summary(config),
        "feature_set": feature_set,
        "feature_matrix_hash": artifact_hashes.get("features.csv"),
        "output_dir": _path_value(state.get("output_dir")),
        "artifact_names": sorted(artifact_paths),
        "artifact_paths": artifact_paths,
        "artifact_hashes": artifact_hashes,
        "metrics": _metrics_summary(state),
        "point_in_time_validation": _validation_summary(state),
        "graph": {
            "status": state.get("status"),
            "completed_nodes": list(state.get("completed_nodes") or []),
            "failed_node": state.get("failed_node"),
            "error_type": state.get("error_type"),
            "error_message": state.get("error_message"),
        },
    }
    return _drop_none(metadata)


def _load_langsmith() -> Any:
    return importlib.import_module("langsmith")


def _trace_tags(extra_tags: list[str] | None) -> list[str]:
    seen: set[str] = set()
    tags: list[str] = []
    for tag in [*DEFAULT_TRACE_TAGS, *(extra_tags or [])]:
        if tag and tag not in seen:
            tags.append(tag)
            seen.add(tag)
    return tags


def _trace_inputs(state: Mapping[str, Any]) -> dict[str, Any]:
    return _drop_none(
        {
            "run_id": state.get("run_id"),
            "config_path": _path_value(state.get("config_path")),
            "output_dir": _path_value(state.get("output_dir")),
            "sample_data": bool(state.get("sample_data")),
        }
    )


def _trace_outputs(state: Mapping[str, Any]) -> dict[str, Any]:
    artifact_paths = _artifact_paths(state)
    return _drop_none(
        {
            "status": state.get("status"),
            "run_id": state.get("run_id"),
            "output_dir": _path_value(state.get("output_dir")),
            "artifact_count": len(artifact_paths),
            "failed_node": state.get("failed_node"),
            "error_type": state.get("error_type"),
        }
    )


def _config_from_state(state: Mapping[str, Any]) -> dict[str, Any]:
    config = state.get("config")
    if isinstance(config, dict):
        return config

    config_path = state.get("config_path")
    if config_path is None:
        return {}
    try:
        loaded = read_json(Path(config_path))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _artifact_manifest_from_state(state: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = _manifest_path(state)
    if manifest_path is None or not manifest_path.exists():
        return {}
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError):
        return {}
    return manifest if isinstance(manifest, dict) else {}


def _manifest_path(state: Mapping[str, Any]) -> Path | None:
    artifact_paths = state.get("artifact_paths")
    if isinstance(artifact_paths, dict) and artifact_paths.get("artifact_manifest.json"):
        return Path(str(artifact_paths["artifact_manifest.json"]))

    output_dir = state.get("output_dir")
    if output_dir is None:
        return None
    return Path(output_dir) / "artifact_manifest.json"


def _feature_set_summary(
    state: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_feature_set = manifest.get("feature_set")
    if isinstance(manifest_feature_set, dict) and manifest_feature_set:
        return _compact_feature_set(manifest_feature_set)

    metrics = state.get("all_metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("feature_set"), dict):
        return _compact_feature_set(metrics["feature_set"])

    selection = state.get("feature_selection")
    summary = getattr(selection, "summary", None)
    if callable(summary):
        return _compact_feature_set(summary())

    return {}


def _compact_feature_set(feature_set: Mapping[str, Any]) -> dict[str, Any]:
    return _drop_none(
        {
            "name": feature_set.get("name"),
            "hash": feature_set.get("hash"),
            "selected_feature_count": feature_set.get("selected_feature_count"),
            "selected_feature_columns": list(feature_set.get("selected_feature_columns") or []),
        }
    )


def _model_config_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    return _drop_none(
        {
            "logistic": config.get("logistic"),
            "ridge": config.get("ridge"),
            "train_window_rows": config.get("train_window_rows"),
            "min_train_rows": config.get("min_train_rows"),
            "prediction_stride": config.get("prediction_stride"),
            "prediction_stride_by_horizon": config.get("prediction_stride_by_horizon"),
            "max_backtest_predictions": config.get("max_backtest_predictions"),
            "long_threshold": config.get("long_threshold"),
            "short_threshold": config.get("short_threshold"),
            "transaction_cost_bps": config.get("transaction_cost_bps"),
        }
    )


def _artifact_paths(state: Mapping[str, Any]) -> dict[str, str]:
    raw_paths = state.get("artifact_paths")
    if not isinstance(raw_paths, dict):
        return {}

    paths: dict[str, str] = {}
    for name, value in sorted(raw_paths.items()):
        paths[str(name)] = _path_value(value)

    if not paths.get("artifact_manifest.json"):
        manifest_path = _manifest_path(state)
        if manifest_path is not None and manifest_path.exists():
            paths["artifact_manifest.json"] = _path_value(manifest_path)
    return paths


def _metrics_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    metrics = state.get("all_metrics")
    if not isinstance(metrics, dict):
        return {}

    horizons = metrics.get("horizons")
    if not isinstance(horizons, dict):
        return {}

    summary: dict[str, Any] = {}
    for horizon_name, horizon_metrics in horizons.items():
        if not isinstance(horizon_metrics, dict):
            continue
        summary[str(horizon_name)] = _drop_none(
            {
                "horizon": horizon_metrics.get("horizon"),
                "prediction_count": horizon_metrics.get("prediction_count"),
                "selected_feature_columns": list(
                    horizon_metrics.get("selected_feature_columns") or []
                ),
                "model": _model_metric_summary(horizon_metrics.get("model")),
            }
        )
    return summary


def _model_metric_summary(model_metrics: Any) -> dict[str, Any]:
    if not isinstance(model_metrics, dict):
        return {}

    summary: dict[str, Any] = {}
    for split_name in ("test", "validation", "all"):
        split_metrics = model_metrics.get(split_name)
        if not isinstance(split_metrics, dict):
            continue
        summary[split_name] = _drop_none(
            {
                "accuracy": split_metrics.get("accuracy"),
                "brier_score": split_metrics.get("brier_score"),
                "sharpe": split_metrics.get("sharpe"),
                "prediction_count": split_metrics.get("prediction_count"),
            }
        )
    return summary


def _validation_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    metrics = state.get("all_metrics")
    validation = {}
    if isinstance(metrics, dict):
        validation = metrics.get("point_in_time_validation") or {}
    if not isinstance(validation, dict):
        validation = {}

    leakage = {}
    if isinstance(validation, dict):
        leakage = validation.get("leakage") or {}

    if not leakage and isinstance(state.get("leakage_report"), dict):
        leakage_report = state["leakage_report"]
        leakage = {
            "status": leakage_report.get("status"),
            "error_count": leakage_report.get("error_count"),
        }

    return _drop_none(
        {
            "config_dates_status": _report_status(validation.get("config_dates")),
            "feature_rows_status": _report_status(validation.get("feature_rows")),
            "feature_set_status": _report_status(validation.get("feature_set")),
            "leakage_status": leakage.get("status") if isinstance(leakage, dict) else None,
            "leakage_error_count": (
                leakage.get("error_count") if isinstance(leakage, dict) else None
            ),
        }
    )


def _report_status(report: Any) -> Any:
    return report.get("status") if isinstance(report, dict) else None


def _path_value(value: Any) -> str:
    if value is None:
        return ""
    path = Path(value)
    try:
        return str(path.resolve().relative_to(project_root()))
    except (OSError, ValueError):
        return str(path)


def _drop_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None and value != ""}
