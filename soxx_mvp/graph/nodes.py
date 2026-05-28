from __future__ import annotations

from collections.abc import Callable
from typing import Any

from soxx_mvp.pipeline import (
    NODE_LOAD_CONFIG,
    NODE_MATERIALIZE_FEATURES,
    NODE_PULL_MARKET_DATA,
    NODE_RUN_BACKTEST,
    NODE_RUN_FORECAST_MODELS,
    NODE_VALIDATE_POINT_IN_TIME_DATA,
    NODE_WRITE_ARTIFACTS,
    load_config_step,
    mark_node_completed,
    materialize_features_step,
    pull_market_data_step,
    run_backtest_step,
    run_forecast_models_step,
    validate_point_in_time_data_step,
    write_artifacts_step,
)

from .state import SoxxGraphState


def load_config_node(state: SoxxGraphState) -> SoxxGraphState:
    return _run_node(NODE_LOAD_CONFIG, load_config_step, state)


def pull_market_data_node(state: SoxxGraphState) -> SoxxGraphState:
    return _run_node(NODE_PULL_MARKET_DATA, pull_market_data_step, state)


def materialize_features_node(state: SoxxGraphState) -> SoxxGraphState:
    return _run_node(NODE_MATERIALIZE_FEATURES, materialize_features_step, state)


def validate_point_in_time_data_node(state: SoxxGraphState) -> SoxxGraphState:
    return _run_node(NODE_VALIDATE_POINT_IN_TIME_DATA, validate_point_in_time_data_step, state)


def run_forecast_models_node(state: SoxxGraphState) -> SoxxGraphState:
    return _run_node(NODE_RUN_FORECAST_MODELS, run_forecast_models_step, state)


def run_backtest_node(state: SoxxGraphState) -> SoxxGraphState:
    return _run_node(NODE_RUN_BACKTEST, run_backtest_step, state)


def write_artifacts_node(state: SoxxGraphState) -> SoxxGraphState:
    return _run_node(NODE_WRITE_ARTIFACTS, write_artifacts_step, state)


def _run_node(
    node_name: str,
    step: Callable[[dict[str, Any]], dict[str, Any]],
    state: SoxxGraphState,
) -> SoxxGraphState:
    try:
        next_state = step(dict(state))
        next_state = mark_node_completed(next_state, node_name)
        next_state["status"] = "completed" if node_name == NODE_WRITE_ARTIFACTS else "running"
        next_state["failed_node"] = ""
        next_state["error_type"] = ""
        next_state["error_message"] = ""
        return next_state
    except Exception as exc:
        return {
            **state,
            "status": "failed",
            "failed_node": node_name,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
