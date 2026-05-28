from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from soxx_mvp.pipeline import (
    NODE_LOAD_CONFIG,
    NODE_MATERIALIZE_FEATURES,
    NODE_PULL_MARKET_DATA,
    NODE_RUN_BACKTEST,
    NODE_RUN_FORECAST_MODELS,
    NODE_VALIDATE_POINT_IN_TIME_DATA,
    NODE_WRITE_ARTIFACTS,
)

from .nodes import (
    load_config_node,
    materialize_features_node,
    pull_market_data_node,
    run_backtest_node,
    run_forecast_models_node,
    validate_point_in_time_data_node,
    write_artifacts_node,
)
from .routing import continue_or_stop
from .state import SoxxGraphState


def build_soxx_graph() -> Any:
    builder = StateGraph(SoxxGraphState)
    builder.add_node(NODE_LOAD_CONFIG, load_config_node)
    builder.add_node(NODE_PULL_MARKET_DATA, pull_market_data_node)
    builder.add_node(NODE_MATERIALIZE_FEATURES, materialize_features_node)
    builder.add_node(NODE_VALIDATE_POINT_IN_TIME_DATA, validate_point_in_time_data_node)
    builder.add_node(NODE_RUN_FORECAST_MODELS, run_forecast_models_node)
    builder.add_node(NODE_RUN_BACKTEST, run_backtest_node)
    builder.add_node(NODE_WRITE_ARTIFACTS, write_artifacts_node)

    builder.add_edge(START, NODE_LOAD_CONFIG)
    _add_guarded_edge(builder, NODE_LOAD_CONFIG, NODE_PULL_MARKET_DATA)
    _add_guarded_edge(builder, NODE_PULL_MARKET_DATA, NODE_MATERIALIZE_FEATURES)
    _add_guarded_edge(builder, NODE_MATERIALIZE_FEATURES, NODE_VALIDATE_POINT_IN_TIME_DATA)
    _add_guarded_edge(builder, NODE_VALIDATE_POINT_IN_TIME_DATA, NODE_RUN_FORECAST_MODELS)
    _add_guarded_edge(builder, NODE_RUN_FORECAST_MODELS, NODE_RUN_BACKTEST)
    _add_guarded_edge(builder, NODE_RUN_BACKTEST, NODE_WRITE_ARTIFACTS)
    builder.add_conditional_edges(
        NODE_WRITE_ARTIFACTS,
        continue_or_stop,
        {"continue": END, "stop": END},
    )
    return builder.compile()


def invoke_soxx_graph(
    initial_state: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> SoxxGraphState:
    graph = build_soxx_graph()
    return graph.invoke(initial_state, config=config)


def _add_guarded_edge(builder: StateGraph, source: str, destination: str) -> None:
    builder.add_conditional_edges(
        source,
        continue_or_stop,
        {"continue": destination, "stop": END},
    )
