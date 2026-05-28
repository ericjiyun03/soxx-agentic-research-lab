from __future__ import annotations

from typing import Any

import pandas as pd

from .baselines import BASELINE_NAMES, baseline_positions
from .features import feature_columns
from .metrics import (
    accuracy,
    annualized_sharpe,
    brier_score,
    cumulative_return,
    mae,
    max_drawdown,
    mean,
    pearson,
    precision,
    recall,
    rmse,
)
from .models import fit_logistic, fit_ridge


def run_walkforward(
    rows: list[dict[str, float | int | str]],
    *,
    horizon: int,
    train_window_rows: int,
    min_train_rows: int,
    prediction_stride: int,
    max_backtest_predictions: int | None,
    long_threshold: float,
    short_threshold: float,
    transaction_cost_bps: float,
    logistic_config: dict[str, float | int],
    ridge_config: dict[str, float | int],
    validation_start_date: str | None = None,
    validation_end_date: str | None = None,
    test_start_date: str | None = None,
    test_end_date: str | None = None,
    cost_sensitivity_bps: list[float] | None = None,
    model_feature_columns: list[str] | None = None,
    primary_ticker: str = "SOXX",
) -> dict[str, Any]:
    columns = list(model_feature_columns) if model_feature_columns else feature_columns(primary_ticker=primary_ticker)
    target_return_col = f"target_return_{horizon}d"
    target_direction_col = f"target_direction_{horizon}d"
    df = _feature_frame(rows, columns + [target_return_col, target_direction_col])

    candidate_indices = []
    for idx, row in df.iterrows():
        split = _split_for_date(
            str(row["date"]),
            validation_start_date=validation_start_date,
            validation_end_date=validation_end_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date,
        )
        if split is None:
            continue
        if pd.isna(row[target_return_col]):
            continue
        if idx < min_train_rows + horizon:
            continue
        if idx % max(1, prediction_stride) != 0:
            continue
        candidate_indices.append(int(idx))

    if max_backtest_predictions is not None and max_backtest_predictions > 0:
        candidate_indices = candidate_indices[-max_backtest_predictions:]

    model_gross_rows: list[dict[str, Any]] = []
    baseline_gross_rows: list[dict[str, Any]] = []

    for idx in candidate_indices:
        row = df.iloc[idx]
        train_end = idx - horizon
        if train_end < min_train_rows:
            continue
        train_start = max(0, train_end - train_window_rows + 1)
        train_rows = df.iloc[train_start : train_end + 1].dropna(
            subset=columns + [target_return_col, target_direction_col]
        )
        if len(train_rows) < min_train_rows:
            continue

        x_train = train_rows[columns].astype(float).to_numpy().tolist()
        y_direction = train_rows[target_direction_col].astype(int).to_list()
        y_return = train_rows[target_return_col].astype(float).to_list()
        x_current = row[columns].astype(float).to_list()

        logistic = fit_logistic(x_train, y_direction, **logistic_config)
        ridge = fit_ridge(x_train, y_return, **ridge_config)

        prob_up = logistic.predict_proba_one(x_current)
        pred_direction = 1 if prob_up >= 0.5 else 0
        pred_return = ridge.predict_one(x_current)
        actual_return = float(row[target_return_col])
        actual_direction = int(row[target_direction_col])
        position = _model_position(prob_up, long_threshold=long_threshold, short_threshold=short_threshold)
        split = _split_for_date(
            str(row["date"]),
            validation_start_date=validation_start_date,
            validation_end_date=validation_end_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date,
        )
        if split is None:
            continue

        base_payload = {
            "date": row["date"],
            "horizon": horizon,
            "split": split,
            "train_start_date": train_rows.iloc[0]["date"],
            "train_end_date": train_rows.iloc[-1]["date"],
            "n_train": int(len(train_rows)),
            "actual_return": actual_return,
            "actual_direction": actual_direction,
            "buy_hold_return": actual_return,
        }
        model_gross_rows.append(
            {
                **base_payload,
                "strategy": "model",
                "prob_up": prob_up,
                "pred_direction": pred_direction,
                "pred_return": pred_return,
                "position": position,
            }
        )

        for baseline_name, baseline_position in baseline_positions(
            row,
            train_rows,
            primary_ticker=primary_ticker,
        ).items():
            baseline_gross_rows.append(
                {
                    **base_payload,
                    "baseline": baseline_name,
                    "strategy": baseline_name,
                    "pred_direction": 1 if baseline_position > 0 else 0,
                    "pred_return": baseline_position * actual_return,
                    "prob_up": 1.0 if baseline_position > 0 else 0.0,
                    "position": baseline_position,
                }
            )

    model_predictions = apply_transaction_costs(
        model_gross_rows,
        transaction_cost_bps=transaction_cost_bps,
        group_keys=["split"],
    )
    baseline_predictions = apply_transaction_costs(
        baseline_gross_rows,
        transaction_cost_bps=transaction_cost_bps,
        group_keys=["baseline", "split"],
    )
    strategy_comparison = score_strategy_comparison(
        model_predictions,
        baseline_predictions,
        horizon=horizon,
        transaction_cost_bps=transaction_cost_bps,
    )
    cost_sensitivity = run_cost_sensitivity(
        model_gross_rows,
        baseline_gross_rows,
        horizon=horizon,
        cost_sensitivity_bps=cost_sensitivity_bps or [transaction_cost_bps],
    )

    return {
        "model_predictions": model_predictions,
        "baseline_predictions": baseline_predictions,
        "strategy_comparison": strategy_comparison,
        "cost_sensitivity": cost_sensitivity,
        "model_feature_columns": columns,
        "metrics": {
            "horizon": horizon,
            "prediction_count": len(model_predictions),
            "selected_feature_count": len(columns),
            "selected_feature_columns": columns,
            "baseline_names": BASELINE_NAMES,
            "model": score_model_predictions(model_predictions, horizon=horizon),
            "strategy_comparison": strategy_comparison,
            "cost_sensitivity": cost_sensitivity,
        },
    }


def apply_transaction_costs(
    rows: list[dict[str, Any]],
    *,
    transaction_cost_bps: float,
    group_keys: list[str],
) -> list[dict[str, Any]]:
    cost_rate = transaction_cost_bps / 10000.0
    previous_positions: dict[tuple[Any, ...], int] = {}
    adjusted_rows: list[dict[str, Any]] = []
    for row in rows:
        group = tuple(row[key] for key in group_keys)
        previous_position = previous_positions.get(group, 0)
        position = int(row["position"])
        turnover = abs(position - previous_position)
        trading_cost = cost_rate * turnover
        gross_return = position * float(row["actual_return"])
        previous_positions[group] = position

        adjusted_rows.append(
            {
                **row,
                "transaction_cost_bps": float(transaction_cost_bps),
                "turnover": turnover,
                "trading_cost": trading_cost,
                "gross_strategy_return": gross_return,
                "strategy_return": gross_return - trading_cost,
            }
        )
    return adjusted_rows


def run_cost_sensitivity(
    model_gross_rows: list[dict[str, Any]],
    baseline_gross_rows: list[dict[str, Any]],
    *,
    horizon: int,
    cost_sensitivity_bps: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cost_bps in cost_sensitivity_bps:
        model_predictions = apply_transaction_costs(
            model_gross_rows,
            transaction_cost_bps=float(cost_bps),
            group_keys=["split"],
        )
        baseline_predictions = apply_transaction_costs(
            baseline_gross_rows,
            transaction_cost_bps=float(cost_bps),
            group_keys=["baseline", "split"],
        )
        rows.extend(
            score_strategy_comparison(
                model_predictions,
                baseline_predictions,
                horizon=horizon,
                transaction_cost_bps=float(cost_bps),
            )
        )
    return rows


def score_model_predictions(
    predictions: list[dict[str, Any]],
    *,
    horizon: int,
) -> dict[str, dict[str, float | int]]:
    return {
        split: _score_model_subset(_split_subset(predictions, split), horizon=horizon)
        for split in _ordered_splits(predictions)
    }


def score_strategy_comparison(
    model_predictions: list[dict[str, Any]],
    baseline_predictions: list[dict[str, Any]],
    *,
    horizon: int,
    transaction_cost_bps: float,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    strategies: list[tuple[str, list[dict[str, Any]]]] = [("model", model_predictions)]
    for baseline_name in BASELINE_NAMES:
        strategies.append(
            (
                baseline_name,
                [row for row in baseline_predictions if row.get("baseline") == baseline_name],
            )
        )

    for strategy_name, strategy_rows in strategies:
        for split in _ordered_splits(strategy_rows):
            scored = _score_strategy_subset(_split_subset(strategy_rows, split), horizon=horizon)
            rows.append(
                {
                    "horizon": horizon,
                    "split": split,
                    "strategy": strategy_name,
                    "transaction_cost_bps": float(transaction_cost_bps),
                    **scored,
                }
            )
    return rows


def latest_forecast(
    rows: list[dict[str, float | int | str]],
    *,
    horizon: int,
    train_window_rows: int,
    min_train_rows: int,
    logistic_config: dict[str, float | int],
    ridge_config: dict[str, float | int],
    model_feature_columns: list[str] | None = None,
) -> dict[str, float | int | str]:
    columns = list(model_feature_columns) if model_feature_columns else feature_columns()
    target_return_col = f"target_return_{horizon}d"
    target_direction_col = f"target_direction_{horizon}d"
    df = _feature_frame(rows, columns + [target_return_col, target_direction_col])
    idx = len(df) - 1
    train_end = idx - horizon
    if train_end < min_train_rows:
        raise ValueError(f"Not enough rows for latest forecast at horizon {horizon}")
    train_start = max(0, train_end - train_window_rows + 1)
    train_rows = df.iloc[train_start : train_end + 1].dropna(
        subset=columns + [target_return_col, target_direction_col]
    )
    if len(train_rows) < min_train_rows:
        raise ValueError(f"Not enough labeled training rows for latest forecast at horizon {horizon}")

    x_train = train_rows[columns].astype(float).to_numpy().tolist()
    y_direction = train_rows[target_direction_col].astype(int).to_list()
    y_return = train_rows[target_return_col].astype(float).to_list()
    x_current = df.iloc[-1][columns].astype(float).to_list()

    logistic = fit_logistic(x_train, y_direction, **logistic_config)
    ridge = fit_ridge(x_train, y_return, **ridge_config)

    prob_up = logistic.predict_proba_one(x_current)
    pred_return = ridge.predict_one(x_current)
    return {
        "as_of_date": df.iloc[-1]["date"],
        "horizon": horizon,
        "train_start_date": train_rows.iloc[0]["date"],
        "train_end_date": train_rows.iloc[-1]["date"],
        "n_train": int(len(train_rows)),
        "prob_up": prob_up,
        "pred_direction": 1 if prob_up >= 0.5 else 0,
        "pred_return": pred_return,
        "naive_soxl_expected_return": 3.0 * pred_return,
    }


def _feature_frame(rows: list[dict[str, float | int | str]], numeric_columns: list[str]) -> pd.DataFrame:
    if not rows:
        raise ValueError("No feature rows generated")
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    for column in numeric_columns:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _model_position(prob_up: float, *, long_threshold: float, short_threshold: float) -> int:
    if prob_up >= long_threshold:
        return 1
    if prob_up <= short_threshold:
        return -1
    return 0


def _split_for_date(
    date_value: str,
    *,
    validation_start_date: str | None,
    validation_end_date: str | None,
    test_start_date: str | None,
    test_end_date: str | None,
) -> str | None:
    has_split_config = any(
        [validation_start_date, validation_end_date, test_start_date, test_end_date]
    )
    if not has_split_config:
        return "all"
    if validation_start_date and validation_end_date and validation_start_date <= date_value <= validation_end_date:
        return "validation"
    if test_start_date and test_end_date and test_start_date <= date_value <= test_end_date:
        return "test"
    return None


def _ordered_splits(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    splits = ["all"]
    for split in ["validation", "test"]:
        if any(row.get("split") == split for row in rows):
            splits.append(split)
    if len(splits) == 1 and any(row.get("split") == "all" for row in rows):
        return ["all"]
    return splits


def _split_subset(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split == "all":
        return list(rows)
    return [row for row in rows if row.get("split") == split]


def _score_model_subset(
    rows: list[dict[str, Any]],
    *,
    horizon: int,
) -> dict[str, float | int]:
    if not rows:
        return _empty_score()
    y_true = [int(row["actual_direction"]) for row in rows]
    y_pred = [int(row["pred_direction"]) for row in rows]
    probs = [float(row["prob_up"]) for row in rows]
    actual_returns = [float(row["actual_return"]) for row in rows]
    predicted_returns = [float(row["pred_return"]) for row in rows]
    return {
        **_score_strategy_subset(rows, horizon=horizon),
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "brier_score": brier_score(y_true, probs),
        "rmse_return": rmse(actual_returns, predicted_returns),
        "mae_return": mae(actual_returns, predicted_returns),
        "return_ic": pearson(actual_returns, predicted_returns),
        "mean_actual_return": mean(actual_returns),
        "mean_predicted_return": mean(predicted_returns),
    }


def _score_strategy_subset(
    rows: list[dict[str, Any]],
    *,
    horizon: int,
) -> dict[str, float | int]:
    if not rows:
        return _empty_score()
    y_true = [int(row["actual_direction"]) for row in rows]
    y_pred = [int(row["pred_direction"]) for row in rows]
    strategy_returns = [float(row["strategy_return"]) for row in rows]
    gross_returns = [float(row["gross_strategy_return"]) for row in rows]
    positions = [int(row["position"]) for row in rows]
    turnovers = [float(row["turnover"]) for row in rows]
    trading_costs = [float(row["trading_cost"]) for row in rows]
    active_rows = [row for row in rows if int(row["position"]) != 0]
    correct_active = [
        row
        for row in active_rows
        if (float(row["actual_return"]) > 0 and int(row["position"]) > 0)
        or (float(row["actual_return"]) < 0 and int(row["position"]) < 0)
    ]
    periods_per_year = 252.0 / max(1, horizon)

    return {
        "prediction_count": len(rows),
        "accuracy": accuracy(y_true, y_pred),
        "hit_rate": len(correct_active) / len(active_rows) if active_rows else 0.0,
        "coverage": len(active_rows) / len(rows) if rows else 0.0,
        "mean_strategy_return": mean(strategy_returns),
        "mean_gross_strategy_return": mean(gross_returns),
        "cumulative_return": cumulative_return(strategy_returns),
        "sharpe": annualized_sharpe(strategy_returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(strategy_returns),
        "average_abs_position": mean([abs(position) for position in positions]),
        "average_turnover": mean(turnovers),
        "total_trading_cost": sum(trading_costs),
    }


def _empty_score() -> dict[str, float | int]:
    return {
        "prediction_count": 0,
        "accuracy": 0.0,
        "hit_rate": 0.0,
        "coverage": 0.0,
        "mean_strategy_return": 0.0,
        "mean_gross_strategy_return": 0.0,
        "cumulative_return": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "average_abs_position": 0.0,
        "average_turnover": 0.0,
        "total_trading_cost": 0.0,
    }
