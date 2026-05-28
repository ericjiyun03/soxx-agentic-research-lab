from __future__ import annotations

import math


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for actual, pred in zip(y_true, y_pred) if actual == pred) / len(y_true)


def precision(y_true: list[int], y_pred: list[int]) -> float:
    tp = sum(1 for actual, pred in zip(y_true, y_pred) if actual == 1 and pred == 1)
    fp = sum(1 for actual, pred in zip(y_true, y_pred) if actual == 0 and pred == 1)
    return tp / (tp + fp) if tp + fp else 0.0


def recall(y_true: list[int], y_pred: list[int]) -> float:
    tp = sum(1 for actual, pred in zip(y_true, y_pred) if actual == 1 and pred == 1)
    fn = sum(1 for actual, pred in zip(y_true, y_pred) if actual == 1 and pred == 0)
    return tp / (tp + fn) if tp + fn else 0.0


def brier_score(y_true: list[int], probs: list[float]) -> float:
    return mean([(prob - actual) ** 2 for actual, prob in zip(y_true, probs)])


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    return math.sqrt(mean([(pred - actual) ** 2 for actual, pred in zip(y_true, y_pred)]))


def mae(y_true: list[float], y_pred: list[float]) -> float:
    return mean([abs(pred - actual) for actual, pred in zip(y_true, y_pred)])


def pearson(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2 or len(y_values) < 2:
        return 0.0
    x_avg = mean(x_values)
    y_avg = mean(y_values)
    numerator = sum((x - x_avg) * (y - y_avg) for x, y in zip(x_values, y_values))
    x_var = sum((x - x_avg) ** 2 for x in x_values)
    y_var = sum((y - y_avg) ** 2 for y in y_values)
    denominator = math.sqrt(x_var * y_var)
    return numerator / denominator if denominator else 0.0


def annualized_sharpe(returns: list[float], periods_per_year: float = 252.0) -> float:
    if len(returns) < 2:
        return 0.0
    avg = mean(returns)
    variance = sum((value - avg) ** 2 for value in returns) / len(returns)
    sigma = math.sqrt(variance)
    if sigma == 0:
        return 0.0
    return avg / sigma * math.sqrt(periods_per_year)


def cumulative_return(returns: list[float]) -> float:
    equity = 1.0
    for ret in returns:
        equity *= 1.0 + ret
    return equity - 1.0


def max_drawdown(returns: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for ret in returns:
        equity *= 1.0 + ret
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0
        worst = min(worst, drawdown)
    return worst
