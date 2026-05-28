from __future__ import annotations

import math
from statistics import mean, pstdev

from .data import Bar
from .feature_registry import registered_feature_columns


def build_feature_rows(
    prices: dict[str, list[Bar]],
    *,
    primary_ticker: str,
    leveraged_ticker: str,
    comparison_ticker: str,
    horizons: list[int],
) -> list[dict[str, float | int | str]]:
    aligned_dates = sorted(set.intersection(*(set(bar.date for bar in bars) for bars in prices.values())))
    if not aligned_dates:
        raise ValueError("No overlapping dates across tickers")

    by_ticker = {
        ticker: {bar.date: bar for bar in bars}
        for ticker, bars in prices.items()
    }

    close = {
        ticker: [by_ticker[ticker][date].adj_close for date in aligned_dates]
        for ticker in prices
    }
    volume = {
        ticker: [by_ticker[ticker][date].volume for date in aligned_dates]
        for ticker in prices
    }

    rows: list[dict[str, float | int | str]] = []
    for idx, date in enumerate(aligned_dates):
        if idx < 21:
            continue

        row: dict[str, float | int | str] = {
            "date": date,
            "feature_available_date": date,
            "row_idx": idx,
            "primary_close": close[primary_ticker][idx],
            "leveraged_close": close[leveraged_ticker][idx],
        }

        for ticker in [primary_ticker, leveraged_ticker, comparison_ticker]:
            prefix = ticker.lower()
            row[f"{prefix}_ret_1d"] = pct_change(close[ticker], idx, 1)
            row[f"{prefix}_ret_5d"] = pct_change(close[ticker], idx, 5)
            row[f"{prefix}_ret_20d"] = pct_change(close[ticker], idx, 20)
            row[f"{prefix}_vol_20d"] = realized_vol(close[ticker], idx, 20)
            row[f"{prefix}_volume_z20"] = zscore(volume[ticker], idx, 20)

        row["soxx_smh_ret_spread_5d"] = float(row[f"{primary_ticker.lower()}_ret_5d"]) - float(
            row[f"{comparison_ticker.lower()}_ret_5d"]
        )
        row["soxl_leverage_realized_5d"] = safe_div(
            float(row[f"{leveraged_ticker.lower()}_ret_5d"]),
            float(row[f"{primary_ticker.lower()}_ret_5d"]),
        )
        row["soxx_drawdown_20d"] = rolling_drawdown(close[primary_ticker], idx, 20)

        for horizon in horizons:
            if idx + horizon < len(aligned_dates):
                target_end_date = aligned_dates[idx + horizon]
                future_return = close[primary_ticker][idx + horizon] / close[primary_ticker][idx] - 1.0
                row[f"target_return_{horizon}d"] = future_return
                row[f"target_direction_{horizon}d"] = 1 if future_return > 0 else 0
                row[f"target_end_date_{horizon}d"] = target_end_date
                row[f"target_available_date_{horizon}d"] = target_end_date
            else:
                row[f"target_return_{horizon}d"] = ""
                row[f"target_direction_{horizon}d"] = ""
                row[f"target_end_date_{horizon}d"] = ""
                row[f"target_available_date_{horizon}d"] = ""

        if not has_bad_feature_values(row):
            rows.append(row)

    return rows


def feature_columns(primary_ticker: str = "SOXX", leveraged_ticker: str = "SOXL", comparison_ticker: str = "SMH") -> list[str]:
    if (primary_ticker, leveraged_ticker, comparison_ticker) == ("SOXX", "SOXL", "SMH"):
        return registered_feature_columns()

    columns: list[str] = []
    for ticker in [primary_ticker, leveraged_ticker, comparison_ticker]:
        prefix = ticker.lower()
        columns.extend(
            [
                f"{prefix}_ret_1d",
                f"{prefix}_ret_5d",
                f"{prefix}_ret_20d",
                f"{prefix}_vol_20d",
                f"{prefix}_volume_z20",
            ]
        )
    columns.extend(["soxx_smh_ret_spread_5d", "soxl_leverage_realized_5d", "soxx_drawdown_20d"])
    return columns


def pct_change(values: list[float], idx: int, window: int) -> float:
    previous = values[idx - window]
    if previous == 0:
        return 0.0
    return values[idx] / previous - 1.0


def trailing_returns(values: list[float], idx: int, window: int) -> list[float]:
    returns: list[float] = []
    start = idx - window + 1
    for current in range(start, idx + 1):
        previous = values[current - 1]
        if previous:
            returns.append(values[current] / previous - 1.0)
    return returns


def realized_vol(values: list[float], idx: int, window: int) -> float:
    returns = trailing_returns(values, idx, window)
    if len(returns) < 2:
        return 0.0
    return pstdev(returns) * math.sqrt(252)


def rolling_drawdown(values: list[float], idx: int, window: int) -> float:
    start = max(0, idx - window + 1)
    window_values = values[start : idx + 1]
    peak = max(window_values)
    if peak == 0:
        return 0.0
    return values[idx] / peak - 1.0


def zscore(values: list[float], idx: int, window: int) -> float:
    start = idx - window + 1
    window_values = values[start : idx + 1]
    if len(window_values) < 2:
        return 0.0
    sigma = pstdev(window_values)
    if sigma == 0:
        return 0.0
    return (values[idx] - mean(window_values)) / sigma


def safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-8:
        return 0.0
    value = numerator / denominator
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value


def has_bad_feature_values(row: dict[str, float | int | str]) -> bool:
    for key, value in row.items():
        if key.startswith("target_") or key in {"date", "feature_available_date", "row_idx"}:
            continue
        if isinstance(value, (float, int)) and (math.isnan(value) or math.isinf(value)):
            return True
    return False
