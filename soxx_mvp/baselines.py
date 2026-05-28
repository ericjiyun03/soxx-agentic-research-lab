from __future__ import annotations

import pandas as pd


BASELINE_NAMES = [
    "buy_hold",
    "momentum",
    "mean_reversion",
    "volatility_regime_momentum",
]


def baseline_positions(
    row: pd.Series,
    train_rows: pd.DataFrame,
    *,
    primary_ticker: str = "SOXX",
) -> dict[str, int]:
    prefix = primary_ticker.lower()
    momentum_position = _sign(float(row[f"{prefix}_ret_5d"]))
    vol_column = f"{prefix}_vol_20d"
    volatility_median = float(train_rows[vol_column].median()) if vol_column in train_rows else 0.0
    current_volatility = float(row[vol_column]) if vol_column in row else 0.0

    return {
        "buy_hold": 1,
        "momentum": momentum_position,
        "mean_reversion": -momentum_position,
        "volatility_regime_momentum": momentum_position if current_volatility <= volatility_median else 0,
    }


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0
