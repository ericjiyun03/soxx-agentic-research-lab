from __future__ import annotations

import csv
import json
import math
import random
import ssl
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .io_utils import ensure_dir


@dataclass(frozen=True)
class Bar:
    date: str
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: float


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def date_to_unix(value: str, *, end_of_day: bool = False) -> int:
    parsed = parse_date(value)
    dt = datetime(parsed.year, parsed.month, parsed.day, tzinfo=timezone.utc)
    if end_of_day:
        dt += timedelta(days=1)
    return int(dt.timestamp())


def fetch_yahoo_daily(
    ticker: str,
    start_date: str,
    end_date: str,
    timeout: int = 30,
    *,
    insecure_ssl: bool = False,
) -> list[Bar]:
    params = urlencode(
        {
            "period1": date_to_unix(start_date),
            "period2": date_to_unix(end_date, end_of_day=True),
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?{params}"
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; SOXX-SOXL-Agentic-Research-Lab/0.1)"
            )
        },
    )
    context = ssl._create_unverified_context() if insecure_ssl else None
    with urlopen(request, timeout=timeout, context=context) as response:
        payload = json.loads(response.read().decode("utf-8"))

    result = payload.get("chart", {}).get("result", [None])[0]
    if not result:
        error = payload.get("chart", {}).get("error")
        raise RuntimeError(f"Yahoo chart API returned no data for {ticker}: {error}")

    timestamps = result.get("timestamp") or []
    quote = result.get("indicators", {}).get("quote", [{}])[0]
    adjclose = result.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose") or []

    bars: list[Bar] = []
    for idx, ts in enumerate(timestamps):
        close = _safe_float(_value_at(quote.get("close"), idx))
        if close is None:
            continue
        adj = _safe_float(_value_at(adjclose, idx)) or close
        bars.append(
            Bar(
                date=datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat(),
                open=_safe_float(_value_at(quote.get("open"), idx)) or close,
                high=_safe_float(_value_at(quote.get("high"), idx)) or close,
                low=_safe_float(_value_at(quote.get("low"), idx)) or close,
                close=close,
                adj_close=adj,
                volume=_safe_float(_value_at(quote.get("volume"), idx)) or 0.0,
            )
        )
    return bars


def _value_at(values: list[float] | None, idx: int) -> float | None:
    if values is None or idx >= len(values):
        return None
    return values[idx]


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def write_bars_csv(path: Path, bars: list[Bar]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(bars[0]).keys()) if bars else [])
        writer.writeheader()
        for bar in bars:
            writer.writerow(asdict(bar))


def read_bars_csv(path: Path) -> list[Bar]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        Bar(
            date=row["date"],
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            adj_close=float(row["adj_close"]),
            volume=float(row["volume"]),
        )
        for row in rows
    ]


def load_or_fetch_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    cache_dir: Path,
    *,
    refresh: bool = False,
    insecure_ssl: bool = False,
) -> dict[str, list[Bar]]:
    ensure_dir(cache_dir)
    prices: dict[str, list[Bar]] = {}
    for ticker in tickers:
        path = cache_dir / f"{ticker}_{start_date}_{end_date}.csv"
        if path.exists() and not refresh:
            bars = read_bars_csv(path)
        else:
            bars = fetch_yahoo_daily(ticker, start_date, end_date, insecure_ssl=insecure_ssl)
            write_bars_csv(path, bars)
        if not bars:
            raise RuntimeError(f"No price bars available for {ticker}")
        prices[ticker] = bars
    return prices


def generate_sample_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    *,
    seed: int = 7,
) -> dict[str, list[Bar]]:
    rng = random.Random(seed)
    start = parse_date(start_date)
    end = parse_date(end_date)
    trading_dates: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            trading_dates.append(current)
        current += timedelta(days=1)

    base_returns: list[float] = []
    drift = 0.00025
    for idx, _ in enumerate(trading_dates):
        cycle = 0.004 * math.sin(idx / 21.0)
        shock = rng.gauss(0.0, 0.014)
        base_returns.append(drift + cycle + shock)

    prices: dict[str, list[Bar]] = {}
    for ticker in tickers:
        if ticker == "SOXL":
            multiplier = 3.0
            start_price = 25.0
            noise_scale = 0.01
        elif ticker == "SMH":
            multiplier = 0.95
            start_price = 120.0
            noise_scale = 0.003
        else:
            multiplier = 1.0
            start_price = 100.0
            noise_scale = 0.004

        close = start_price
        bars: list[Bar] = []
        for idx, day in enumerate(trading_dates):
            daily_return = multiplier * base_returns[idx] + rng.gauss(0.0, noise_scale)
            close = max(1.0, close * (1.0 + daily_return))
            open_price = close / (1.0 + rng.gauss(0.0, 0.004))
            high = max(open_price, close) * (1.0 + abs(rng.gauss(0.0, 0.006)))
            low = min(open_price, close) * (1.0 - abs(rng.gauss(0.0, 0.006)))
            bars.append(
                Bar(
                    date=day.isoformat(),
                    open=open_price,
                    high=high,
                    low=low,
                    close=close,
                    adj_close=close,
                    volume=max(100000.0, rng.gauss(5000000.0, 750000.0)),
                )
            )
        prices[ticker] = bars
    return prices
