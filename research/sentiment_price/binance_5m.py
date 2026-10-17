"""binance_5m.py - the E0.2a Binance 5m loader as a reusable function (logic unchanged)."""
import json
import time
from pathlib import Path

import pandas as pd
import requests

STEP_MS = 5 * 60 * 1000
DEFAULT_CACHE = Path("/media/ata/Data/personal/binbot/research/sentiment_price/data/btcusdt_5m_batches")


def load_binance_5m(start: pd.Timestamp, end: pd.Timestamp, cache_dir: Path = DEFAULT_CACHE,
                    symbol: str = "BTCUSDT") -> pd.DataFrame:
    """5m klines, UTC candle-OPEN index, [start, end). Raises on any gap/duplicate."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    start_ms = int(start.timestamp() * 1000); end_ms = int(end.timestamp() * 1000)
    all_candles = []
    cursor = start_ms
    with requests.Session() as session:
        while cursor < end_ms:
            batch_end = min(cursor + 1000 * STEP_MS, end_ms)
            cache_file = cache_dir / f"{cursor}_{batch_end}.json"
            if cache_file.exists():
                batch = json.loads(cache_file.read_text())
            else:
                response = session.get(
                    "https://api.binance.com/api/v3/klines",
                    params={"symbol": symbol, "interval": "5m", "timeZone": "0",
                            "startTime": cursor, "endTime": batch_end - 1, "limit": 1000},
                    timeout=30,
                )
                response.raise_for_status()
                batch = response.json()
            if not isinstance(batch, list) or not batch:
                raise ValueError(f"Unexpected response at {cursor}: {batch}")
            if [int(r[0]) for r in batch] != list(range(cursor, batch_end, STEP_MS)):
                raise ValueError(f"Missing, duplicate or unexpected candles in batch {cursor}")
            if not cache_file.exists():
                tmp = cache_file.with_suffix(".tmp")
                tmp.write_text(json.dumps(batch)); tmp.replace(cache_file)
                time.sleep(0.2)
            all_candles.extend(batch)
            cursor = batch_end
            print(f"\rLoaded {len(all_candles):,} candles", end="", flush=True)
    print()
    columns = ["open_time", "open", "high", "low", "close", "volume", "close_time",
               "quote_volume", "trade_count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
    bars = pd.DataFrame(all_candles, columns=columns)
    bars["open_time"] = pd.to_datetime(bars["open_time"], unit="ms", utc=True)
    bars["close_time"] = pd.to_datetime(bars["close_time"], unit="ms", utc=True)
    numeric = ["open", "high", "low", "close", "volume", "quote_volume", "trade_count",
               "taker_buy_volume", "taker_buy_quote_volume"]
    bars[numeric] = bars[numeric].apply(pd.to_numeric, errors="raise")
    bars = bars.drop(columns="ignore").set_index("open_time").sort_index()
    expected = pd.date_range(start, end - pd.Timedelta(minutes=5), freq="5min")
    assert bars.index.is_unique and bars.index.equals(expected), "Unexpected candle coverage"
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1)).all()
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1)).all()
    return bars
