"""Leakage-safe data construction primitives for the corrected BTC baseline.

There is deliberately no model fitting or exchange integration in this file.
Targets are constructed separately from features and are never passed into
feature generation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


BAR = pd.Timedelta(minutes=1)
HORIZON_MINUTES = 10


@dataclass(frozen=True)
class Partition:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp  # exclusive


PARTITIONS = (
    Partition("train", pd.Timestamp("2025-10-10", tz="UTC"), pd.Timestamp("2026-05-15", tz="UTC")),
    Partition("validation", pd.Timestamp("2026-05-15", tz="UTC"), pd.Timestamp("2026-07-27", tz="UTC")),
    Partition("test", pd.Timestamp("2026-07-27", tz="UTC"), pd.Timestamp("2026-10-08", tz="UTC")),
)


def validate_one_minute_bars(bars: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError("Bars require a timezone-aware DatetimeIndex")
    if not bars.index.is_monotonic_increasing or not bars.index.is_unique:
        raise ValueError("Bar timestamps must be strictly increasing and unique")


def make_causal_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Fixed, modest feature set known at the close of timestamp t's bar."""
    validate_one_minute_bars(bars)
    close = bars["close"]
    out = pd.DataFrame(index=bars.index)
    out["return_1m"] = close.pct_change(1)
    out["return_5m"] = close.pct_change(5)
    out["return_10m"] = close.pct_change(10)
    out["range_fraction"] = (bars["high"] - bars["low"]) / close
    out["body_fraction"] = (close - bars["open"]) / bars["open"]
    out["volume_log1p"] = np.log1p(bars["volume"])
    out["volume_ratio_20"] = bars["volume"] / bars["volume"].rolling(20).mean()
    out["volatility_20"] = out["return_1m"].rolling(20).std()
    return out


def make_forward_target(bars: pd.DataFrame, horizon_minutes: int = HORIZON_MINUTES) -> pd.DataFrame:
    """Return executable entry/exit prices and decimal future return.

    A decision after bar t closes enters at open[t+1] and exits exactly ten
    minutes later at open[t+11]. Timestamp checks invalidate labels over gaps.
    """
    validate_one_minute_bars(bars)
    if horizon_minutes != HORIZON_MINUTES:
        raise ValueError("The rebuild contract freezes the horizon at 10 minutes")
    index_series = pd.Series(bars.index, index=bars.index)
    entry_time = index_series.shift(-1)
    exit_time = index_series.shift(-(1 + horizon_minutes))
    entry_open = bars["open"].shift(-1)
    exit_open = bars["open"].shift(-(1 + horizon_minutes))
    valid = (entry_time == index_series + BAR) & (
        exit_time == index_series + BAR * (1 + horizon_minutes)
    )
    target = pd.DataFrame(
        {"entry_time": entry_time, "entry_open": entry_open,
         "exit_time": exit_time, "exit_open": exit_open},
        index=bars.index,
    )
    target["future_return_10m"] = exit_open / entry_open - 1.0
    return target.where(valid)


def partition_mask(targets: pd.DataFrame, partition: Partition) -> pd.Series:
    """Keep decisions and their complete forward labels inside one partition."""
    decision_in = (targets.index >= partition.start) & (targets.index < partition.end)
    label_in = (
        (targets["entry_time"] >= partition.start)
        & (targets["exit_time"] < partition.end)
    )
    return pd.Series(decision_in & label_in, index=targets.index).fillna(False)


def legacy_target_from_roc(roc_10_percent: pd.Series) -> pd.Series:
    """Exact algebraic reconstruction of linear.ipynb's legacy target."""
    return -roc_10_percent / (1.0 + roc_10_percent / 100.0)
