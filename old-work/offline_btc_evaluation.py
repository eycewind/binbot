#!/usr/bin/env python3
"""Offline-only reproduction/evaluation of the legacy BTC 1-minute RF model.

This module intentionally does not import BinanceClient, bot scripts, or
BatchFeatures.  The feature formulas below are transcribed from cell 11 of
old-work/bot.ipynb because the current BatchFeatures.py has incompatible names
and formulas.  Orders are simulated at the next bar's open after a signal is
formed at the current bar's close.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURES = [
    "open", "high", "low", "volume", "sma_10", "sma_50", "ema_10",
    "rsi_14", "macd", "macd_signal", "macd_hist", "bollinger_upper",
    "bollinger_lower", "atr_14", "volume_ma_20", "volume_ratio", "roc_10",
    "close_lag_1", "close_lag_3", "macd_lag_1", "candle_body", "upper_wick",
    "lower_wick",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_bars(db_path: Path) -> pd.DataFrame:
    # Read-only URI prevents accidental database mutation.
    uri = f"file:{db_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        df = pd.read_sql_query(
            "SELECT timestamp, open, high, low, close, volume "
            "FROM candlesticks WHERE pair = 'BTCUSDT' ORDER BY timestamp",
            connection,
        )
    if df.empty:
        raise ValueError("No BTCUSDT rows in the selected database")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp")


def make_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Exact formulas and order from bot.ipynb cell 11."""
    df = bars.copy()
    df["sma_10"] = df["close"].rolling(window=10).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    sma_20 = df["close"].rolling(window=20).mean()
    std_20 = df["close"].rolling(window=20).std()
    df["bollinger_upper"] = sma_20 + 2 * std_20
    df["bollinger_lower"] = sma_20 - 2 * std_20

    high_low = df["high"] - df["low"]
    high_prev_close = (df["high"] - df["close"].shift()).abs()
    low_prev_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(window=14).mean()
    df["volume_ma_20"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma_20"]
    df["roc_10"] = (df["close"] - df["close"].shift(10)) / df["close"].shift(10) * 100
    df["close_lag_1"] = df["close"].shift(1)
    df["close_lag_3"] = df["close"].shift(3)
    # This apparent notebook bug is preserved exactly.
    df["macd_lag_1"] = df["close"].shift(1)
    df["candle_body"] = df["close"] - df["open"]
    df["upper_wick"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower_wick"] = df[["close", "open"]].min(axis=1) - df["low"]
    return df.dropna(subset=FEATURES)


def build_round_trips(frame: pd.DataFrame, buy_threshold: float, sell_threshold: float) -> list[dict]:
    """Generate fixed round trips; signals at close fill at the next open."""
    desired = np.where(
        frame["prediction"] > buy_threshold,
        1,
        np.where(frame["prediction"] < sell_threshold, -1, 0),
    )
    trades: list[dict] = []
    entry = None
    # Last signal has no following bar at which it can fill.
    for i, signal in enumerate(desired[:-1]):
        fill_time = frame.index[i + 1]
        fill_price = float(frame["open"].iloc[i + 1])
        if signal == 1 and entry is None:
            entry = (fill_time, fill_price, frame.index[i], float(frame["prediction"].iloc[i]))
        elif signal == -1 and entry is not None:
            trades.append({
                "entry_time": entry[0], "entry_price": entry[1],
                "entry_signal_time": entry[2], "entry_prediction": entry[3],
                "exit_time": fill_time, "exit_price": fill_price,
                "exit_signal_time": frame.index[i], "exit_prediction": float(frame["prediction"].iloc[i]),
                "forced_exit": False,
            })
            entry = None
    if entry is not None:
        trades.append({
            "entry_time": entry[0], "entry_price": entry[1],
            "entry_signal_time": entry[2], "entry_prediction": entry[3],
            "exit_time": frame.index[-1], "exit_price": float(frame["close"].iloc[-1]),
            "exit_signal_time": pd.NaT, "exit_prediction": np.nan, "forced_exit": True,
        })
    return trades


def price_trades(trades: list[dict], fee_rate: float, slippage_rate: float) -> pd.DataFrame:
    rows = []
    for trade in trades:
        buy = trade["entry_price"] * (1 + slippage_rate)
        sell = trade["exit_price"] * (1 - slippage_rate)
        gross_return = trade["exit_price"] / trade["entry_price"] - 1
        # Entry fee is added to purchase notional; exit fee is deducted from
        # sale notional. This is the exact all-cash round-trip multiplier.
        net_return = (sell / buy) * (1 - fee_rate) / (1 + fee_rate) - 1
        rows.append({**trade, "gross_return": gross_return, "net_return": net_return})
    return pd.DataFrame(rows)


def compound(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1) if len(returns) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("best_model_10candles_reduced_1min.joblib"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--buy-threshold", type=float, default=0.09)
    parser.add_argument("--sell-threshold", type=float, default=-0.14)
    parser.add_argument("--fee-bps-per-side", type=float, default=10.0)
    parser.add_argument("--slippage-bps-per-side", type=float, default=0.0)
    args = parser.parse_args()

    bars = load_bars(args.db)
    median_seconds = float(bars.index.to_series().diff().dt.total_seconds().median())
    features = make_features(bars)
    model = joblib.load(args.model)
    artifact_features = list(model.feature_names_in_)
    if artifact_features != FEATURES:
        raise ValueError(f"Artifact feature order mismatch: {artifact_features!r}")
    evaluated = features.copy()
    evaluated["prediction"] = model.predict(evaluated[FEATURES])
    # Diagnostic target documented by linear.ipynb; it is backward-looking.
    evaluated["notebook_target"] = (
        evaluated["close"].shift(10) - evaluated["close"]
    ) / evaluated["close"] * 100

    trades = build_round_trips(evaluated, args.buy_threshold, args.sell_threshold)
    trade_df = price_trades(
        trades, args.fee_bps_per_side / 10_000, args.slippage_bps_per_side / 10_000
    )
    valid = evaluated.dropna(subset=["notebook_target"])
    summary = {
        "model": str(args.model), "model_sha256": sha256(args.model),
        "database": str(args.db), "database_sha256": sha256(args.db),
        "rows_raw": len(bars), "rows_evaluated": len(evaluated),
        "data_start_utc": bars.index.min().isoformat(), "data_end_utc": bars.index.max().isoformat(),
        "median_bar_seconds": median_seconds, "expected_bar_seconds": 60,
        "frequency_matches_model_label": median_seconds == 60,
        "feature_order": FEATURES,
        "target": "(close[t-10] - close[t]) / close[t] * 100 (backward-looking; not forecastable target)",
        "lookback_window_bars": 50,
        "signal_time": "bar t close after all t features are known",
        "fill_time": "bar t+1 open",
        "buy_rule": f"prediction > {args.buy_threshold}",
        "sell_rule": f"prediction < {args.sell_threshold}",
        "fee_bps_per_side": args.fee_bps_per_side,
        "slippage_bps_per_side": args.slippage_bps_per_side,
        "trade_count_round_trips": len(trade_df),
        "average_gross_return_per_trade": float(trade_df["gross_return"].mean()) if len(trade_df) else None,
        "gross_compounded_return": compound(trade_df["gross_return"]) if len(trade_df) else 0.0,
        "net_compounded_return_same_trades": compound(trade_df["net_return"]) if len(trade_df) else 0.0,
        "target_mse": float(mean_squared_error(valid["notebook_target"], valid["prediction"])),
        "target_mae": float(mean_absolute_error(valid["notebook_target"], valid["prediction"])),
        "target_r2": float(r2_score(valid["notebook_target"], valid["prediction"])),
        "training_dates": None,
        "training_dates_note": "Not stored in artifact and source training database is absent; do not infer from scaler/model values.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    evaluated[["open", "high", "low", "close", "volume", "prediction", "notebook_target"]].to_csv(
        args.output_dir / "predictions.csv"
    )
    trade_df.to_csv(args.output_dir / "trades.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
