#!/usr/bin/env python3
"""One-shot training and evaluation for BASELINE_REBUILD_CONTRACT.md."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sqlite3
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from baseline_rebuild import PARTITIONS, make_causal_features, make_forward_target, partition_mask


FEATURES = [
    "return_1m", "return_5m", "return_10m", "range_fraction",
    "body_fraction", "volume_log1p", "volume_ratio_20", "volatility_20",
]
INITIAL_EQUITY = 10_000.0
FEE_RATE = 0.001
SLIPPAGE_RATES = (0.0, 0.0001, 0.0002)
MODEL_BREAK_EVEN = (1 + FEE_RATE) / (1 - FEE_RATE) - 1
MOMENTUM_LOOKBACK_MINUTES = 10
MOMENTUM_THRESHOLD = 0.001


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_bars(path: Path) -> pd.DataFrame:
    uri = f"file:{path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        bars = pd.read_sql_query(
            "SELECT timestamp, open, high, low, close, volume FROM candlesticks "
            "WHERE pair='BTCUSDT' ORDER BY timestamp", connection,
        )
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], unit="ms", utc=True)
    return bars.set_index("timestamp")


def accepted_trades(frame: pd.DataFrame, signal: pd.Series, strategy: str) -> pd.DataFrame:
    """Accept non-overlapping fixed-horizon trades, exit before same-open entry."""
    columns = [
        "strategy", "decision_time", "entry_time", "entry_open", "exit_time",
        "exit_open", "gross_return",
    ]
    rows = []
    held_until = None
    for decision_time in frame.index[signal]:
        row = frame.loc[decision_time]
        entry_time = row["entry_time"]
        # A scheduled exit at this timestamp is processed before a new entry.
        if held_until is not None and entry_time < held_until:
            continue
        held_until = row["exit_time"]
        rows.append({
            "strategy": strategy, "decision_time": decision_time,
            "entry_time": entry_time, "entry_open": float(row["entry_open"]),
            "exit_time": held_until, "exit_open": float(row["exit_open"]),
            "gross_return": float(row["exit_open"] / row["entry_open"] - 1),
        })
    return pd.DataFrame(rows, columns=columns)


def price_ledger_and_equity(
    bars: pd.DataFrame, trades: pd.DataFrame, fee_rate: float, slippage_rate: float
) -> tuple[pd.DataFrame, pd.Series, dict]:
    ledger = trades.copy()
    ledger["entry_execution_price"] = np.nan
    ledger["exit_execution_price"] = np.nan
    ledger["quantity_btc"] = np.nan
    ledger["entry_fee_usdt"] = np.nan
    ledger["exit_fee_usdt"] = np.nan
    ledger["net_return"] = np.nan
    cash = INITIAL_EQUITY
    position = 0.0
    entry_events = {row.entry_time: i for i, row in ledger.iterrows()}
    exit_events = {row.exit_time: i for i, row in ledger.iterrows()}
    equity = []
    exposed = 0
    turnover_notional = 0.0
    total_fees = 0.0

    for timestamp, bar in bars.iterrows():
        # Contract ordering: exit first, then a new entry at the same open.
        if timestamp in exit_events:
            i = exit_events[timestamp]
            exit_px = float(ledger.at[i, "exit_open"]) * (1 - slippage_rate)
            exit_notional = position * exit_px
            exit_fee = exit_notional * fee_rate
            cash = exit_notional - exit_fee
            ledger.at[i, "exit_execution_price"] = exit_px
            ledger.at[i, "exit_fee_usdt"] = exit_fee
            entry_cash = float(ledger.at[i, "entry_execution_price"] * ledger.at[i, "quantity_btc"])
            entry_fee = float(ledger.at[i, "entry_fee_usdt"])
            ledger.at[i, "net_return"] = cash / (entry_cash + entry_fee) - 1
            turnover_notional += exit_notional
            total_fees += exit_fee
            position = 0.0
        if timestamp in entry_events:
            i = entry_events[timestamp]
            entry_px = float(ledger.at[i, "entry_open"]) * (1 + slippage_rate)
            quantity = cash / (entry_px * (1 + fee_rate))
            entry_notional = quantity * entry_px
            entry_fee = entry_notional * fee_rate
            ledger.at[i, "entry_execution_price"] = entry_px
            ledger.at[i, "quantity_btc"] = quantity
            ledger.at[i, "entry_fee_usdt"] = entry_fee
            turnover_notional += entry_notional
            total_fees += entry_fee
            cash = 0.0
            position = quantity
        if position:
            exposed += 1
            liquidation = position * float(bar["close"]) * (1 - slippage_rate) * (1 - fee_rate)
            equity.append(liquidation)
        else:
            equity.append(cash)

    curve = pd.Series(equity, index=bars.index, name="equity")
    max_drawdown = float((curve / curve.cummax() - 1).min())
    duration_years = (bars.index[-1] - bars.index[0] + pd.Timedelta(minutes=1)).total_seconds() / (365.25 * 86400)
    average_equity = float(curve.mean())
    metrics = {
        "return": float(curve.iloc[-1] / INITIAL_EQUITY - 1),
        "maximum_drawdown": max_drawdown,
        "total_fees_usdt": total_fees,
        "turnover_notional_usdt": turnover_notional,
        "annualized_turnover": float(turnover_notional / average_equity / duration_years),
        "exposure_fraction": float(exposed / len(bars)),
        "average_return_per_trade": float(ledger["net_return"].mean()) if len(ledger) else None,
    }
    return ledger, curve, metrics


def predictive_metrics(actual: pd.Series, prediction: pd.Series) -> dict:
    return {
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(mean_squared_error(actual, prediction) ** 0.5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    bars = load_bars(args.db)
    features = make_causal_features(bars)
    targets = make_forward_target(bars)
    usable = {}
    for partition in PARTITIONS:
        mask = partition_mask(targets, partition) & features.notna().all(axis=1)
        usable[partition.name] = features.loc[mask, FEATURES].join(targets.loc[mask])

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, fit_intercept=True)),
    ])
    train = usable["train"]
    pipeline.fit(train[FEATURES], train["future_return_10m"])
    training_mean = float(train["future_return_10m"].mean())
    joblib.dump(pipeline, args.output_dir / "ridge_pipeline.joblib")

    summary = {
        "training_rows": len(train), "training_target_mean": training_mean,
        "model_break_even_decimal": MODEL_BREAK_EVEN,
        "fee_bps_per_side": FEE_RATE * 10_000,
        "zero_slippage_is_optimistic": True,
        "partitions": {},
    }
    for split in ("validation", "test"):
        data = usable[split].copy()
        data["ridge_prediction"] = pipeline.predict(data[FEATURES])
        data["training_mean_prediction"] = training_mean
        data["model_signal"] = data["ridge_prediction"] > MODEL_BREAK_EVEN
        data["momentum_signal"] = data["return_10m"] > MOMENTUM_THRESHOLD
        data.to_csv(args.output_dir / f"predictions_{split}.csv")
        split_start, split_end = next((p.start, p.end) for p in PARTITIONS if p.name == split)
        split_bars = bars.loc[(bars.index >= split_start) & (bars.index < split_end)]
        split_summary = {
            "rows": len(data),
            "prediction_metrics": {
                "ridge": predictive_metrics(data["future_return_10m"], data["ridge_prediction"]),
                "training_mean": predictive_metrics(data["future_return_10m"], data["training_mean_prediction"]),
            },
            "strategies": {},
        }
        signals = {
            "no_trade": pd.Series(False, index=data.index),
            "momentum": data["momentum_signal"],
            "ridge": data["model_signal"],
        }
        for strategy, signal in signals.items():
            raw_trades = accepted_trades(data, signal, strategy)
            baseline_ledger = None
            strategy_summary = {
                "trade_count": len(raw_trades),
                "average_gross_return_per_trade": float(raw_trades["gross_return"].mean()) if len(raw_trades) else None,
                "scenarios": {},
            }
            for slippage in SLIPPAGE_RATES:
                ledger, _, metrics = price_ledger_and_equity(split_bars, raw_trades, FEE_RATE, slippage)
                if slippage == 0:
                    baseline_ledger = ledger
                    gross_ledger, _, gross_metrics = price_ledger_and_equity(split_bars, raw_trades, 0.0, 0.0)
                    strategy_summary["gross"] = gross_metrics
                    strategy_summary["average_net_return_per_trade"] = metrics["average_return_per_trade"]
                strategy_summary["scenarios"][f"slippage_{slippage * 10000:.0f}bps_per_side"] = metrics
            baseline_ledger.to_csv(args.output_dir / f"trades_{split}_{strategy}.csv", index=False)
            split_summary["strategies"][strategy] = strategy_summary
        summary["partitions"][split] = split_summary

    config = {
        "command": " ".join(sys.argv),
        "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "sklearn": sklearn.__version__,
        "features": FEATURES,
        "target": "decimal open[t+11] / open[t+1] - 1",
        "pipeline": "StandardScaler(train only) -> Ridge(alpha=1.0, fit_intercept=True)",
        "model_entry_rule": f"prediction > {MODEL_BREAK_EVEN!r}",
        "momentum_rule": f"return_{MOMENTUM_LOOKBACK_MINUTES}m > {MOMENTUM_THRESHOLD}",
        "fee_rate_per_side": FEE_RATE, "slippage_rates_per_side": SLIPPAGE_RATES,
        "initial_equity_usdt": INITIAL_EQUITY,
        "hashes": {
            "input_database_sha256": sha256(args.db),
            "contract_sha256": sha256(Path("BASELINE_REBUILD_CONTRACT.md")),
            "data_code_sha256": sha256(Path("baseline_rebuild.py")),
            "runner_code_sha256": sha256(Path(__file__)),
        },
    }
    config["hashes"]["model_sha256"] = sha256(args.output_dir / "ridge_pipeline.joblib")
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "reproduce.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "conda run -n ml-2 python -m unittest -v test_baseline_rebuild.py test_run_baseline_rebuild.py\n"
        "conda run -n ml-2 python run_baseline_rebuild.py --db ../db/BTCUSDT_1m_52weeks.db "
        "--output-dir baseline_rebuild_results/reproduced_run\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
