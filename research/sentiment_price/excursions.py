"""
excursions.py - MFE / MAE baselines for random (unconditional) entries on 5m bars.

Conventions
-----------
* Entry price = CLOSE of the entry bar (the signal is known at that close).
* The trade path is bars t+1 .. t+H. MFE / MAE use those bars' HIGH / LOW,
  so the entry bar's own range is never counted.
* All excursions are positive fractions (0.01 = 1%).
  long : MFE = max(high)/entry - 1      MAE = 1 - min(low)/entry
  short: MFE = 1 - min(low)/entry       MAE = max(high)/entry - 1
* First passage (TP vs SL): if one bar touches both levels we count the STOP
  (we cannot know the order inside a 5m bar - conservative). Fills are assumed
  at the level itself (no slippage through the level).
* "Every bar is an entry" instead of a random sample: this is the full
  unconditional distribution, so there is no sampling noise in the baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

BARS_PER_HOUR = 12


# ------------------------------------------------------------------ volatility
def ewma_vol(close: pd.Series, n_bars: int = 288) -> pd.Series:
    """EWMA std of 5m log returns, known at the close of each bar (uses bar t)."""
    r = np.log(close).diff()
    return np.sqrt((r ** 2).ewm(span=n_bars, adjust=False).mean())


# --------------------------------------------------------- forward excursions
def forward_excursions(bars: pd.DataFrame, H: int) -> pd.DataFrame:
    """MFE/MAE/time-exit return for long and short entries at every bar, horizon H bars.
    Rows without H future bars are NaN."""
    c = bars["close"]
    hi = bars["high"].rolling(H).max().shift(-H)     # max high over t+1..t+H
    lo = bars["low"].rolling(H).min().shift(-H)
    fwd = c.shift(-H)
    out = pd.DataFrame(index=bars.index)
    out["long_mfe"] = hi / c - 1
    out["long_mae"] = 1 - lo / c
    out["long_ret"] = fwd / c - 1
    out["short_mfe"] = 1 - lo / c
    out["short_mae"] = hi / c - 1
    out["short_ret"] = 1 - fwd / c
    return out


@njit(cache=True)
def _first_passage(close, high, low, H, tp, sl, side):
    """Per entry: outcome (+1 TP, -1 SL, 0 time exit), gross return, bars held."""
    n = close.shape[0]
    outc = np.full(n, np.nan); ret = np.full(n, np.nan); held = np.full(n, np.nan)
    for i in range(n - H):
        e = close[i]
        if side > 0:
            up = e * (1.0 + tp); dn = e * (1.0 - sl)
        else:
            up = e * (1.0 + sl); dn = e * (1.0 - tp)
        o = 0.0; r = 0.0; k = H
        for j in range(i + 1, i + H + 1):
            if side > 0:
                hit_sl = low[j] <= dn; hit_tp = high[j] >= up
            else:
                hit_sl = high[j] >= up; hit_tp = low[j] <= dn
            if hit_sl:
                o = -1.0; r = -sl; k = j - i; break
            if hit_tp:
                o = 1.0; r = tp; k = j - i; break
        if o == 0.0:
            r = side * (close[i + H] / e - 1.0)
        outc[i] = o; ret[i] = r; held[i] = k
    return outc, ret, held


def first_passage(bars: pd.DataFrame, H: int, tp: float, sl: float, side: int = 1) -> pd.DataFrame:
    o, r, k = _first_passage(bars["close"].values, bars["high"].values, bars["low"].values,
                             H, tp, sl, side)
    return pd.DataFrame(dict(outcome=o, ret=r, bars_held=k), index=bars.index)


def tp_sl_grid(bars, mask, H, tps, sls, side=1, cost=0.002):
    """Random-entry results for a TP x SL grid with a time exit at H bars.
    Returns long-format DataFrame: P(TP first), P(SL first), mean net return (bps)."""
    rows = []
    for tp in tps:
        for sl in sls:
            fp = first_passage(bars, H, tp, sl, side)[mask].dropna()
            rows.append(dict(tp_pct=100 * tp, sl_pct=100 * sl,
                             p_tp=(fp.outcome == 1).mean(), p_sl=(fp.outcome == -1).mean(),
                             p_time=(fp.outcome == 0).mean(),
                             gross_bps=1e4 * fp.ret.mean(), net_bps=1e4 * (fp.ret.mean() - cost),
                             mean_hold_h=fp.bars_held.mean() / BARS_PER_HOUR))
    return pd.DataFrame(rows)


# ------------------------------------------------------------- summary tables
def mfe_table(bars, mask, horizons_h, thresholds, side="long"):
    rows = []
    for hh in horizons_h:
        ex = forward_excursions(bars, int(hh * BARS_PER_HOUR))[mask].dropna()
        mfe, mae, ret = ex[f"{side}_mfe"], ex[f"{side}_mae"], ex[f"{side}_ret"]
        row = {"hold_h": hh}
        for x in thresholds:
            row[f"P(MFE>={100*x:g}%)"] = (mfe >= x).mean()
        row["med_MFE_%"] = 100 * mfe.median()
        row["med_MAE_%"] = 100 * mae.median()
        row["P(MAE>=1%)"] = (mae >= 0.01).mean()
        row["mean_ret_bps"] = 1e4 * ret.mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("hold_h")


def survival(x: pd.Series, grid: np.ndarray) -> np.ndarray:
    v = np.sort(x.dropna().values)
    return 1.0 - np.searchsorted(v, grid, side="left") / len(v)


# ------------------------------------------------- real trades vs matched random
def _entry_index(bars_index: pd.DatetimeIndex, t: pd.Timestamp) -> int:
    """Last candle fully closed at time t: open <= t - 5min."""
    return int(bars_index.searchsorted(t - pd.Timedelta(minutes=5), side="right") - 1)


def trade_excursions(bars: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """trades: columns entry_time, exit_time (UTC), side (+1 long / -1 short).
    Entry/exit priced at the close of the last candle completed at that time."""
    idx = bars.index
    c = bars["close"].values; h = bars["high"].values; l = bars["low"].values
    rows = []
    for tr in trades.itertuples():
        i = _entry_index(idx, pd.Timestamp(tr.entry_time))
        j = _entry_index(idx, pd.Timestamp(tr.exit_time))
        if i < 0 or j <= i:
            rows.append(dict(i=i, D=np.nan, mfe=np.nan, mae=np.nan, ret=np.nan)); continue
        e = c[i]; hi = h[i + 1:j + 1].max(); lo = l[i + 1:j + 1].min()
        if tr.side > 0:
            mfe, mae, ret = hi / e - 1, 1 - lo / e, c[j] / e - 1
        else:
            mfe, mae, ret = 1 - lo / e, hi / e - 1, 1 - c[j] / e
        rows.append(dict(i=i, D=j - i, mfe=mfe, mae=mae, ret=ret))
    return pd.concat([trades.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def matched_random(bars: pd.DataFrame, real: pd.DataFrame, n_per: int = 200,
                   window_days: float = 15, vol_match: bool = True, seed: int = 0) -> pd.DataFrame:
    """For each real trade draw n_per random entries with the SAME side and duration,
    within +-window_days of the real entry, and (optionally) in the same EWMA-vol
    decile at entry. Returns one row per random entry with trade_id, mfe, mae, ret."""
    rng = np.random.default_rng(seed)
    c = bars["close"].values; h = bars["high"].values; l = bars["low"].values
    n = len(c)
    vol = ewma_vol(bars["close"])
    dec = pd.qcut(vol.rank(method="first"), 10, labels=False).values
    w = int(window_days * 288)
    out = []
    for k, tr in real.iterrows():
        if not np.isfinite(tr.D):
            continue
        D = int(tr.D); i0 = int(tr.i)
        lo_i = max(1, i0 - w); hi_i = min(n - D - 1, i0 + w)
        cand = np.arange(lo_i, hi_i)
        if vol_match:
            cand = cand[dec[cand] == dec[i0]]
        if len(cand) == 0:
            continue
        pick = rng.choice(cand, size=n_per, replace=len(cand) < n_per)
        for i in pick:
            e = c[i]; hi = h[i + 1:i + D + 1].max(); lo = l[i + 1:i + D + 1].min()
            if tr.side > 0:
                out.append((k, hi / e - 1, 1 - lo / e, c[i + D] / e - 1))
            else:
                out.append((k, 1 - lo / e, hi / e - 1, 1 - c[i + D] / e))
    return pd.DataFrame(out, columns=["trade_id", "mfe", "mae", "ret"])


def compare_real_vs_random(real: pd.DataFrame, rand: pd.DataFrame, thresholds, n_boot: int = 2000,
                           seed: int = 1) -> pd.DataFrame:
    """For each statistic: real value, random mean, random 5-95% band, and a one-sided
    p-value = share of random 'portfolios' (one random match per real trade, resampled)
    that are at least as good as the real trades."""
    rng = np.random.default_rng(seed)
    real = real.dropna(subset=["mfe"])
    groups = {k: g for k, g in rand.groupby("trade_id")}
    ids = [k for k in real.index if k in groups]
    real = real.loc[ids]
    arrs = {k: groups[k][["mfe", "mae", "ret"]].values for k in ids}
    stats = {f"P(MFE>={100*x:g}%)": (lambda m, a, r, x=x: (m >= x).mean(), True) for x in thresholds}
    stats["P(MAE>=1%)"] = (lambda m, a, r: (a >= 0.01).mean(), False)
    stats["median MFE %"] = (lambda m, a, r: 100 * np.median(m), True)
    stats["median MAE %"] = (lambda m, a, r: 100 * np.median(a), False)
    stats["mean ret bps (gross)"] = (lambda m, a, r: 1e4 * r.mean(), True)
    boot = {s: np.empty(n_boot) for s in stats}
    for b in range(n_boot):
        sample = np.array([arrs[k][rng.integers(len(arrs[k]))] for k in ids])
        for s, (f, _) in stats.items():
            boot[s][b] = f(sample[:, 0], sample[:, 1], sample[:, 2])
    rows = []
    for s, (f, higher_better) in stats.items():
        rv = f(real.mfe.values, real.mae.values, real.ret.values)
        bv = boot[s]
        p = (bv >= rv).mean() if higher_better else (bv <= rv).mean()
        rows.append(dict(stat=s, real=rv, random_mean=bv.mean(), random_p05=np.quantile(bv, .05),
                         random_p95=np.quantile(bv, .95), p_value=p))
    return pd.DataFrame(rows).set_index("stat")
