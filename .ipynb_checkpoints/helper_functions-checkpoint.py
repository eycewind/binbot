import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import pandas as pd
import requests
import pandas as pd
import numpy as np
from BinanceClient import BinanceClient
import itertools
import bottleneck as bn

BAR_MINUTES = 5

PRE_HOURS = 12
BASE_HOURS = 72

PRE_BARS  = PRE_HOURS  * 60 // BAR_MINUTES   # 144
BASE_BARS = BASE_HOURS * 60 // BAR_MINUTES   # 864

COMPRESSION_THRESH = -2.6
RANGE_BARS = 24   # last 2 hours

H1 = 24   # 2h
H2 = 72   # 6h
BAR_MIN = 5

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# extra exclusion after shock to avoid post-shock regime bleeding into "non-event"
POST_EXCLUDE_BARS = 12   # 1h = 12 bars


DB_DIR = Path("dbs")
BAD_SUFFIXES = ("UPUSDT","DOWNUSDT","BULLUSDT","BEARUSDT","3LUSDT","3SUSDT","5LUSDT","5SUSDT")

import pandas as pd

def filter_date_range(
    df,
    start_ts,
    end_ts,
    *,
    require_full_coverage=True,   # NEW
    min_coverage_ratio=1.0,       # NEW (1.0 = full)
    return_none_if_short=True     # NEW
):
    """
    Keep only rows in [start_ts, end_ts]. Optionally require the df to cover
    the full window (or a minimum fraction of it) to avoid newly listed symbols.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV df with DatetimeIndex
    start_ts, end_ts : str|pd.Timestamp
        Inclusive start, inclusive end (we'll treat as inclusive)
    require_full_coverage : bool
        If True, require df.index.min() <= start_ts and df.index.max() >= end_ts
        (or minimum fraction via min_coverage_ratio)
    min_coverage_ratio : float
        Minimum fraction of requested window that must be covered by df.
        Use 1.0 for full coverage, 0.95 for small gaps tolerance, etc.
    return_none_if_short : bool
        If True, return None when coverage is insufficient; else return empty df.

    Returns
    -------
    pd.DataFrame or None
    """
    if df is None or df.empty:
        return None if return_none_if_short else df.copy()

    df = df.sort_index()
    start_ts = pd.Timestamp(start_ts)
    end_ts   = pd.Timestamp(end_ts)

    # compute coverage vs requested window
    req_secs = (end_ts - start_ts).total_seconds()
    have_start = df.index.min()
    have_end   = df.index.max()

    if require_full_coverage:
        # strict full coverage OR ratio-based coverage
        overlap_start = max(have_start, start_ts)
        overlap_end   = min(have_end, end_ts)
        overlap_secs  = max(0.0, (overlap_end - overlap_start).total_seconds())
        ratio = overlap_secs / req_secs if req_secs > 0 else 0.0

        if ratio < min_coverage_ratio:
            return None if return_none_if_short else df.iloc[0:0].copy()

    # slice
    out = df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()
    if out.empty:
        return None if return_none_if_short else out

    return out


def truncate_weeks(
    df,
    weeks,
    mode="first",
    *,
    min_required_weeks=None,   # NEW
    return_none_if_short=True  # NEW
):
    """
    Truncate dataframe to a given number of weeks.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV dataframe with DatetimeIndex
    weeks : float
        Number of weeks to keep
    mode : {"first","last"}
        - "first": keep earliest weeks
        - "last" : keep latest weeks
    min_required_weeks : float or None
        If set, skip df if total coverage < this many weeks
    return_none_if_short : bool
        If True, return None when df is too short
        If False, return empty df

    Returns
    -------
    pd.DataFrame or None
    """
    if isinstance(df, tuple) and len(df) == 2:
        df = df[1]

    if df is None or df.empty:
        return None if return_none_if_short else df.copy()

    df = df.sort_index()

    total_weeks = (df.index.max() - df.index.min()).total_seconds() / (7 * 24 * 3600)

    # 🔒 HARD FILTER
    if min_required_weeks is not None and total_weeks < min_required_weeks:
        return None if return_none_if_short else df.iloc[0:0].copy()

    # 🪓 TRUNCATION
    if mode == "first":
        end = df.index.min() + pd.Timedelta(weeks=weeks)
        return df.loc[df.index <= end].copy()

    elif mode == "last":
        start = df.index.max() - pd.Timedelta(weeks=weeks)
        return df.loc[df.index >= start].copy()

    else:
        raise ValueError("mode must be 'first' or 'last'")


def interval_slug(s: str) -> str:
    return s.strip().replace(" ", "").replace("/", "").lower()

def make_db_name(pair: str, interval: str, weeks: int) -> str:
    return f"{pair}_{interval_slug(interval)}_{weeks}weeks.db"

def list_downloaded_pairs(db_dir=DB_DIR):
    """
    Returns list of symbols inferred from SQLite filenames in db_dir.
    Assumes file contains a token like BTCUSDT somewhere in the name.
    """
    symbols = set()

    for p in Path(db_dir).glob("*.db"):
        name = p.stem  # filename without extension
        # find the first token that looks like a Binance symbol ending with USDT
        m = re.search(r"([A-Z0-9]{2,20}USDT)", name.upper())
        if m:
            symbols.add(m.group(1))

    # filter out leveraged tokens etc
    symbols = [s for s in symbols if s.endswith("USDT") and not s.endswith(BAD_SUFFIXES)]
    return sorted(symbols)
    
def load_or_fetch_pair_df(pair: str, interval: str, weeks: int) -> tuple[str, "pd.DataFrame"]:
    db_name = make_db_name(pair, interval, weeks)
    db_path = "./db/" + db_name

    print(f"[{pair}] DB: {db_path}")

    binance_client = BinanceClient(db_path)
    binance_client.set_interval(interval)

    df = None

    if os.path.exists(db_path):
        df = binance_client.fetch_data_from_db(pair)
        if df is not None and not df.empty:
            print(f"[{pair}] Loaded {len(df):,} rows from DB.")
        else:
            df = None

    if df is None:
        print(f"[{pair}] No usable DB data found -> fetching from Binance...")

        api_secret = os.getenv("BINANCE_SECRET_KEY")
        api_key = os.getenv("BINANCE_API_KEY")
        binance_client.make(api_key, api_secret)

        server_time = binance_client.get_server_time()
        end_dt = datetime.fromtimestamp(server_time["serverTime"] / 1000, tz=timezone.utc)
        start_dt = end_dt - timedelta(weeks=weeks)

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        data = binance_client.fetch_data(pair, start_ms, end_ms)
        if data is None or data.empty:
            raise RuntimeError(f"[{pair}] No data returned from Binance for the requested range.")

        binance_client.store_data_to_db(pair, data)

        df = binance_client.fetch_data_from_db(pair)
        if df is None or df.empty:
            raise RuntimeError(f"[{pair}] Data fetched/stored but DB load returned empty.")

        print(f"[{pair}] Fetched + stored + loaded {len(df):,} rows.")

    df = df.sort_index()
    return db_path, df

def detect_volume_shocks(
    df,
    vol_win=144,          # 12h median
    impulse_k=12,         # 60 min impulse
    rvol_thresh=6.0,
    impulse_thresh=0.04,
    lookahead=24,
    cooldown=12,
):
    # ✅ keep NaN check exactly (dropna) and keep copy (same as you had)
    d = df[["open","high","low","close","volume"]].dropna().copy()
    n = len(d)
    if n == 0:
        return pd.DataFrame()

    # --- rolling median (same structural step)
    # If you want extra speed without changing logic, you can optionally use bottleneck.
    # But we’ll keep pandas by default (safe & identical).
    
    # vol_med = d["volume"].rolling(vol_win).median()
    vol_np = d["volume"].to_numpy(dtype=np.float64)
    vol_med_np = bn.move_median(vol_np, window=vol_win, min_count=vol_win)
    vol_med = pd.Series(vol_med_np, index=d.index)
    rvol = d["volume"] / vol_med

    # Signed impulse (direction)
    impulse_signed = d["close"] / d["close"].shift(impulse_k) - 1
    # Absolute impulse (magnitude)
    impulse_abs = impulse_signed.abs()

    # --- convert once to numpy arrays (biggest speedup; structure unchanged)
    close_np = d["close"].to_numpy(dtype=np.float64)
    rvol_np = rvol.to_numpy(dtype=np.float64)
    imp_abs_np = impulse_abs.to_numpy(dtype=np.float64)
    imp_signed_np = impulse_signed.to_numpy(dtype=np.float64)

    # index as numpy array (fast access)
    idx_np = d.index.to_numpy()

    events = []

    # start i at max(vol_win, impulse_k) to avoid early NaNs
    start_i = max(vol_win, impulse_k)
    i = start_i

    end_i = n - lookahead

    # localize thresholds (tiny but free speed)
    rv_th = float(rvol_thresh)
    im_th = float(impulse_thresh)

    # localize append (tiny but free speed)
    ev_append = events.append

    while i < end_i:
        rv = rvol_np[i]
        ia = imp_abs_np[i]

        # keep same condition structure
        if (rv >= rv_th) and (ia >= im_th):
            px0 = close_np[i]
            isg = imp_signed_np[i]

            ev_append({
                "event_ts": idx_np[i],
                "close_event": float(px0),
                "rvol": float(rv),
                # keep existing name as magnitude for backward-compat
                "impulse": float(ia),
                # new: signed impulse for direction
                "impulse_signed": float(isg),
                "shock_direction": "long" if isg > 0 else "short",
            })
            i += cooldown
        else:
            i += 1

    return pd.DataFrame(events)
import pandas as pd
import numpy as np

def detect_compression_state(
    df: pd.DataFrame,
    *,
    atr_short: int = 20,          # ~100 min on 5m
    atr_long: int = 100,          # ~8 hours on 5m
    vol_win: int = 144,           # volume median window (12 hours)
    vol_ratio_thresh: float = 0.6,
    rvol_thresh: float = 0.6,
    min_duration: int = 12        # bars of sustained compression (60 min)
):
    """
    Detects pre-shock compression state.

    Returns df with added columns:
      - atr
      - atr_med
      - vol_compression
      - rvol
      - volu_compression
      - compression_raw
      - compression_duration
      - is_compressed
    """

    d = df.copy().sort_index()
    d = d[["open","high","low","close","volume"]].dropna()

    # -----------------------
    # 1) Volatility (ATR)
    # -----------------------
    high = d["high"]
    low  = d["low"]
    close = d["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    d["atr"] = tr.rolling(atr_short).mean()
    d["atr_med"] = d["atr"].rolling(atr_long).median()

    d["vol_compression"] = d["atr"] / d["atr_med"]
    d["is_vol_compressed"] = d["vol_compression"] <= vol_ratio_thresh

    # -----------------------
    # 2) Volume compression
    # -----------------------
    vol_med = d["volume"].rolling(vol_win).median()
    d["rvol"] = d["volume"] / vol_med
    d["is_volume_compressed"] = d["rvol"] <= rvol_thresh

    # -----------------------
    # 3) Raw compression flag
    # -----------------------
    d["compression_raw"] = (
        d["is_vol_compressed"] &
        d["is_volume_compressed"]
    )

    # -----------------------
    # 4) Duration counter
    # -----------------------
    duration = np.zeros(len(d), dtype=int)

    for i in range(1, len(d)):
        if d["compression_raw"].iloc[i]:
            duration[i] = duration[i-1] + 1
        else:
            duration[i] = 0

    d["compression_duration"] = duration

    # -----------------------
    # 5) Final state
    # -----------------------
    d["is_compressed"] = d["compression_duration"] >= min_duration

    return d
def get_compression_times(df_c):
    """
    df_c = output of detect_compression_state(df)
    Returns DatetimeIndex of bars where compression starts
    (we take the FIRST bar of each compression block)
    """
    comp = df_c["is_compressed"]

    # compression start = is_compressed == True AND previous == False
    comp_start = comp & (~comp.shift(1).fillna(False))

    return df_c.index[comp_start]

def get_shock_times(events_sym):
    """
    events_sym = events filtered to one symbol
    Returns DatetimeIndex of shock timestamps
    """
    return pd.to_datetime(events_sym["event_ts"]).sort_values()

def compression_leads_shock(
    compression_times,
    shock_times,
    *,
    max_lead_minutes=120
):
    """
    For each compression start time, check whether
    a shock occurs within max_lead_minutes AFTER it.
    """
    lead_td = pd.Timedelta(minutes=max_lead_minutes)

    hits = 0
    for t in compression_times:
        if ((shock_times > t) & (shock_times <= t + lead_td)).any():
            hits += 1

    total = len(compression_times)

    return {
        "compression_events": total,
        "hits": hits,
        "hit_rate": hits / total if total > 0 else np.nan
    }

import numpy as np
import pandas as pd

# -----------------------------
# Helpers: ATR + linear slope
# -----------------------------
def _atr14(series_df: pd.DataFrame, n: int = 14) -> pd.Series:
    # Same TR definition, faster than concat+max(axis=1)
    h = series_df["high"]
    l = series_df["low"]
    c = series_df["close"]
    pc = c.shift(1)

    tr1 = (h - l)
    tr2 = (h - pc).abs()
    tr3 = (l - pc).abs()

    # elementwise max across the three Series (preserves index)
    tr = np.maximum(tr1.to_numpy(dtype=float),
         np.maximum(tr2.to_numpy(dtype=float), tr3.to_numpy(dtype=float)))
    tr = pd.Series(tr, index=series_df.index)

    return tr.rolling(n).mean()

def _lin_slope(y: np.ndarray) -> float:
    """Slope of y versus index (0..n-1). Returns 0 if too short or constant."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3 or np.all(~np.isfinite(y)):
        return np.nan
    x = np.arange(n, dtype=float)
    y = y - np.nanmean(y)
    x = x - x.mean()
    denom = np.nansum(x * x)
    return float(np.nansum(x * y) / denom) if denom > 0 else 0.0

# -----------------------------
# Candle overlap + wick/body
# -----------------------------
def _overlap_ratio(window: pd.DataFrame) -> float:
    """Mean normalized overlap between consecutive candles (0..1-ish)."""
    if len(window) < 2:
        return np.nan
    h = window["high"].to_numpy(dtype=float)
    l = window["low"].to_numpy(dtype=float)
    rng = (h - l)
    rng[rng == 0] = np.nan

    ov = np.minimum(h[1:], h[:-1]) - np.maximum(l[1:], l[:-1])
    ov = np.maximum(ov, 0.0)
    ov_norm = ov / rng[1:]
    return float(np.nanmean(ov_norm))

def _wick_to_body_ratio(window: pd.DataFrame) -> float:
    """(upper+lower wick)/body averaged over window."""
    o = window["open"].to_numpy(dtype=float)
    c = window["close"].to_numpy(dtype=float)
    h = window["high"].to_numpy(dtype=float)
    l = window["low"].to_numpy(dtype=float)

    body = np.abs(c - o)
    body[body == 0] = np.nan

    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    wick = upper + lower
    return float(np.nanmean(wick / body))

# -----------------------------
# Feature extraction for one window
# -----------------------------
def extract_window_features(
    d: pd.DataFrame,
    t0: pd.Timestamp,
    *,
    pre_bars: int = 36,
    gap_bars: int = 2,
    atr_long_med_win: int = 144
) -> dict | None:
    """
    Compute pre-shock window features for a single event time t0.

    Window is: [t0 - pre_bars, t0 - gap_bars]
    Assumes d is indexed by datetime and has open/high/low/close/volume.
    """
    if not isinstance(d.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex")

    # Only sort if needed (same result, avoids repeated work)
    if not d.index.is_monotonic_increasing:
        d = d.sort_index()

    # Need baseline columns (same logic)
    if "atr14" not in d.columns:
        d["atr14"] = _atr14(d, 14)
    if "vol_med_12h" not in d.columns:
        d["vol_med_12h"] = d["volume"].rolling(atr_long_med_win).median()
    if "rvol_12h" not in d.columns:
        d["rvol_12h"] = d["volume"] / d["vol_med_12h"]
    if "atr14_med_12h" not in d.columns:
        d["atr14_med_12h"] = d["atr14"].rolling(atr_long_med_win).median()

    # Locate t0 (same behavior)
    if t0 not in d.index:
        loc = d.index.searchsorted(t0, side="right") - 1
        if loc < 0:
            return None
        t0 = d.index[loc]

    # Faster get_loc if unique index; pandas handles both anyway
    end_loc = d.index.get_loc(t0) - gap_bars
    start_loc = end_loc - pre_bars + 1
    if start_loc < 0 or end_loc <= start_loc:
        return None

    # Keep copy (no functional change), but do minimal work before it
    w = d.iloc[start_loc:end_loc+1].copy()
    if len(w) < max(10, pre_bars // 2):
        return None

    # Baseline at window end
    atr_med = d["atr14_med_12h"].iloc[end_loc]
    if not np.isfinite(atr_med) or atr_med <= 0:
        return None

    # Pull arrays once (avoids repeated Series conversions)
    atr14_np = w["atr14"].to_numpy(dtype=float)
    rvol12_np = w["rvol_12h"].to_numpy(dtype=float)

    atr_mean = float(np.nanmean(atr14_np))
    atr_mean_ratio = atr_mean / float(atr_med)

    atr_slope = _lin_slope(atr14_np)
    rvol_mean = float(np.nanmean(rvol12_np))
    rvol_slope = _lin_slope(rvol12_np)

    # Use numpy for max/min (faster than Series reductions)
    high_np = w["high"].to_numpy(dtype=float)
    low_np  = w["low"].to_numpy(dtype=float)
    window_high = float(np.nanmax(high_np))
    window_low  = float(np.nanmin(low_np))

    window_close = float(w["close"].iloc[-1])
    range_pct = (window_high - window_low) / window_close if window_close != 0 else np.nan

    overlap_ratio = _overlap_ratio(w)
    wick_to_body = _wick_to_body_ratio(w)

    return {
        "t0": t0,
        "win_start": w.index[0],
        "win_end": w.index[-1],
        "atr_mean_ratio": atr_mean_ratio,
        "atr_slope": atr_slope,
        "rvol_mean": rvol_mean,
        "rvol_slope": rvol_slope,
        "range_pct": range_pct,
        "overlap_ratio": overlap_ratio,
        "wick_to_body_ratio": wick_to_body,
    }

# -----------------------------
# Build dataset: shock vs control
# -----------------------------
def build_preshock_dataset(
    dfs: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    *,
    pre_bars: int = 36,
    gap_bars: int = 2,
    control_per_event: int = 1,
    control_lookahead_bars: int = 36,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Returns a labeled dataset with:
      label=1 for pre-shock windows
      label=0 for matched control windows (same symbol, same window length)
    """
    rng = np.random.default_rng(seed)

    ev = events.copy()
    ev["event_ts"] = pd.to_datetime(ev["event_ts"])
    ev = ev.sort_values(["symbol", "event_ts"])

    rows = []
    rows_append = rows.append  # tiny speedup

    # Precompute shock times per symbol for fast control selection (same)
    shock_times_by_sym = {
        sym: pd.to_datetime(g["event_ts"]).sort_values().to_numpy(dtype="datetime64[ns]")
        for sym, g in ev.groupby("symbol", sort=False)
    }

    # constant used in control window end time (same as your 5-min bars assumption)
    bar_minutes = 5

    for sym, g in ev.groupby("symbol", sort=False):
        if sym not in dfs or dfs[sym].empty:
            continue

        d = dfs[sym].copy()
        if not d.index.is_monotonic_increasing:
            d = d.sort_index()

        d = d[["open","high","low","close","volume"]].dropna()
        if len(d) < (pre_bars + gap_bars + 200):
            continue

        if not isinstance(d.index, pd.DatetimeIndex):
            raise ValueError(f"{sym} df index must be DatetimeIndex")

        # Precompute baseline columns once per symbol (same)
        d["atr14"] = _atr14(d, 14)
        d["vol_med_12h"] = d["volume"].rolling(144).median()
        d["rvol_12h"] = d["volume"] / d["vol_med_12h"]
        d["atr14_med_12h"] = d["atr14"].rolling(144).median()

        shock_times = shock_times_by_sym.get(sym, np.array([], dtype="datetime64[ns]"))

        # Control sampling bounds (same)
        end_min = pre_bars + gap_bars
        end_max = len(d) - control_lookahead_bars - 1
        if end_max <= end_min:
            continue

        # Cache index for fast access
        d_index = d.index

        # Helper: fast future-shock check using binary search (same result as np.any mask)
        def _has_future_shock(t1n: np.datetime64, t2n: np.datetime64) -> bool:
            if shock_times.size == 0:
                return False
            # need shocks with (shock_times > t1n) & (shock_times <= t2n)
            j = shock_times.searchsorted(t1n, side="right")
            return (j < shock_times.size) and (shock_times[j] <= t2n)

        for t0 in pd.to_datetime(g["event_ts"]):
            # --- Positive (pre-shock) ---
            feat = extract_window_features(d, t0, pre_bars=pre_bars, gap_bars=gap_bars)
            if feat is None:
                continue

            feat.update({"symbol": sym, "label": 1})
            rows_append(feat)

            # --- Controls (non-shock) ---
            if control_per_event <= 0:
                continue

            attempts = 0
            got = 0
            feat_t0 = feat["t0"]  # cache

            while got < control_per_event and attempts < 200:
                attempts += 1
                end_loc = int(rng.integers(end_min, end_max))
                t_ctrl = d_index[end_loc]

                # Same leakage exclusion
                if abs((t_ctrl - feat_t0).total_seconds()) < 6 * 3600:
                    continue

                # Same future window definition
                t1 = t_ctrl
                t2 = t_ctrl + pd.Timedelta(minutes=bar_minutes * control_lookahead_bars)

                # Faster future shock check (same condition)
                if shock_times.size > 0:
                    t1n = t1.to_datetime64()
                    t2n = t2.to_datetime64()
                    if _has_future_shock(t1n, t2n):
                        continue

                feat_c = extract_window_features(d, t_ctrl, pre_bars=pre_bars, gap_bars=gap_bars)
                if feat_c is None:
                    continue

                feat_c.update({"symbol": sym, "label": 0})
                rows_append(feat_c)
                got += 1

    return pd.DataFrame(rows)
import numpy as np
import pandas as pd

def compute_instability_features_over_window(
    d: pd.DataFrame,
    *,
    pre_bars: int = 36,
    gap_bars: int = 2,
    vol_med_win: int = 144
) -> pd.DataFrame:
    """
    Adds rolling-window (per bar) pre-shock features aligned to each bar t0,
    computed from [t0-pre_bars, t0-gap_bars].
    Output columns are same as dataset: atr_mean_ratio, atr_slope, rvol_mean, rvol_slope,
    range_pct, wick_to_body_ratio.
    """
    df = d.copy()
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # ✅ keep NaN check (dropna) exactly as before
    df = df[["open","high","low","close","volume"]].dropna()

    # precompute base series (same)
    df["atr14"] = _atr14(df, 14)
    df["atr14_med_12h"] = df["atr14"].rolling(vol_med_win).median()
    df["vol_med_12h"] = df["volume"].rolling(vol_med_win).median()
    df["rvol_12h"] = df["volume"] / df["vol_med_12h"]

    n = len(df)
    cols = ["atr_mean_ratio","atr_slope","rvol_mean","rvol_slope","range_pct","wick_to_body_ratio"]

    # Pre-fill with NaNs (same end result, much faster than df.at in a loop)
    out_atr_mean_ratio = np.full(n, np.nan, dtype=np.float64)
    out_atr_slope      = np.full(n, np.nan, dtype=np.float64)
    out_rvol_mean      = np.full(n, np.nan, dtype=np.float64)
    out_rvol_slope     = np.full(n, np.nan, dtype=np.float64)
    out_range_pct      = np.full(n, np.nan, dtype=np.float64)
    out_wick_body      = np.full(n, np.nan, dtype=np.float64)

    # Pull arrays once
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)

    atr14 = df["atr14"].to_numpy(dtype=np.float64)
    atr14_med = df["atr14_med_12h"].to_numpy(dtype=np.float64)
    rvol12 = df["rvol_12h"].to_numpy(dtype=np.float64)

    # Precompute x-centered and denom for slope (window length is constant = pre_bars)
    # This matches _lin_slope behavior for fixed-length windows (same linear regression formula)
    x = np.arange(pre_bars, dtype=np.float64)
    x = x - x.mean()
    denom = np.nansum(x * x)  # constant > 0 for pre_bars>=2

    def _lin_slope_fixed(y_window: np.ndarray) -> float:
        # Same handling as _lin_slope: if too short / all nonfinite => nan
        # Here length is fixed, but keep guard for robustness.
        if y_window.size < 3 or np.all(~np.isfinite(y_window)):
            return np.nan
        yw = y_window - np.nanmean(y_window)
        # if denom is 0 (shouldn't happen), return 0 like original
        return float(np.nansum(x * yw) / denom) if denom > 0 else 0.0

    start_endloc = pre_bars + gap_bars
    if n <= start_endloc:
        # add columns and return quickly
        df["atr_mean_ratio"] = out_atr_mean_ratio
        df["atr_slope"] = out_atr_slope
        df["rvol_mean"] = out_rvol_mean
        df["rvol_slope"] = out_rvol_slope
        df["range_pct"] = out_range_pct
        df["wick_to_body_ratio"] = out_wick_body
        return df

    for end_loc in range(start_endloc, n):
        # window ends at end_loc-gap_bars
        w_end = end_loc - gap_bars
        w_start = w_end - pre_bars + 1

        atr_med = atr14_med[w_end]
        if not np.isfinite(atr_med) or atr_med <= 0:
            continue

        # Slice views (no DataFrame slicing)
        sl = slice(w_start, w_end + 1)

        atr_w = atr14[sl]
        rvol_w = rvol12[sl]

        atr_mean = float(np.nanmean(atr_w))
        out_atr_mean_ratio[end_loc] = atr_mean / float(atr_med)
        out_atr_slope[end_loc] = _lin_slope_fixed(atr_w)

        out_rvol_mean[end_loc] = float(np.nanmean(rvol_w))
        out_rvol_slope[end_loc] = _lin_slope_fixed(rvol_w)

        # Range pct
        window_high = float(np.nanmax(h[sl]))
        window_low  = float(np.nanmin(l[sl]))
        window_close = float(c[w_end])
        out_range_pct[end_loc] = (window_high - window_low) / window_close if window_close else np.nan

        # wick_to_body_ratio (same math as _wick_to_body_ratio, but on arrays)
        o_w = o[sl]; c_w = c[sl]; h_w = h[sl]; l_w = l[sl]
        body = np.abs(c_w - o_w)
        body[body == 0] = np.nan
        upper = h_w - np.maximum(o_w, c_w)
        lower = np.minimum(o_w, c_w) - l_w
        wick = upper + lower
        out_wick_body[end_loc] = float(np.nanmean(wick / body))

    # Assign columns once (much faster than df.at in loop)
    df["atr_mean_ratio"] = out_atr_mean_ratio
    df["atr_slope"] = out_atr_slope
    df["rvol_mean"] = out_rvol_mean
    df["rvol_slope"] = out_rvol_slope
    df["range_pct"] = out_range_pct
    df["wick_to_body_ratio"] = out_wick_body

    return df


def classify_instability(df_feat: pd.DataFrame) -> pd.Series:
    """
    Returns boolean Series indexed like df_feat: True when in pre-shock instability state.
    """
    return (
        (df_feat["rvol_mean"] >= 2.0) &
        (df_feat["rvol_slope"] > 0.0) &
        (df_feat["atr_mean_ratio"] >= 1.10) &
        (df_feat["wick_to_body_ratio"] >= 1.60) &
        (df_feat["range_pct"] >= 0.035)
    )

def best_threshold_youden(ds: pd.DataFrame, feature: str, higher_is_more_shock: bool = True):
    x = ds[feature].to_numpy(dtype=float)
    y = ds["label"].to_numpy(dtype=int)

    # candidate thresholds = percentiles (fast, robust)
    qs = np.linspace(5, 95, 91)
    thrs = np.percentile(x[np.isfinite(x)], qs)

    best = None
    for thr in thrs:
        pred = (x >= thr) if higher_is_more_shock else (x <= thr)
        pred = pred & np.isfinite(x)

        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        tn = np.sum((pred == 0) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))

        tpr = tp / (tp + fn) if (tp + fn) else 0
        fpr = fp / (fp + tn) if (fp + tn) else 0
        j = tpr - fpr

        if best is None or j > best["J"]:
            best = {"feature": feature, "thr": float(thr), "TPR": tpr, "FPR": fpr, "J": j}

    return best

def best_threshold_youden(ds: pd.DataFrame, feature: str, higher_is_more_shock: bool = True):
    x = ds[feature].to_numpy(dtype=float)
    y = ds["label"].to_numpy(dtype=int)

    # candidate thresholds = percentiles (fast, robust)
    qs = np.linspace(5, 95, 91)
    thrs = np.percentile(x[np.isfinite(x)], qs)

    best = None
    for thr in thrs:
        pred = (x >= thr) if higher_is_more_shock else (x <= thr)
        pred = pred & np.isfinite(x)

        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        tn = np.sum((pred == 0) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))

        tpr = tp / (tp + fn) if (tp + fn) else 0
        fpr = fp / (fp + tn) if (fp + tn) else 0
        j = tpr - fpr

        if best is None or j > best["J"]:
            best = {"feature": feature, "thr": float(thr), "TPR": tpr, "FPR": fpr, "J": j}

    return best

def mark_future_shock(df_feat: pd.DataFrame, events: pd.DataFrame, horizon_bars: int):
    """
    For each bar in df_feat, mark whether a shock occurs within the next horizon_bars.
    """
    shock_times = pd.to_datetime(events["event_ts"]).values

    df = df_feat.copy()
    df["future_shock"] = False

    idx = df.index.values
    for i in range(len(idx)):
        t0 = idx[i]
        t1 = idx[min(i + horizon_bars, len(idx) - 1)]
        # any shock in (t0, t1]?
        df.iloc[i, df.columns.get_loc("future_shock")] = np.any(
            (shock_times > t0) & (shock_times <= t1)
        )

    return df

def classify_pre_shock_instability_v1(df_feat: pd.DataFrame) -> pd.Series:
    """
    Rule-based pre-shock instability classifier.
    Calibrated via Youden J on your dataset.
    """
    return (
        (df_feat["range_pct"] >= 0.037) &
        (df_feat["rvol_slope"] >= 0.032) &
        (df_feat["rvol_mean"] >= 2.10)
    )

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Faster ATR (same definition)
# ------------------------------------------------------------
def _atr(series_df: pd.DataFrame, n: int = 14) -> pd.Series:
    h = series_df["high"]
    l = series_df["low"]
    c = series_df["close"]
    pc = c.shift(1)

    tr1 = (h - l).to_numpy(dtype=float)
    tr2 = (h - pc).abs().to_numpy(dtype=float)
    tr3 = (l - pc).abs().to_numpy(dtype=float)

    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = pd.Series(tr, index=series_df.index)

    return tr.rolling(n).mean()

def _lin_slope_fixed(y_window: np.ndarray, x_centered: np.ndarray, denom: float) -> float:
    # matches your _lin_slope behavior for fixed-length windows
    if y_window.size < 3 or np.all(~np.isfinite(y_window)):
        return np.nan
    ym = np.nanmean(y_window)
    yw = y_window - ym
    return float(np.nansum(x_centered * yw) / denom) if denom > 0 else 0.0

# ------------------------------------------------------------
# Optimized rolling/window features aligned to each bar t0
# ------------------------------------------------------------
def compute_instability_features_over_window(
    d: pd.DataFrame,
    *,
    pre_bars: int = 36,
    gap_bars: int = 2,
    vol_med_win: int = 144,   # 12h baseline
    atr_n: int = 14
) -> pd.DataFrame:
    df = d.copy()
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # ✅ keep NaN check
    df = df[["open","high","low","close","volume"]].dropna()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df index must be DatetimeIndex")

    # base series (computed once) - same
    df["atr14"] = _atr(df, atr_n)
    df["atr14_med_12h"] = df["atr14"].rolling(vol_med_win).median()
    df["vol_med_12h"] = df["volume"].rolling(vol_med_win).median()
    df["rvol_12h"] = df["volume"] / df["vol_med_12h"]

    n = len(df)
    cols = ["atr_mean_ratio","atr_slope","rvol_mean","rvol_slope","range_pct","wick_to_body_ratio"]

    # Preallocate outputs (same final content as filling df.at)
    out_atr_mean_ratio = np.full(n, np.nan, dtype=np.float64)
    out_atr_slope      = np.full(n, np.nan, dtype=np.float64)
    out_rvol_mean      = np.full(n, np.nan, dtype=np.float64)
    out_rvol_slope     = np.full(n, np.nan, dtype=np.float64)
    out_range_pct      = np.full(n, np.nan, dtype=np.float64)
    out_wick_body      = np.full(n, np.nan, dtype=np.float64)

    # Pull arrays once
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)

    atr14 = df["atr14"].to_numpy(dtype=np.float64)
    atr14_med = df["atr14_med_12h"].to_numpy(dtype=np.float64)
    rvol12 = df["rvol_12h"].to_numpy(dtype=np.float64)

    # Precompute slope x once (window length constant = pre_bars)
    x = np.arange(pre_bars, dtype=np.float64)
    x = x - x.mean()
    denom = float(np.nansum(x * x))  # constant

    start_endloc = pre_bars + gap_bars
    if n <= start_endloc:
        for col, arr in zip(cols, [out_atr_mean_ratio, out_atr_slope, out_rvol_mean, out_rvol_slope, out_range_pct, out_wick_body]):
            df[col] = arr
        return df

    for end_loc in range(start_endloc, n):
        # IMPORTANT: you align features to t0=end_loc,
        # but window end is w_end=end_loc-gap_bars (same as your code)
        w_end = end_loc - gap_bars
        w_start = w_end - pre_bars + 1

        atr_med = atr14_med[w_end]
        if not np.isfinite(atr_med) or atr_med <= 0:
            continue

        sl = slice(w_start, w_end + 1)

        atr_w = atr14[sl]
        rvol_w = rvol12[sl]

        atr_mean = float(np.nanmean(atr_w))
        out_atr_mean_ratio[end_loc] = atr_mean / float(atr_med)
        out_atr_slope[end_loc] = _lin_slope_fixed(atr_w, x, denom)

        out_rvol_mean[end_loc] = float(np.nanmean(rvol_w))
        out_rvol_slope[end_loc] = _lin_slope_fixed(rvol_w, x, denom)

        window_high = float(np.nanmax(h[sl]))
        window_low  = float(np.nanmin(l[sl]))
        window_close = float(c[w_end])
        out_range_pct[end_loc] = (window_high - window_low) / window_close if window_close else np.nan

        # Wick/body ratio (same math as your _wick_to_body_ratio)
        o_w = o[sl]; c_w = c[sl]; h_w = h[sl]; l_w = l[sl]
        body = np.abs(c_w - o_w)
        body[body == 0] = np.nan
        upper = h_w - np.maximum(o_w, c_w)
        lower = np.minimum(o_w, c_w) - l_w
        wick = upper + lower

        xratio = wick / body
        xratio = xratio[np.isfinite(xratio)]
        out_wick_body[end_loc] = float(np.mean(xratio)) if xratio.size else np.nan

    # Assign once (fast)
    df["atr_mean_ratio"] = out_atr_mean_ratio
    df["atr_slope"] = out_atr_slope
    df["rvol_mean"] = out_rvol_mean
    df["rvol_slope"] = out_rvol_slope
    df["range_pct"] = out_range_pct
    df["wick_to_body_ratio"] = out_wick_body

    return df

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def future_shock_flag(index: pd.DatetimeIndex, event_ts, horizon_bars: int) -> np.ndarray:
    """
    Returns boolean array: shock occurs within next horizon_bars (aligned to index).
    O(N) via cumulative sum, no per-bar scanning.
    """
    n = len(index)
    if event_ts is None or len(event_ts) == 0:
        return np.zeros(n, dtype=bool)

    shock_pos = index.searchsorted(pd.to_datetime(event_ts))
    shock_pos = shock_pos[(shock_pos >= 0) & (shock_pos < n)]
    if shock_pos.size == 0:
        return np.zeros(n, dtype=bool)

    shock_flag = np.zeros(n, dtype=np.int8)
    shock_flag[shock_pos] = 1

    c = np.cumsum(shock_flag, dtype=np.int32)
    c_pad = np.concatenate([[0], c])  # length n+1

    h = horizon_bars
    ends = np.minimum(np.arange(n) + h, n - 1)
    # sum in (i, i+h] = c[ends] - c[i]
    future_counts = c_pad[ends + 1] - c_pad[np.arange(n) + 1]
    return future_counts > 0

import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm

# ---- Worker must be TOP-LEVEL for multiprocessing (do not nest it) ----
def _worker_compute_instability(sym: str, df: pd.DataFrame):
    """
    Returns (sym, is_instability_df) where is_instability_df has one column: ['is_instability']
    """
    if df is None or df.empty:
        return sym, None

    df_feat = compute_instability_features_over_window(df)
    df_feat["is_instability"] = classify_pre_shock_instability_v1(df_feat)

    # Return only the small piece to reduce inter-process overhead
    out = df_feat[["is_instability"]].copy()
    return sym, out


def build_instability_cache_parallel(dfs: dict, pairs: list, n_workers: int = 12):
    tasks = [(sym, dfs[sym]) for sym in pairs if sym in dfs and dfs[sym] is not None and not dfs[sym].empty]

    instability_cache = {}
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_worker_compute_instability, sym, df): sym for sym, df in tasks}

        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Instability x{n_workers}"):
            sym = futures[fut]
            try:
                sym2, out = fut.result()
                if out is not None:
                    instability_cache[sym2] = out
            except Exception as e:
                print(f"[{sym}] failed: {e}")

    t1 = time.time()
    print(f"Instability cache built for {len(instability_cache)}/{len(tasks)} symbols in {(t1-t0)/60:.2f} min")
    return instability_cache

def log_returns(close):
    return np.log(close).diff()

def price_entropy(close):
    net_move = abs(close.iloc[-1] - close.iloc[0])
    path_len = close.diff().abs().sum()
    return net_move / path_len if path_len > 0 else np.nan

def failed_micro_breakouts(close, high, low):
    hi = close.max()
    lo = close.min()
    rng = hi - lo
    if rng == 0:
        return 0

    upper = lo + 0.9 * rng
    lower = lo + 0.1 * rng

    failures = 0
    for i in range(1, len(close) - 1):
        if close.iloc[i] > upper and close.iloc[i + 1] < upper:
            failures += 1
        elif close.iloc[i] < lower and close.iloc[i + 1] > lower:
            failures += 1
    return failures

def extract_pre_shock_features(df, event_ts):
    """
    df: single-symbol OHLCV DataFrame
    event_ts: timestamp of shock (aligned to index)
    """

    try:
        event_loc = df.index.get_loc(event_ts)
    except KeyError:
        return None

    if event_loc < BASE_BARS:
        return None  # not enough history

    pre  = df.iloc[event_loc - PRE_BARS  : event_loc]
    base = df.iloc[event_loc - BASE_BARS : event_loc]

    # --- Returns
    r_pre  = log_returns(pre["close"])
    r_base = log_returns(base["close"])

    vol_ratio = r_pre.std() / r_base.std()

    # --- ATR proxy (true range)
    tr_pre  = (pre["high"] - pre["low"]).mean()
    tr_base = (base["high"] - base["low"]).mean()
    atr_ratio = tr_pre / tr_base

    # --- Entropy
    entropy = price_entropy(pre["close"])

    # --- Volume suppression
    vol_ratio_vol = pre["volume"].mean() / base["volume"].mean()

    # --- Failed breakouts
    failed = failed_micro_breakouts(
        pre["close"], pre["high"], pre["low"]
    )

    return {
        "vol_ratio": vol_ratio,
        "atr_ratio": atr_ratio,
        "entropy": entropy,
        "volume_ratio": vol_ratio_vol,
        "failed_breakouts": failed,
    }

def build_forbidden_mask(df_index, event_ts_list, pre_bars=PRE_BARS, post_bars=POST_EXCLUDE_BARS):
    n = len(df_index)
    forbid = np.zeros(n, dtype=bool)

    # positions of event timestamps in this df index
    # (event_ts should already be aligned; if not, nearest-match with method='pad' but that can introduce noise)
    locs = df_index.get_indexer(event_ts_list)
    locs = locs[locs >= 0]  # keep only those found

    for loc in locs:
        a = max(0, loc - pre_bars)
        b = min(n, loc + post_bars + 1)
        forbid[a:b] = True

    return forbid

def sample_non_event_timestamps(dfs, events, rng, per_symbol_cap=None):
    # group event timestamps by symbol
    ev_by_sym = events.groupby("symbol")["event_ts"].apply(list).to_dict()

    sampled = []  # list of dicts {symbol, event_ts}

    for sym, df in dfs.items():
        if sym not in ev_by_sym:
            continue

        idx = df.index
        if len(idx) <= BASE_BARS + PRE_BARS + 10:
            continue

        event_ts_list = ev_by_sym[sym]
        forbid = build_forbidden_mask(idx, event_ts_list)

        # valid candidates must:
        # - have 72h baseline available => loc >= BASE_BARS
        # - not be forbidden
        valid_mask = np.ones(len(idx), dtype=bool)
        valid_mask[:BASE_BARS] = False
        valid_mask &= ~forbid

        valid_locs = np.flatnonzero(valid_mask)
        if len(valid_locs) == 0:
            continue

        # target sample count = number of shocks for this symbol (or cap)
        k = len(event_ts_list)
        if per_symbol_cap is not None:
            k = min(k, per_symbol_cap)

        k = min(k, len(valid_locs))
        chosen_locs = rng.choice(valid_locs, size=k, replace=False)

        for loc in chosen_locs:
            sampled.append({"symbol": sym, "event_ts": idx[loc]})

    return pd.DataFrame(sampled)

def find_pre_shock_entry(df, event_ts):
    event_loc = df.index.get_loc(event_ts)

    pre = df.iloc[event_loc - PRE_BARS : event_loc]
    atr_pre = (pre["high"] - pre["low"]).mean()

    # compression range (last 2h)
    comp = pre.iloc[-RANGE_BARS:]
    range_high = comp["high"].max()
    range_low  = comp["low"].min()

    # scan bars after compression range
    # scan = pre.iloc[RANGE_BARS:]
    MAX_LEAD_BARS = 36   # 3 hours
    scan = pre.iloc[-MAX_LEAD_BARS:]


    # for ts, row in scan.iterrows():
    for ts, row in scan.iloc[::-1].iterrows():  # newest -> oldest
        body = abs(row["close"] - row["open"])

        # long
        if row["close"] > range_high and body > 0.75 * atr_pre:
            return {
                "entry_ts": ts,
                "direction": "long",
                "entry_price": row["close"]
            }

        # short
        if row["close"] < range_low and body > 0.75 * atr_pre:
            return {
                "entry_ts": ts,
                "direction": "short",
                "entry_price": row["close"]
            }

    return None

def find_pre_shock_entry_v2(
    df,
    event_ts,
    *,
    PRE_BARS: int = 144,
    RANGE_BARS: int = 24,
    MAX_LEAD_BARS: int = 36,

    # Primary "delay naturally" knob (start small!)
    BREAK_ATR_BUFFER: float = 0.10,   # try 0.05 -> 0.15 -> 0.20

    # keep your old impulse logic, but allow tuning
    BODY_ATR_MULT: float = 0.75,

    # optional quality filter (start OFF)
    USE_STRONG_CLOSE: bool = False,
    STRONG_CLOSE_Q: float = 0.70,

    # optional volume filter (start OFF)
    USE_RVOL: bool = False,
    RVOL_WIN: int = 144,
    RVOL_MIN: float = 1.05,
    RVOL_RISING: bool = False,

    allow_short: bool = True,
):
    """
    Balanced v2:
      - scans newest->oldest (naturally later)
      - adds small breakout buffer (naturally later)
      - optional strong-close / RVOL gates (off by default)
    """
    import numpy as np
    import pandas as pd

    if df is None or df.empty:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("find_pre_shock_entry_v2: df must have a DatetimeIndex")

    df = df.sort_index()
    event_ts = pd.Timestamp(event_ts)

    # snap to previous bar if not exact
    if event_ts not in df.index:
        loc = df.index.searchsorted(event_ts, side="right") - 1
        if loc < 0:
            return None
        event_ts = df.index[loc]

    event_loc = df.index.get_loc(event_ts)
    if event_loc < (PRE_BARS + 5):
        return None

    pre = df.iloc[event_loc - PRE_BARS : event_loc].copy()
    if len(pre) < max(PRE_BARS, RANGE_BARS, MAX_LEAD_BARS):
        return None

    # same atr_pre proxy you already used
    atr_pre = (pre["high"].astype(float) - pre["low"].astype(float)).mean()
    if not np.isfinite(atr_pre) or atr_pre <= 0:
        return None

    comp = pre.iloc[-RANGE_BARS:]
    range_high = float(comp["high"].max())
    range_low  = float(comp["low"].min())

    scan = pre.iloc[-MAX_LEAD_BARS:].copy()
    if scan.empty:
        return None

    # optional RVOL
    if USE_RVOL:
        vol_med = pre["volume"].astype(float).rolling(RVOL_WIN).median()
        scan["rvol"] = scan["volume"].astype(float) / vol_med.reindex(scan.index)
    else:
        scan["rvol"] = np.nan

    up_level = range_high + BREAK_ATR_BUFFER * atr_pre
    dn_level = range_low  - BREAK_ATR_BUFFER * atr_pre

    # newest -> oldest (THIS is your "natural delay" #1)
    for i in range(len(scan) - 1, -1, -1):
        row = scan.iloc[i]
        ts  = scan.index[i]

        hi = float(row["high"]); lo = float(row["low"])
        op = float(row["open"]); cl = float(row["close"])

        body = abs(cl - op)

        # impulse gate (same idea as old function)
        if body <= BODY_ATR_MULT * atr_pre:
            continue

        rng = max(hi - lo, 1e-12)
        close_pos = (cl - lo) / rng  # 0..1

        # ---- LONG candidate ----
        if cl > up_level:
            if USE_STRONG_CLOSE and (close_pos < STRONG_CLOSE_Q):
                continue

            if USE_RVOL:
                rvol = float(row.get("rvol", np.nan))
                if not (np.isfinite(rvol) and rvol >= RVOL_MIN):
                    continue
                if RVOL_RISING and i >= 2:
                    r0 = float(scan.iloc[i]["rvol"])
                    r1 = float(scan.iloc[i - 1]["rvol"])
                    r2 = float(scan.iloc[i - 2]["rvol"])
                    if not (np.isfinite(r0) and np.isfinite(r1) and np.isfinite(r2) and (r0 > r1 > r2)):
                        continue

            return {"entry_ts": ts, "direction": "long", "entry_price": cl}

        # ---- SHORT candidate ----
        if allow_short and (cl < dn_level):
            if USE_STRONG_CLOSE and (close_pos > (1.0 - STRONG_CLOSE_Q)):
                continue

            if USE_RVOL:
                rvol = float(row.get("rvol", np.nan))
                if not (np.isfinite(rvol) and rvol >= RVOL_MIN):
                    continue
                if RVOL_RISING and i >= 2:
                    r0 = float(scan.iloc[i]["rvol"])
                    r1 = float(scan.iloc[i - 1]["rvol"])
                    r2 = float(scan.iloc[i - 2]["rvol"])
                    if not (np.isfinite(r0) and np.isfinite(r1) and np.isfinite(r2) and (r0 > r1 > r2)):
                        continue

            return {"entry_ts": ts, "direction": "short", "entry_price": cl}

    return None



def compute_shock_direction(row, dfs):
    sym = row["symbol"]
    event_ts = row["event_ts"]

    df = dfs[sym]

    try:
        p0 = df.loc[event_ts - pd.Timedelta(minutes=60), "close"]
        p1 = df.loc[event_ts, "close"]
    except KeyError:
        return np.nan

    ret = (p1 - p0) / p0
    return "long" if ret > 0 else "short"

def summarize(trades):
    return {
        "trades": len(trades),
        "win_rate": (trades["ret_net"] > 0).mean(),
        "avg_ret": trades["ret_net"].mean(),
        "median_ret": trades["ret_net"].median(),
        "total_ret": trades["ret_net"].sum(),
        "avg_win": trades.loc[trades["ret_net"] > 0, "ret_net"].mean(),
        "avg_loss": trades.loc[trades["ret_net"] <= 0, "ret_net"].mean(),
    }

def post_shock_stats_for_trade(sym, entry_ts, event_ts, direction, dfs, H):
    df = dfs[sym]
    exit_start = event_ts + pd.Timedelta(minutes=BAR_MIN)  # we avoid same-bar
    if entry_ts not in df.index or exit_start not in df.index:
        return None

    entry_px = float(df.loc[entry_ts, "close"])

    loc0 = df.index.get_loc(exit_start)
    if loc0 + H >= len(df):
        return None

    path = df["close"].iloc[loc0:loc0+H+1].astype(float).values
    # long-only assumed, but keep direction for safety
    if direction == "long":
        rets = (path - entry_px) / entry_px
    else:
        rets = (entry_px - path) / entry_px

    mfe = float(np.max(rets))
    mae = float(np.min(rets))
    ttp = int(np.argmax(rets))  # bars to peak

    ret_end = float(rets[-1])
    return mfe, mae, ttp, ret_end

import numpy as np
import pandas as pd

def ATR(df, n=14):
    """
    Simple ATR (rolling mean of True Range)
    Assumes df has columns: high, low, close
    """
    high = df["high"]
    low  = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)

    return tr.rolling(n).mean()


def trailing_params_from_compression(comp_score):
    if comp_score <= -4.5:  # deep
        return dict(
            ACTIVATE_ATR=2.2, TRAIL_RETRACE=0.45, DEAD_BARS=42, DEAD_MFE_ATR=1.2,
            A_BARS=12, A_MFE_ATR=0.20, A_USE_EARLY_STOP=False
        )
    elif comp_score <= -3.8:  # Q2
        return dict(
            ACTIVATE_ATR=1.8, TRAIL_RETRACE=0.40, DEAD_BARS=30, DEAD_MFE_ATR=1.0,
            A_BARS=6, A_MFE_ATR=0.35, A_USE_EARLY_STOP=True
        )
    elif comp_score <= -3.2:  # Q3
        return dict(
            ACTIVATE_ATR=1.5, TRAIL_RETRACE=0.35, DEAD_BARS=24, DEAD_MFE_ATR=0.9,
            A_BARS=4, A_MFE_ATR=0.45, A_USE_EARLY_STOP=True
        )
    else:  # shallow
        return dict(
            ACTIVATE_ATR=1.3, TRAIL_RETRACE=0.30, DEAD_BARS=18, DEAD_MFE_ATR=0.8,
            A_BARS=3, A_MFE_ATR=0.60, A_USE_EARLY_STOP=True
        )

def simulate_long_exit(
    df, entry_ts, event_ts, entry_px,
    compression_score,                 # REQUIRED (for compression-aware trailing)
    H_MAX=72, BAR_MIN=5, FEE=0.001,
    ATR_N=14,
    STOP_ATR=1.2,
    TIME_STOP_BARS=72,

    # ------------------------
    # Phase A: validation gate (defaults; may be overridden by compression)
    # ------------------------
    A_BARS=3,              # validate within first N bars after event+1
    A_MFE_ATR=0.6,         # must achieve at least this MFE (ATR units)
    A_EARLY_STOP_ATR=1.0,  # early stop during Phase A (ATR units)
    A_USE_EARLY_STOP=False,

    # ------------------------
    # New: compression-aware policy knobs
    # ------------------------
    DEEP_SCORE_TH=-4.0,          # treat <= as "deep"
    DEEP_A_BARS=12,              # give deep compressions more time in Phase A
    DEEP_A_MFE_ATR=0.20,         # require only a small initial pop
    DEEP_DISABLE_EARLY_STOP=True,# disable A_EARLY_STOP for deep compressions

    # New: use fixed ATR reference for activation/dead thresholds
    USE_ATR0_FOR_THRESHOLDS=True,
):
    """
    Long-only post-shock exit simulator:
      - Start management at event_ts + 1 bar
      - Phase A gate (validation) [now compression-aware]
      - Compression-aware trailing activation + giveback
      - Hard stop, dead trade rule, time cap
    """

    # 0) start management from event_ts + 1 bar
    df = df.sort_index()
    start_ts = event_ts + pd.Timedelta(minutes=BAR_MIN)

    if df.empty:
        return None
    if entry_ts not in df.index:
        return None
    if start_ts not in df.index:
        return None

    loc0 = df.index.get_loc(start_ts)

    # ATR series
    atr = ATR(df, n=ATR_N)
    if loc0 >= len(atr) or not np.isfinite(atr.iloc[loc0]):
        return None

    atr0 = float(atr.iloc[loc0])

    # Compression-aware parameters (from your mapping)
    params = trailing_params_from_compression(float(compression_score))
    ACTIVATE_ATR  = float(params["ACTIVATE_ATR"])
    TRAIL_RETRACE = float(params["TRAIL_RETRACE"])
    DEAD_BARS     = int(params["DEAD_BARS"])
    DEAD_MFE_ATR  = float(params["DEAD_MFE_ATR"])

    # ------------------------
    # Phase A: compression-aware overrides
    # ------------------------
    is_deep = float(compression_score) <= float(DEEP_SCORE_TH)

    if is_deep:
        # Deep compressions in your filtered Feb–Sep window are getting killed in Phase A.
        A_BARS = max(int(A_BARS), int(DEEP_A_BARS))
        A_MFE_ATR = min(float(A_MFE_ATR), float(DEEP_A_MFE_ATR))
        if DEEP_DISABLE_EARLY_STOP:
            A_USE_EARLY_STOP = False

        # Also make trailing less "twitchy" for deep compressions (local safety clamp).
        # These are conservative, not extreme.
        ACTIVATE_ATR = min(ACTIVATE_ATR, 0.9)
        TRAIL_RETRACE = max(TRAIL_RETRACE, 0.55)
        DEAD_BARS = max(DEAD_BARS, 72)
        DEAD_MFE_ATR = min(DEAD_MFE_ATR, 0.5)

    # Hard stop (static from ATR0)
    # hard_stop_px = entry_px - STOP_ATR * atr0
    if is_deep:
        hard_stop_px = entry_px - (STOP_ATR * 1.5) * atr0
    else:
        hard_stop_px = entry_px - STOP_ATR * atr0


    # Helpers for converting ATR-units -> return threshold
    def _atr_ret(atr_value: float, atr_mult: float) -> float:
        # ATR * mult expressed as fraction of entry price (return space)
        return (atr_mult * atr_value) / entry_px

    # ------------------------
    # Phase A: validation gate
    # ------------------------
    a_end_loc = min(loc0 + int(A_BARS) - 1, len(df) - 1)

    mfe_A = -np.inf
    mae_A = np.inf

    for loc in range(loc0, a_end_loc + 1):
        close = float(df["close"].iloc[loc])
        r = (close - entry_px) / entry_px
        mfe_A = max(mfe_A, r)
        mae_A = min(mae_A, r)

        # Optional early stop during Phase A using ATR0
        if A_USE_EARLY_STOP and (r <= -_atr_ret(atr0, A_EARLY_STOP_ATR)):
            exit_ts = df.index[loc]
            exit_px = close
            ret = r
            ret_net = ret - 2 * FEE
            return {
                "exit_ts": exit_ts,
                "exit_px": exit_px,
                "ret": float(ret),
                "ret_net": float(ret_net),
                "mfe": float(max(0.0, mfe_A)),
                "mae": float(mae_A),
                "exit_reason": "A_EARLY_STOP",
                "trail_active": False,
                "mfe_A": float(mfe_A),
                "mae_A": float(mae_A),
                "ACTIVATE_ATR": float(ACTIVATE_ATR),
                "TRAIL_RETRACE": float(TRAIL_RETRACE),
                "DEAD_BARS": int(DEAD_BARS),
                "DEAD_MFE_ATR": float(DEAD_MFE_ATR),
            }

    # Must show minimum pop by end of Phase A
    if mfe_A < _atr_ret(atr0, A_MFE_ATR):
        if not is_deep:
            exit_ts = df.index[a_end_loc]
            exit_px = float(df["close"].iloc[a_end_loc])
            ret = (exit_px - entry_px) / entry_px
            ret_net = ret - 2 * FEE
            return {
                "exit_ts": exit_ts,
                "exit_px": float(exit_px),
                "ret": float(ret),
                "ret_net": float(ret_net),
                "mfe": float(max(0.0, mfe_A)),
                "mae": float(mae_A),
                "exit_reason": "A_FAIL_NO_POP",
                "trail_active": False,
                "mfe_A": float(mfe_A),
                "mae_A": float(mae_A),
                "ACTIVATE_ATR": float(ACTIVATE_ATR),
                "TRAIL_RETRACE": float(TRAIL_RETRACE),
                "DEAD_BARS": int(DEAD_BARS),
                "DEAD_MFE_ATR": float(DEAD_MFE_ATR),
            }
        else:
            # deep compressions: do NOT fail early
            pass

    # If pass Phase A, continue from next bar
    loc_after_A = min(a_end_loc + 1, len(df) - 1)

    # 1) state
    mfe = max(0.0, float(mfe_A))
    mae = float(mae_A) if np.isfinite(mae_A) else 0.0
    trail_active = False
    trail_peak = None  # peak AFTER activation

    # 2) iterate forward
    max_loc = min(loc0 + int(H_MAX), len(df) - 1)

    exit_ts = None
    exit_px = None
    exit_reason = None

    for loc in range(loc_after_A, max_loc + 1):
        ts = df.index[loc]
        close = float(df["close"].iloc[loc])

        ret_from_entry = (close - entry_px) / entry_px
        mfe = max(mfe, ret_from_entry)
        mae = min(mae, ret_from_entry)

        bars_since_start = loc - loc0

        # A) Hard stop
        # Soft time decay exit (prevents hard stop damage)
        if bars_since_start >= int(0.5 * TIME_STOP_BARS):
            if(not trail_active) and (mfe < 0.4 * _atr_ret(atr0, ACTIVATE_ATR)):
                exit_ts = ts
                exit_px = close
                exit_reason = "SOFT_TIME_EXIT"
                break
        # Emergency stop (catastrophic only)
        em_mult = 8.0 if is_deep else 6.0
        if close <= entry_px - em_mult * atr0:
            exit_ts = ts
            exit_px = close
            exit_reason = "EMERGENCY_STOP"
            break

        # if close <= hard_stop_px:
        #     exit_ts = ts
        #     exit_px = close
        #     exit_reason = "HARD_STOP"
        #     break

        # ATR per bar (fallback to atr0)
        atr_loc = atr.iloc[loc] if loc < len(atr) else np.nan
        atr_now = float(atr_loc) if np.isfinite(atr_loc) else atr0

        # Choose threshold ATR reference
        thr_atr = atr0 if USE_ATR0_FOR_THRESHOLDS else atr_now

        # B) Activate trailing after compression-aware expansion
        if (not trail_active) and (mfe >= _atr_ret(thr_atr, ACTIVATE_ATR)):
            trail_active = True
            trail_peak = mfe  # start peak at activation

        # Update trailing peak only AFTER activation
        if trail_active:
            if trail_peak is None:
                trail_peak = ret_from_entry
            else:
                trail_peak = max(trail_peak, ret_from_entry)

        # C) Compression-aware dead trade rule
        if bars_since_start >= int(DEAD_BARS):
            if mfe < _atr_ret(thr_atr, DEAD_MFE_ATR):
                exit_ts = ts
                exit_px = close
                exit_reason = "DEAD_TIME"
                break

        # D) Compression-aware trailing giveback
        if trail_active and (trail_peak is not None) and (trail_peak > 0):
            if ret_from_entry <= (1.0 - float(TRAIL_RETRACE)) * float(trail_peak):
                exit_ts = ts
                exit_px = close
                exit_reason = "TRAIL_GIVEBACK"
                break

        # E) Time cap
        if bars_since_start >= int(TIME_STOP_BARS):
            exit_ts = ts
            exit_px = close
            exit_reason = "TIME_CAP"
            break

    # If never exited in loop
    if exit_ts is None:
        exit_ts = df.index[max_loc]
        exit_px = float(df["close"].iloc[max_loc])
        exit_reason = "END_OF_WINDOW"

    ret = (exit_px - entry_px) / entry_px
    ret_net = ret - 2 * FEE

    return {
        "exit_ts": exit_ts,
        "exit_px": float(exit_px),
        "ret": float(ret),
        "ret_net": float(ret_net),
        "mfe": float(mfe),
        "mae": float(mae),
        "exit_reason": str(exit_reason),
        "trail_active": bool(trail_active),
        "mfe_A": float(mfe_A),
        "mae_A": float(mae_A),
        "ACTIVATE_ATR": float(ACTIVATE_ATR),
        "TRAIL_RETRACE": float(TRAIL_RETRACE),
        "DEAD_BARS": int(DEAD_BARS),
        "DEAD_MFE_ATR": float(DEAD_MFE_ATR),
    }


def simulate_long_exit_ca(
    df, entry_ts, event_ts, entry_px,
    compression_score,
    BAR_MIN=5, FEE=0.001,
    ATR_N=14,
):
    """
    Compression-aware long exit.
    Starts managing at event_ts + 1 bar.
    Exits on: HARD_STOP, FAIL_NO_POP, EARLY_STOP, TRAIL_GIVEBACK, TIME_CAP, END_OF_WINDOW
    """

    # 0) Pull compression-aware parameters
    P = exit_policy_from_score(float(compression_score) if compression_score is not None else np.nan)
    STOP_ATR        = P["STOP_ATR"]
    ACTIVATE_ATR    = P["ACTIVATE_ATR"]
    TRAIL_RETRACE   = P["TRAIL_RETRACE"]
    DEAD_BARS       = P["DEAD_BARS"]
    DEAD_MFE_ATR    = P["DEAD_MFE_ATR"]
    TIME_STOP_BARS  = P["TIME_STOP_BARS"]
    H_MAX           = P["H_MAX"]

    start_ts = event_ts + pd.Timedelta(minutes=BAR_MIN)
    if entry_ts not in df.index: return None
    if start_ts not in df.index: return None

    loc0 = df.index.get_loc(start_ts)

    atr = ATR(df, n=ATR_N)
    if loc0 >= len(atr) or (not np.isfinite(atr.iloc[loc0])):
        return None

    atr0 = float(atr.iloc[loc0])
    hard_stop_px = entry_px - STOP_ATR * atr0  # price units

    # State
    mfe = 0.0
    mae = 0.0
    trail_active = False
    exit_ts = None
    exit_px = None
    exit_reason = None

    max_loc = min(loc0 + int(H_MAX), len(df) - 1)

    # To help Phase-A validation: detect "no pop" early
    # We'll define "pop" as reaching +DEAD_MFE_ATR * ATR within DEAD_BARS
    popped = False

    for loc in range(loc0, max_loc + 1):
        ts = df.index[loc]
        close = float(df["close"].iloc[loc])

        ret_from_entry = (close - entry_px) / entry_px
        mfe = max(mfe, ret_from_entry)
        mae = min(mae, ret_from_entry)

        atr_now = float(atr.iloc[loc]) if np.isfinite(atr.iloc[loc]) else atr0

        # A) Hard stop
        if close <= hard_stop_px:
            exit_ts, exit_px, exit_reason = ts, close, "HARD_STOP"
            break

        bars_since_start = loc - loc0

        # "pop" detection in ATR terms
        if (not popped) and (mfe >= (DEAD_MFE_ATR * atr_now / entry_px)):
            popped = True

        # B) If after DEAD_BARS we still didn't pop → FAIL_NO_POP
        if bars_since_start >= DEAD_BARS and (not popped):
            exit_ts, exit_px, exit_reason = ts, close, "A_FAIL_NO_POP"
            break

        # C) Activate trailing only after real expansion
        if (not trail_active) and (mfe >= (ACTIVATE_ATR * atr_now / entry_px)):
            trail_active = True

        # D) Once popped but fails to progress → EARLY_STOP
        # (Example rule: after pop, if we give back too much before trail activates)
        # Keep this mild to avoid over-pruning; tune later.
        if (not trail_active) and popped and bars_since_start >= DEAD_BARS:
            # if after pop we are back near flat, kill it
            if ret_from_entry <= 0.2 * (DEAD_MFE_ATR * atr_now / entry_px):
                exit_ts, exit_px, exit_reason = ts, close, "A_EARLY_STOP"
                break

        # E) Trailing giveback
        if trail_active and mfe > 0:
            if ret_from_entry <= (1.0 - TRAIL_RETRACE) * mfe:
                exit_ts, exit_px, exit_reason = ts, close, "TRAIL_GIVEBACK"
                break

        # F) Time cap
        if bars_since_start >= TIME_STOP_BARS:
            exit_ts, exit_px, exit_reason = ts, close, "TIME_CAP"
            break

    if exit_ts is None:
        exit_ts = df.index[max_loc]
        exit_px = float(df["close"].iloc[max_loc])
        exit_reason = "END_OF_WINDOW"

    ret = (exit_px - entry_px) / entry_px
    ret_net = ret - 2*FEE

    return dict(
        exit_ts=exit_ts,
        exit_px=exit_px,
        ret=ret,
        ret_net=ret_net,
        mfe=mfe,
        mae=mae,
        exit_reason=exit_reason,
        trail_active=trail_active,
        # audit params
        p_STOP_ATR=STOP_ATR,
        p_ACTIVATE_ATR=ACTIVATE_ATR,
        p_TRAIL_RETRACE=TRAIL_RETRACE,
        p_DEAD_BARS=DEAD_BARS,
        p_DEAD_MFE_ATR=DEAD_MFE_ATR,
        p_TIME_STOP_BARS=TIME_STOP_BARS,
        p_H_MAX=H_MAX,
    )

def compare_vs_baseline(trades_base, trades_v5, ret_col="ret_net"):
    keys = ["symbol", "entry_ts", "event_ts"]

    base_keep = keys + [ret_col]
    v5_keep   = keys + [ret_col]

    # bring these along if they exist (especially from v5)
    if "comp_bucket" in trades_base.columns: base_keep.append("comp_bucket")
    if "comp_bucket" in trades_v5.columns:   v5_keep.append("comp_bucket")

    if "exit_reason" in trades_base.columns: base_keep.append("exit_reason")
    if "exit_reason" in trades_v5.columns:   v5_keep.append("exit_reason")

    if "compression_score" in trades_base.columns: base_keep.append("compression_score")
    if "compression_score" in trades_v5.columns:   v5_keep.append("compression_score")

    b = trades_base[base_keep].copy()
    v = trades_v5[v5_keep].copy()

    # normalize timestamps (prevents silent mismatches)
    for df_ in (b, v):
        df_["entry_ts"] = pd.to_datetime(df_["entry_ts"]).dt.tz_localize(None)
        df_["event_ts"] = pd.to_datetime(df_["event_ts"]).dt.tz_localize(None)

    cmp = b.merge(v, on=keys, how="inner", suffixes=("_base", "_v5"))

    cmp["ret_delta"] = cmp[f"{ret_col}_v5"] - cmp[f"{ret_col}_base"]
    cmp["improved"] = cmp["ret_delta"] > 0
    return cmp



def summarize_cmp(cmp: pd.DataFrame, ret_col="ret_net") -> dict:
    rb = f"{ret_col}_base"
    rv = f"{ret_col}_v5"

    overall = {
        "n": int(len(cmp)),
        "mean_base": float(cmp[rb].mean()),
        "mean_v5": float(cmp[rv].mean()),
        "mean_delta": float(cmp["ret_delta"].mean()),
        "median_delta": float(cmp["ret_delta"].median()),
        "winrate_base": float((cmp[rb] > 0).mean()),
        "winrate_v5": float((cmp[rv] > 0).mean()),
        "pct_improved": float(cmp["improved"].mean()),
    }

    # column detection
    bucket_col = next((c for c in ["comp_bucket_v5","comp_bucket"] if c in cmp.columns), None)
    reason_col = next((c for c in ["exit_reason_v5","exit_reason"] if c in cmp.columns), None)

    by_bucket = None
    if bucket_col:
        by_bucket = (cmp.groupby(bucket_col)["ret_delta"]
                       .agg(count="count", mean="mean", median="median",
                            win_rate=lambda s: (s > 0).mean())
                       .sort_values("count", ascending=False))

    by_reason = None
    if reason_col:
        by_reason = (cmp.groupby(reason_col)[rv]
                       .agg(count="count", mean="mean", median="median",
                            win_rate=lambda s: (s > 0).mean())
                       .sort_values("count", ascending=False))

    return {"overall": overall, "by_bucket": by_bucket, "by_reason": by_reason}

def simulate_portfolio_from_trades(
    trades_ref: pd.DataFrame,
    *,
    max_open: int = 5,
    cooldown_bars: int = 12,
    bar_minutes: int = 5,
    bucket_allow = ("Q1 (deep)",),   # start robust: Q1 only
    rank_by = ("compression_score",),# deepest first (more negative wins)
    initial_equity: float = 1.0,
):
    # keep only valid rows
    t = trades_ref.copy()
    t = t[t["ok"].fillna(True)] if "ok" in t.columns else t
    t = t.dropna(subset=["entry_ts","exit_ts","ret_net","symbol"])
    t["entry_ts"] = pd.to_datetime(t["entry_ts"])
    t["exit_ts"]  = pd.to_datetime(t["exit_ts"])

    # optional filter by bucket
    if "comp_bucket" in t.columns and bucket_allow is not None:
        t = t[t["comp_bucket"].isin(bucket_allow)]

    # sort entry stream
    sort_cols = ["entry_ts"] + list(rank_by)
    t = t.sort_values(sort_cols, ascending=[True] + [True]*len(rank_by))  # compression_score smaller = better

    cooldown_td = pd.Timedelta(minutes=cooldown_bars * bar_minutes)

    open_positions = []  # list of dicts: {symbol, exit_ts, notional, entry_ts}
    last_exit = {}       # symbol -> last exit_ts
    cash = initial_equity
    equity = initial_equity

    # event log for analysis
    taken = []
    skipped = []

    def close_positions(up_to_ts):
        nonlocal cash, equity, open_positions
        still_open = []
        for p in open_positions:
            if p["exit_ts"] <= up_to_ts:
                # realize PnL
                pnl = p["notional"] * p["ret_net"]
                cash += p["notional"] + pnl
            else:
                still_open.append(p)
        open_positions = still_open
        equity = cash + sum(p["notional"] for p in open_positions)  # mark-to-cash (no MTM); simple

    for r in t.itertuples(index=False):
        entry_ts = r.entry_ts
        exit_ts  = r.exit_ts
        sym      = r.symbol

        # close positions that have exited before this entry
        close_positions(entry_ts)

        # enforce one-per-symbol
        if any(p["symbol"] == sym for p in open_positions):
            skipped.append((sym, entry_ts, "SYMBOL_ALREADY_OPEN"))
            continue

        # enforce cooldown
        if sym in last_exit and entry_ts < (last_exit[sym] + cooldown_td):
            skipped.append((sym, entry_ts, "COOLDOWN"))
            continue

        # enforce max_open
        if len(open_positions) >= max_open:
            skipped.append((sym, entry_ts, "MAX_OPEN"))
            continue

        # allocate fixed notional per position: equity / max_open (simple + stable)
        # if you prefer “use all cash equally among remaining slots”, tell me.
        notional = equity / max_open
        if cash < notional:
            skipped.append((sym, entry_ts, "INSUFFICIENT_CASH"))
            continue

        # open position
        cash -= notional
        open_positions.append({
            "symbol": sym,
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "notional": float(notional),
            "ret_net": float(r.ret_net),
            "compression_score": float(getattr(r, "compression_score", np.nan)),
            "comp_bucket": getattr(r, "comp_bucket", None),
            "exit_reason": getattr(r, "exit_reason", None),
        })

        taken.append(sym)

        # record last_exit now (so overlapping signals before exit are blocked)
        last_exit[sym] = exit_ts

    # close remaining at end
    if len(t):
        close_positions(pd.Timestamp.max)

    # build results
    out = {
        "final_equity": equity,
        "total_return": equity / initial_equity - 1.0,
        "trades_taken": len(taken),
        "trades_skipped": len(skipped),
        "skip_reasons": pd.Series([s[2] for s in skipped]).value_counts() if skipped else pd.Series(dtype=int),
    }
    return out

def simulate_portfolio_v1(
    trades_ref: pd.DataFrame,
    *,
    max_open: int = 5,
    cooldown_bars: int = 12,
    bar_minutes: int = 5,
    bucket_allow=("Q1 (deep)",),          # set None to allow all
    rank_cols=("compression_score",),     # ascending: most negative first
    initial_equity: float = 1.0,
    notional_mode: str = "use_all_cash",  # "fixed_slot" or "use_all_cash"
    slip_bps: float = 0.0,                # optional: extra bps per round-trip subtracted from ret_net
    debug: bool = False,
):
    t = trades_ref.copy()
    max_per_entry_ts = 2

    need = {"symbol","entry_ts","exit_ts","ret_net"}
    missing = need - set(t.columns)
    if missing:
        raise ValueError(f"trades_ref missing columns: {missing}")

    t["entry_ts"] = pd.to_datetime(t["entry_ts"]).dt.tz_localize(None)
    t["exit_ts"]  = pd.to_datetime(t["exit_ts"]).dt.tz_localize(None)

    if bucket_allow is not None and "comp_bucket" in t.columns:
        t = t[t["comp_bucket"].isin(bucket_allow)].copy()

    if slip_bps and slip_bps > 0:
        slip = slip_bps / 1e4
        t["ret_net_adj"] = t["ret_net"] - slip
        ret_col = "ret_net_adj"
    else:
        ret_col = "ret_net"

    t = t.sort_values(["entry_ts"] + list(rank_cols), ascending=[True] + [True]*len(rank_cols))
    cooldown_td = pd.Timedelta(minutes=cooldown_bars * bar_minutes)

    cash = float(initial_equity)
    equity = float(initial_equity)

    open_pos = {}   # sym -> position dict
    last_exit = {}  # sym -> last exit_ts

    taken_rows = []
    skipped_rows = []

    def close_up_to(ts_now):
        nonlocal cash, equity, open_pos
        to_close = [sym for sym, p in open_pos.items() if p["exit_ts"] <= ts_now]
        for sym in to_close:
            p = open_pos.pop(sym)
            pnl = p["notional"] * p[ret_col]
            cash += p["notional"] + pnl
            last_exit[sym] = p["exit_ts"]
        equity = cash + sum(p["notional"] for p in open_pos.values())

    if debug:
        print("DEBUG notional_mode:", notional_mode)

    for entry_ts, batch in t.groupby("entry_ts", sort=True):
        close_up_to(entry_ts)

        slots = max_open - len(open_pos)
        if slots <= 0:
            for r in batch.itertuples(index=False):
                skipped_rows.append((r.symbol, entry_ts, "MAX_OPEN"))
            continue

        candidates = []
        for r in batch.itertuples(index=False):
            sym = r.symbol

            if sym in open_pos:
                skipped_rows.append((sym, entry_ts, "SYMBOL_ALREADY_OPEN"))
                continue

            if sym in last_exit and entry_ts < (last_exit[sym] + cooldown_td):
                skipped_rows.append((sym, entry_ts, "COOLDOWN"))
                continue

            candidates.append(r)

        if not candidates:
            continue

        cand_df = pd.DataFrame([c._asdict() for c in candidates])
        cand_df = cand_df.sort_values(list(rank_cols), ascending=[True]*len(rank_cols))

        slots_left = slots
        for _, row in cand_df.head(min(slots, max_per_entry_ts)).iterrows():
            sym = row["symbol"]
            exit_ts = row["exit_ts"]

            if notional_mode == "fixed_slot":
                notional = initial_equity / max_open
            elif notional_mode == "use_all_cash":
                notional = cash / slots_left if slots_left > 0 else 0.0
            else:
                raise ValueError("notional_mode must be 'fixed_slot' or 'use_all_cash'")

            if debug:
                print("DEBUG", entry_ts, sym, "cash", cash, "notional", notional, "open", len(open_pos), "slots_left", slots_left)

            if notional <= 0:
                skipped_rows.append((sym, entry_ts, "BAD_NOTIONAL"))
                continue

            if notional_mode == "use_all_cash":
                if cash < notional:
                    skipped_rows.append((sym, entry_ts, "INSUFFICIENT_CASH"))
                    continue
                cash -= notional
            else:
                cash -= notional  # fixed-slot: treat slots as reserved

            open_pos[sym] = {
                "symbol": sym,
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "notional": float(notional),
                "ret_net": float(row["ret_net"]),
                "ret_net_adj": float(row.get("ret_net_adj", np.nan)) if "ret_net_adj" in row else np.nan,
                "comp_bucket": row.get("comp_bucket", None),
                "compression_score": float(row.get("compression_score", np.nan)),
                "exit_reason": row.get("exit_reason", None),
                ret_col: float(row[ret_col]),
            }

            taken_rows.append({
                "symbol": sym,
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "notional": float(notional),
                "ret_net_used": float(row[ret_col]),
                "comp_bucket": row.get("comp_bucket", None),
                "compression_score": float(row.get("compression_score", np.nan)),
                "exit_reason": row.get("exit_reason", None),
            })

            slots_left -= 1

        if len(cand_df) > slots:
            for sym in cand_df.iloc[slots:]["symbol"].tolist():
                skipped_rows.append((sym, entry_ts, "MAX_OPEN_BATCH"))

    close_up_to(pd.Timestamp.max)

    result = {
        "final_equity": float(equity),
        "total_return": float(equity / initial_equity - 1.0),
        "trades_taken": int(len(taken_rows)),
        "trades_skipped": int(len(skipped_rows)),
        "skip_reasons": pd.Series([x[2] for x in skipped_rows]).value_counts() if skipped_rows else pd.Series(dtype=int),
    }

    logs = {
        "taken": pd.DataFrame(taken_rows),
        "skipped": pd.DataFrame(skipped_rows, columns=["symbol","entry_ts","reason"]) if skipped_rows else pd.DataFrame(columns=["symbol","entry_ts","reason"]),
    }

    return result, logs

def run_portfolio_sweep(trades_ref):
    grid = {
        "bucket_allow": [
            ("Q1 (deep)",),
            ("Q1 (deep)","Q2"),
            None,  # all
        ],
        "max_open": [3,5,8],
        "cooldown_bars": [6,12,24],
        "slip_bps": [0, 5, 10],
    }

    rows = []
    for bucket_allow, max_open, cooldown_bars, slip_bps in itertools.product(
        grid["bucket_allow"], grid["max_open"], grid["cooldown_bars"], grid["slip_bps"]
    ):
        res, logs = simulate_portfolio_v1(
            trades_ref,
            bucket_allow=bucket_allow,
            max_open=max_open,
            cooldown_bars=cooldown_bars,
            slip_bps=slip_bps,
            notional_mode="use_all_cash",
            rank_cols=("compression_score",),
            debug=False,
        )
        rows.append({
            "bucket_allow": "ALL" if bucket_allow is None else "+".join(bucket_allow),
            "max_open": max_open,
            "cooldown_bars": cooldown_bars,
            "slip_bps": slip_bps,
            **{k: res[k] for k in ["final_equity","total_return","trades_taken","trades_skipped"]},
            "max_open_skips": int(res["skip_reasons"].get("MAX_OPEN", 0)) if "skip_reasons" in res else 0,
            "max_open_batch_skips": int(res["skip_reasons"].get("MAX_OPEN_BATCH", 0)) if "skip_reasons" in res else 0,
        })

    out = pd.DataFrame(rows).sort_values("total_return", ascending=False)
    return out

def equity_curve_from_taken(taken_df, initial_equity=1.0):
    df = taken_df.copy()
    df["exit_ts"] = pd.to_datetime(df["exit_ts"])
    df = df.sort_values("exit_ts")
    # realized equity: equity *= (1 + weighted ret) is not correct here because you use notionals
    # so compute pnl explicitly:
    df["pnl"] = df["notional"] * df["ret_net_used"]
    equity = initial_equity
    curve = []
    for ts, g in df.groupby("exit_ts"):
        equity += g["pnl"].sum()
        curve.append((ts, equity))
    return pd.DataFrame(curve, columns=["ts","equity"])


def plot_context_v2(
    df,
    trades,
    center_ts,
    lookback=200,
    lookforward=200,
    title="",
    *,
    show_compression: bool = False,
    comp_kwargs: dict | None = None,
    vol_ma_win: int = 20,
    range_def: str = "hl_over_close",
    mark_center: bool = True,
):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if df is None or df.empty:
        raise ValueError("plot_context_v2: df is empty")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("plot_context_v2: df must have a DatetimeIndex")

    df = df.sort_index()
    center_ts = pd.Timestamp(center_ts)

    # if center_ts is not exactly in the index, snap to the previous bar
    if center_ts not in df.index:
        loc = df.index.searchsorted(center_ts, side="right") - 1
        if loc < 0:
            raise ValueError("plot_context_v2: center_ts is before df start")
        center_ts = df.index[loc]

    # slice window (inclusive-ish)
    idx = df.index.get_loc(center_ts)
    lo = max(0, idx - int(lookback))
    hi = min(len(df) - 1, idx + int(lookforward))
    d = df.iloc[lo:hi + 1].copy()

    # indicators
    d["atr"] = ATR(d, 14)
    if range_def == "hl_over_close":
        d["range"] = (d["high"].astype(float) - d["low"].astype(float)) / d["close"].astype(float)
    elif range_def == "true_range_over_close":
        pc = d["close"].shift(1)
        tr = pd.concat([
            (d["high"] - d["low"]),
            (d["high"] - pc).abs(),
            (d["low"] - pc).abs(),
        ], axis=1).max(axis=1)
        d["range"] = tr.astype(float) / d["close"].astype(float)
    else:
        raise ValueError("plot_context_v2: range_def must be 'hl_over_close' or 'true_range_over_close'")

    d["vol_ma"] = d["volume"].astype(float).rolling(int(vol_ma_win)).mean()

    # optional compression diagnostics (vol_compression, rvol, is_compressed)
    comp_cols = []
    comp_thr = {}
    if show_compression:
        # Need extra bars BEFORE the plot window for rolling baselines
        ck = dict(
            atr_short=20,
            atr_long=100,
            vol_win=144,
            vol_ratio_thresh=0.6,
            rvol_thresh=0.6,
            min_duration=12,
        )
        if comp_kwargs:
            ck.update(comp_kwargs)

        extra = int(max(ck.get("atr_long", 100), ck.get("vol_win", 144), 200))
        lo2 = max(0, lo - extra)
        d2 = df.iloc[lo2:hi + 1].copy()

        try:
            d2c = detect_compression_state(d2, **ck)
            # align back to d
            for c in ("vol_compression", "rvol", "is_compressed"):
                if c in d2c.columns:
                    d[c] = d2c.reindex(d.index)[c]
                    comp_cols.append(c)

            comp_thr = {
                "vol_ratio_thresh": float(ck.get("vol_ratio_thresh", 0.6)),
                "rvol_thresh": float(ck.get("rvol_thresh", 0.6)),
            }
        except Exception as e:
            # Don't kill plotting if compression calc fails for a short window
            print(f"plot_context_v2: compression diagnostics skipped ({e})")

    n_axes = 4 + (3 if (show_compression and len(comp_cols) == 3) else 0)
    fig_h = 2.0 * n_axes
    fig, axs = plt.subplots(n_axes, 1, figsize=(14, fig_h), sharex=True)
    if n_axes == 1:
        axs = [axs]

    # ---- Price
    axs[0].plot(d.index, d["close"], label="close")

    if mark_center:
        axs[0].axvline(center_ts, ls="--", alpha=0.35)

    # mark trades if provided
    if trades is not None and len(trades):
        t = trades.copy()
        if "entry_ts" in t.columns:
            t["entry_ts"] = pd.to_datetime(t["entry_ts"]).dt.tz_localize(None)
        if "exit_ts" in t.columns:
            t["exit_ts"] = pd.to_datetime(t["exit_ts"]).dt.tz_localize(None)

        t = t[(t["entry_ts"] >= d.index[0]) & (t["entry_ts"] <= d.index[-1])]

        ent = t["entry_ts"][t["entry_ts"].isin(d.index)]
        exi = t["exit_ts"][t["exit_ts"].isin(d.index)]

        if len(ent):
            axs[0].scatter(ent, d.loc[ent, "close"], marker="^", c="g")
        if len(exi):
            axs[0].scatter(exi, d.loc[exi, "close"], marker="v", c="r")
    axs[0].set_ylabel("Price")

    # ---- Volume
    axs[1].plot(d.index, d["vol_ma"], lw=1, label=f"vol_ma({int(vol_ma_win)})")
    axs[1].legend(loc="upper left")
    axs[1].set_ylabel("Volume")

    # ---- Volatility
    axs[2].plot(d.index, d["range"], label="Range", alpha=0.7)
    axs[2].legend()
    axs[2].set_ylabel("Vol")

    # ---- ATR
    axs[3].plot(d.index, d["atr"], label="ATR(14)")
    axs[3].legend()
    axs[3].set_ylabel("ATR")

    # ---- Compression diagnostics (optional)
    ax_i = 4
    if show_compression and len(comp_cols) == 3:
        # vol_compression (atr/atr_med)
        axs[ax_i].plot(d.index, d["vol_compression"], label="vol_compression")
        axs[ax_i].axhline(comp_thr.get("vol_ratio_thresh", 0.6), ls="--", alpha=0.4)
        axs[ax_i].legend(loc="upper left")
        axs[ax_i].set_ylabel("ATR/ATRmed")
        ax_i += 1

        # rvol
        axs[ax_i].plot(d.index, d["rvol"], label="rvol")
        axs[ax_i].axhline(comp_thr.get("rvol_thresh", 0.6), ls="--", alpha=0.4)
        axs[ax_i].legend(loc="upper left")
        axs[ax_i].set_ylabel("RVOL")
        ax_i += 1

        # is_compressed (0/1)
        y = d["is_compressed"].astype(float)
        axs[ax_i].step(d.index, y, where="post", label="is_compressed")
        axs[ax_i].set_ylim(-0.05, 1.05)
        axs[ax_i].set_yticks([0, 1])
        axs[ax_i].legend(loc="upper left")
        axs[ax_i].set_ylabel("Compressed")

    plt.suptitle(title)
    plt.show()


def simulate_long_exit_impulse(
    df, entry_ts, event_ts, entry_px,
    H_MAX=72, BAR_MIN=5, FEE=0.001,
    VOL_MA=20,                 # volume baseline
    RANGE_MA=20,               # range baseline
    ACTIVATE_RVOL=2.0,          # require at least this RVOL to consider "impulse on"
    ACTIVATE_RANGE_Z=1.0,       # require range expansion vs baseline
    STALL_FRAC=0.35,            # exit if impulse_score drops below this fraction of its peak
    MIN_HOLD_BARS=2,            # don't insta-exit
    GIVEBACK=0.40,              # exit if price gives back this fraction from peak
    TIME_CAP_BARS=48,           # shorter by default than your 72
    EMERGENCY_STOP_RET=-0.05,   # catastrophic stop in return space (e.g., -5%)
):
    """
    Pure impulse-based exit:
      - Start managing at event_ts + 1 bar
      - Define impulse_score using RVOL * RangeZ
      - After activation, track peak impulse + peak price return
      - Exit when impulse stalls OR price gives back from peak OR time cap OR emergency stop
    """
    import numpy as np
    import pandas as pd

    df = df.sort_index()
    start_ts = event_ts + pd.Timedelta(minutes=BAR_MIN)

    if df.empty or (entry_ts not in df.index) or (start_ts not in df.index):
        return None

    loc0 = df.index.get_loc(start_ts)
    max_loc = min(loc0 + int(H_MAX), len(df) - 1)

    # Precompute impulse features on full df (then slice by loc)
    vol = df["volume"].astype(float)
    vol_ma = vol.rolling(int(VOL_MA)).mean()
    rvol = vol / vol_ma

    rng = (df["high"].astype(float) - df["low"].astype(float)) / df["close"].astype(float)
    rng_ma = rng.rolling(int(RANGE_MA)).mean()
    rng_std = rng.rolling(int(RANGE_MA)).std(ddof=0)
    rng_z = (rng - rng_ma) / (rng_std.replace(0, np.nan))

    # A smooth-ish impulse score (positive only)
    impulse_score = (rvol.clip(lower=0) * rng_z.clip(lower=0)).replace([np.inf, -np.inf], np.nan)

    # State
    activated = False
    peak_imp = -np.inf
    peak_ret = 0.0
    mfe = 0.0
    mae = 0.0

    exit_ts = None
    exit_px = None
    exit_reason = None

    for loc in range(loc0, max_loc + 1):
        ts = df.index[loc]
        close = float(df["close"].iloc[loc])
        ret = (close - entry_px) / entry_px

        mfe = max(mfe, ret)
        mae = min(mae, ret)

        bars_since_start = loc - loc0

        # Emergency stop (catastrophic)
        if ret <= float(EMERGENCY_STOP_RET):
            exit_ts, exit_px, exit_reason = ts, close, "IMPULSE_EMERGENCY_STOP"
            break

        imp = impulse_score.iloc[loc]
        rv = rvol.iloc[loc]
        rz = rng_z.iloc[loc]

        # Activation gate: require "real impulse"
        if (not activated):
            if np.isfinite(imp) and np.isfinite(rv) and np.isfinite(rz):
                if (rv >= float(ACTIVATE_RVOL)) and (rz >= float(ACTIVATE_RANGE_Z)):
                    activated = True
                    peak_imp = float(imp)
                    peak_ret = float(mfe)
            continue

        # After activation, update peaks
        if np.isfinite(imp):
            peak_imp = max(peak_imp, float(imp))
        peak_ret = max(peak_ret, float(mfe))

        if bars_since_start < int(MIN_HOLD_BARS):
            continue

        # 1) Impulse stall exit (the key “shorten the window” rule)
        if np.isfinite(imp) and np.isfinite(peak_imp) and peak_imp > 0:
            if float(imp) <= float(STALL_FRAC) * float(peak_imp):
                exit_ts, exit_px, exit_reason = ts, close, "IMPULSE_STALL"
                break

        # 2) Price giveback from peak (captures TRUMP-like fade)
        if peak_ret > 0:
            if ret <= (1.0 - float(GIVEBACK)) * float(peak_ret):
                exit_ts, exit_px, exit_reason = ts, close, "IMPULSE_GIVEBACK"
                break

        # 3) Time cap
        if bars_since_start >= int(TIME_CAP_BARS):
            exit_ts, exit_px, exit_reason = ts, close, "IMPULSE_TIME_CAP"
            break

    if exit_ts is None:
        exit_ts = df.index[max_loc]
        exit_px = float(df["close"].iloc[max_loc])
        exit_reason = "IMPULSE_END_OF_WINDOW"

    ret = (exit_px - entry_px) / entry_px
    ret_net = ret - 2 * FEE

    return {
        "exit_ts": exit_ts,
        "exit_px": float(exit_px),
        "ret": float(ret),
        "ret_net": float(ret_net),
        "mfe": float(mfe),
        "mae": float(mae),
        "exit_reason": str(exit_reason),
        "trail_active": False,
        # Helpful debug fields (optional)
        "peak_impulse": float(peak_imp) if np.isfinite(peak_imp) else np.nan,
        "peak_ret": float(peak_ret),
        "ACTIVATE_RVOL": float(ACTIVATE_RVOL),
        "ACTIVATE_RANGE_Z": float(ACTIVATE_RANGE_Z),
        "STALL_FRAC": float(STALL_FRAC),
        "GIVEBACK": float(GIVEBACK),
        "TIME_CAP_BARS": int(TIME_CAP_BARS),
    }


def plot_trades_on_price(df, trades, title=""):
    plt.figure(figsize=(12,4))
    plt.plot(df.index, df["close"], label="close", alpha=0.8)

    buys = trades["entry_ts"]
    sells = trades["exit_ts"]

    plt.scatter(buys, df.loc[buys]["close"], marker="^", color="green", s=60, label="entry")
    plt.scatter(sells, df.loc[sells]["close"], marker="v", color="red", s=60, label="exit")

    plt.legend()
    plt.title(title)
    plt.show()


def find_pre_shock_entry_delayed(
    df,
    event_ts,
    *,
    PRE_BARS=240,          # how far back we look (bars)
    RANGE_BARS=24,         # compression box length (bars)
    MAX_LEAD_BARS=36,      # scan window inside PRE (bars)
    ATR_N=14,
    BODY_ATR_MULT=0.75,    # same as your old filter (body > 0.75*atr_pre)
    BREAK_ATR_BUFFER=0.20, # require breakout clears level by buffer*atr_pre
    CONFIRM_BARS=2,        # 2-bar close confirmation (delays entry)
    RVOL_WIN=144,
    RVOL_MIN=1.2,          # require volume expansion at entry (optional but recommended)
    USE_RVOL=True,
):
    """
    Like find_pre_shock_entry(), but delays entry using:
      - breakout buffer (avoid tiny pokes)
      - N-bar confirmation (avoid 1-bar fakes)
      - optional RVOL filter (enter only when volume expands)
    Returns same dict as before or None.
    """
    import numpy as np
    import pandas as pd

    if event_ts not in df.index:
        loc = df.index.searchsorted(pd.Timestamp(event_ts), side="right") - 1
        if loc < 0:
            return None
        event_ts = df.index[loc]

    event_loc = df.index.get_loc(event_ts)
    if event_loc < PRE_BARS + 5:
        return None

    pre = df.iloc[event_loc - PRE_BARS : event_loc].copy()

    # ATR proxy used in your old logic
    atr_pre = (pre["high"] - pre["low"]).mean()

    # compression range = last RANGE_BARS in PRE
    comp = pre.iloc[-RANGE_BARS:]
    range_high = comp["high"].max()
    range_low  = comp["low"].min()

    # Scan last MAX_LEAD_BARS (closest to event)
    scan = pre.iloc[-MAX_LEAD_BARS:].copy()

    # Optional RVOL on scan window (needs some lookback)
    if USE_RVOL:
        vol_med = pre["volume"].rolling(RVOL_WIN).median()
        scan["rvol"] = scan["volume"] / vol_med.reindex(scan.index)
    else:
        scan["rvol"] = np.nan

    # breakout must clear by a buffer to avoid tiny pokes
    up_level = range_high + BREAK_ATR_BUFFER * atr_pre
    dn_level = range_low  - BREAK_ATR_BUFFER * atr_pre

    idx = scan.index

    for i in range(len(scan)):
        ts = idx[i]
        row = scan.loc[ts]

        body = abs(row["close"] - row["open"])

        # basic impulse candle requirement (same style as you had)
        if not (body > BODY_ATR_MULT * atr_pre):
            continue

        # optional volume expansion requirement at entry bar
        if USE_RVOL:
            rvol = row["rvol"]
            if not (np.isfinite(rvol) and rvol >= RVOL_MIN):
                continue

        # -------- LONG candidate --------
        if row["close"] > up_level:
            # confirmation: next CONFIRM_BARS-1 closes must also stay above up_level
            ok = True
            for j in range(1, CONFIRM_BARS):
                if i + j >= len(scan):
                    ok = False
                    break
                if scan.iloc[i + j]["close"] <= up_level:
                    ok = False
                    break

            if ok:
                entry_ts = idx[i + CONFIRM_BARS - 1]  # delayed entry on confirmation bar
                entry_px = scan.loc[entry_ts, "close"]
                return {"entry_ts": entry_ts, "direction": "long", "entry_price": entry_px}

        # -------- SHORT candidate --------
        if row["close"] < dn_level:
            ok = True
            for j in range(1, CONFIRM_BARS):
                if i + j >= len(scan):
                    ok = False
                    break
                if scan.iloc[i + j]["close"] >= dn_level:
                    ok = False
                    break

            if ok:
                entry_ts = idx[i + CONFIRM_BARS - 1]
                entry_px = scan.loc[entry_ts, "close"]
                return {"entry_ts": entry_ts, "direction": "short", "entry_price": entry_px}

    return None
