"""
ss_bands.py - state-space Bollinger-style bands for 5m crypto bars.

Causality convention (same as E0.2a): everything is stamped at the CLOSE of a
completed candle. For every method we produce

    center_fc[t], scale_fc[t]  -> the band for bar t, published at close of t-1
    z[t] = (close[t] - center_fc[t]) / scale_fc[t]

so "close of bar t is inside the k-band" <=> |z[t]| <= k. Containment, width,
reaction lag and breakout signals are all computed from z, which keeps the
comparison between methods identical and free of look-ahead.

Method ladder (each step adds one idea, so each step's value can be measured):
    BB(N)    : SMA center, rolling std scale                  (classic)
    EWMA(N)  : EWMA center == steady-state local-level Kalman (adds nothing but weighting)
    LLT(N)   : local linear trend Kalman (level + slope)      (adds slope -> less trend lag)
    JA-LLT   : LLT whose level noise is boosted by P(jump) from a two-state
               Markov-switching (calm/jump) filter on returns -> re-anchors fast
               after jumps while staying smooth otherwise (IMM-style idea)

Kalman models run on log price. Process / observation noise are scaled by an
EWMA of squared 1-bar log returns known at t-1 (h_prev), so the filter adapts to
volatility regimes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import fsolve, minimize

try:
    from numba import njit
except ImportError:  # works without numba, just slower
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

LN2PI = np.log(2.0 * np.pi)


def alpha_from_n(n: float) -> float:
    return 2.0 / (n + 1.0)


# ---------------------------------------------------------------- primitives
@njit(cache=True)
def _ewma_sq(x, lam, init):
    """out[t] = lam*out[t-1] + (1-lam)*x[t]^2  (uses x up to and including t)."""
    out = np.empty(x.shape[0])
    v = init
    for t in range(x.shape[0]):
        xt = x[t]
        if np.isfinite(xt):
            v = lam * v + (1.0 - lam) * xt * xt
        out[t] = v
    return out


@njit(cache=True)
def _ewma(x, lam):
    out = np.empty(x.shape[0])
    v = x[0]
    for t in range(x.shape[0]):
        v = lam * v + (1.0 - lam) * x[t]
        out[t] = v
    return out


def return_variance(logp: np.ndarray, n: float = 288.0, init_bars: int = 288) -> np.ndarray:
    """EWMA variance of 1-bar log returns, h[t] uses returns up to t.
    Returns h_prev, i.e. h shifted by one bar (known before bar t closes).
    init uses the first `init_bars` returns (warm-up period only)."""
    r = np.diff(logp, prepend=logp[0])
    init = np.mean(r[1:init_bars + 1] ** 2)
    h = _ewma_sq(r, 1.0 - alpha_from_n(n), init)
    h_prev = np.empty_like(h)
    h_prev[0] = init
    h_prev[1:] = h[:-1]
    return h_prev


# ------------------------------------------------------ local linear trend KF
@njit(cache=True)
def _llt_filter(y, s2, qm, qb, r):
    """Local linear trend Kalman filter, scalar-unrolled 2x2.
    y: log price, s2: per-bar noise scale (h_prev). Q = s2*diag(qm,qb), R = s2*r.
    Returns m_pred, b_pred, S, m_filt, b_filt, loglik_t (per-bar)."""
    n = y.shape[0]
    m_pred = np.full(n, np.nan); b_pred = np.full(n, np.nan); S_out = np.full(n, np.nan)
    m_f = np.empty(n); b_f = np.empty(n); ll = np.zeros(n)
    m = y[0]; b = 0.0
    P00 = s2[0] * 10.0; P01 = 0.0; P11 = s2[0]
    m_f[0] = m; b_f[0] = b
    for t in range(1, n):
        q0 = s2[t] * qm; q1 = s2[t] * qb; R = s2[t] * r
        mp = m + b; bp = b
        p00 = P00 + 2.0 * P01 + P11 + q0
        p01 = P01 + P11
        p11 = P11 + q1
        S = p00 + R
        v = y[t] - mp
        k0 = p00 / S; k1 = p01 / S
        m = mp + k0 * v; b = bp + k1 * v
        P00 = (1.0 - k0) * p00
        P01 = (1.0 - k0) * p01
        P11 = p11 - k1 * p01
        m_pred[t] = mp; b_pred[t] = bp; S_out[t] = S
        m_f[t] = m; b_f[t] = b
        ll[t] = -0.5 * (LN2PI + np.log(S) + v * v / S)
    return m_pred, b_pred, S_out, m_f, b_f, ll


def llt_ratios_for_gains(k0: float, k1: float) -> tuple[float, float]:
    """Find (qm, qb) with R=1 whose steady-state Kalman gains are (k0, k1)."""
    def gains(lq):
        qm, qb = np.exp(lq)
        P00, P01, P11 = 1.0, 0.0, 1.0
        for _ in range(20000):
            p00 = P00 + 2 * P01 + P11 + qm; p01 = P01 + P11; p11 = P11 + qb
            S = p00 + 1.0; g0 = p00 / S; g1 = p01 / S
            n00 = (1 - g0) * p00; n01 = (1 - g0) * p01; n11 = p11 - g1 * p01
            if abs(n00 - P00) + abs(n01 - P01) + abs(n11 - P11) < 1e-14:
                P00, P01, P11 = n00, n01, n11
                break
            P00, P01, P11 = n00, n01, n11
        return np.array([np.log(g0) - np.log(k0), np.log(g1) - np.log(k1)])
    lq = fsolve(gains, x0=np.log([k0 * k0, k1 * k1]), xtol=1e-12)
    return float(np.exp(lq[0])), float(np.exp(lq[1]))


def llt_ratios_for_n(n: float) -> tuple[float, float]:
    """Gains of Brown's double exponential smoothing with a = 2/(N+1):
    k0 = 1-(1-a)^2, k1 = a^2. Tracks a linear trend with no steady-state lag."""
    a = alpha_from_n(n)
    return llt_ratios_for_gains(1 - (1 - a) ** 2, a * a)


# ------------------------------------------- jump regime filter (Hamilton)
@njit(cache=True)
def _msv_filter(r, h, s0, s1, p00, p11):
    """Two-state Markov-switching variance filter on 1-bar log returns.
    calm: r ~ N(0, s0*h_prev), jump: r ~ N(0, s1*h_prev).
    Returns p_jump[t] (posterior using r up to t) and per-bar loglik."""
    n = r.shape[0]
    pj = np.full(n, np.nan); ll = np.zeros(n)
    mu1 = 0.02
    for t in range(1, n):
        c1 = (1.0 - p00) * (1.0 - mu1) + p11 * mu1
        c0 = 1.0 - c1
        v0 = s0 * h[t]; v1 = s1 * h[t]
        l0 = -0.5 * (LN2PI + np.log(v0) + r[t] * r[t] / v0)
        l1 = -0.5 * (LN2PI + np.log(v1) + r[t] * r[t] / v1)
        lm = max(l0, l1)
        e0 = c0 * np.exp(l0 - lm); e1 = c1 * np.exp(l1 - lm)
        ll[t] = lm + np.log(e0 + e1)
        mu1 = min(max(e1 / (e0 + e1), 1e-9), 1.0 - 1e-9)
        pj[t] = mu1
    return pj, ll


def fit_jump_filter(logp, h_prev, train_mask):
    """MLE of (s0, s1>s0, p00, p11) on TRAIN rows only."""
    r = np.diff(logp, prepend=logp[0])

    def unpack(th):
        s0 = np.exp(th[0]); s1 = s0 * (1.0 + np.exp(th[1]))
        return s0, s1, 1 / (1 + np.exp(-th[2])), 1 / (1 + np.exp(-th[3]))

    def nll(th):
        out = _msv_filter(r, h_prev, *unpack(th))[1][train_mask].sum()
        return -out if np.isfinite(out) else 1e30

    res = minimize(nll, x0=np.array([np.log(0.7), np.log(8.0), 4.0, 1.0]), method="Nelder-Mead",
                   options=dict(xatol=1e-4, fatol=1e-3, maxiter=2000))
    s0, s1, p00, p11 = unpack(res.x)
    return dict(s0=s0, s1=s1, p00=p00, p11=p11, nll=res.fun)


def jump_probability(logp, h_prev, params) -> np.ndarray:
    r = np.diff(logp, prepend=logp[0])
    p = params
    return _msv_filter(r, h_prev, p["s0"], p["s1"], p["p00"], p["p11"])[0]


@njit(cache=True)
def _ja_llt_filter(y, s2, qm, qb, r, qj, pj):
    """Jump-adaptive LLT: level process noise at bar t is s2*(qm + pj[t]*qj).
    pj[t] uses the bar-t return, which is legal for the bar-t UPDATE. The
    one-step forecast m_pred[t] = m + b is formed before bar t and never
    depends on pj[t] (Q only changes the covariance, not the predicted mean)."""
    n = y.shape[0]
    m_pred = np.full(n, np.nan); m_f = np.empty(n); b_f = np.empty(n)
    m = y[0]; b = 0.0
    P00 = s2[0] * 10.0; P01 = 0.0; P11 = s2[0]
    m_f[0] = m; b_f[0] = b
    for t in range(1, n):
        p = pj[t] if np.isfinite(pj[t]) else 0.0
        q0 = s2[t] * (qm + p * qj); q1 = s2[t] * qb; R = s2[t] * r
        mp = m + b; bp = b
        p00 = P00 + 2.0 * P01 + P11 + q0
        p01 = P01 + P11
        p11 = P11 + q1
        S = p00 + R
        v = y[t] - mp
        k0 = p00 / S; k1 = p01 / S
        m = mp + k0 * v; b = bp + k1 * v
        P00 = (1.0 - k0) * p00; P01 = (1.0 - k0) * p01; P11 = p11 - k1 * p01
        m_pred[t] = mp; m_f[t] = m; b_f[t] = b
    return m_pred, m_f, b_f


# ------------------------------------------------------------- band methods
def _from_log_pred(close, logc_pred, dev_scale_prev, extra=None):
    """Build the standard output frame from a log-space one-step center forecast
    and a log-space scale (both already aligned to bar t, i.e. known at t-1)."""
    out = pd.DataFrame(index=close.index)
    out["center_fc"] = np.exp(logc_pred)
    out["scale_log"] = dev_scale_prev                 # band half-width per k, in log units
    out["z"] = (np.log(close.values) - logc_pred) / dev_scale_prev
    out["width_bps_per_k"] = 2e4 * dev_scale_prev     # full band width for k=1, bps
    if extra:
        for k, v in extra.items():
            out[k] = v
    return out


def band_bb(close: pd.Series, n: int) -> pd.DataFrame:
    """Classic Bollinger, published at t-1 and applied to bar t."""
    sma = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    c_fc = sma.shift(1); s_fc = sd.shift(1)
    out = pd.DataFrame(index=close.index)
    out["center_fc"] = c_fc
    out["scale_log"] = s_fc / c_fc
    out["z"] = (close - c_fc) / s_fc
    out["width_bps_per_k"] = 2e4 * s_fc / c_fc
    out["center_now"] = sma
    return out


def _dev_scale(logp, logc_pred, n_scale):
    """EWMA RMS of (log close - predicted center); scale_prev[t] uses devs up to t-1."""
    dev = logp - logc_pred
    first = np.flatnonzero(np.isfinite(dev))[0]
    init = np.nanmean(dev[first:first + 288] ** 2)
    D = _ewma_sq(np.where(np.isfinite(dev), dev, np.nan), 1 - alpha_from_n(n_scale), init)
    prev = np.empty_like(D); prev[0] = np.nan; prev[1:] = D[:-1]
    return np.sqrt(prev)


def band_ewma(close: pd.Series, n: float, n_scale: float | None = None) -> pd.DataFrame:
    """EWMA center == steady-state local-level Kalman filter."""
    logp = np.log(close.values)
    e = _ewma(logp, 1 - alpha_from_n(n))
    pred = np.empty_like(e); pred[0] = np.nan; pred[1:] = e[:-1]
    sc = _dev_scale(logp, pred, n_scale or n)
    return _from_log_pred(close, pred, sc, dict(center_now=np.exp(e)))


def band_llt(close: pd.Series, n: float, n_scale: float | None = None, h_prev=None) -> pd.DataFrame:
    """Local linear trend Kalman center (Brown-equivalent gains for window n).
    The one-step forecast m+b projects the slope into the next bar."""
    logp = np.log(close.values)
    if h_prev is None:
        h_prev = return_variance(logp)
    qm, qb = llt_ratios_for_n(n)
    m_pred, b_pred, S, m_f, b_f, _ = _llt_filter(logp, h_prev, qm, qb, 1.0)
    sc = _dev_scale(logp, m_pred, n_scale or n)
    return _from_log_pred(close, m_pred, sc,
                          dict(center_now=np.exp(m_f), slope_bps=1e4 * b_f))


def _dev_scale_weighted(logp, logc_pred, n_scale, weight):
    """Like _dev_scale, but bar t only updates the scale with weight[t] in [0,1].
    weight = 1 - p_jump keeps one jump bar from inflating the band for N bars."""
    dev = logp - logc_pred
    lam = 1 - alpha_from_n(n_scale)
    first = np.flatnonzero(np.isfinite(dev))[0]
    D = np.nanmean(dev[first:first + 288] ** 2)
    prev = np.full(len(dev), np.nan)
    for t in range(len(dev)):
        prev[t] = D
        if np.isfinite(dev[t]) and np.isfinite(weight[t]):
            D = D + (1 - lam) * weight[t] * (dev[t] ** 2 - D)
    return np.sqrt(prev)


def band_ja_llt(close: pd.Series, n: float, jump_params: dict, n_scale: float | None = None,
                h_prev=None, robust_scale: bool = True, qj: float | None = None) -> pd.DataFrame:
    """Jump-adaptive LLT bands.
    Calm: behaves like LLT(n) (smooth center).
    When the switching filter says P(jump) is high, the level gets extra process
    noise qj (default: the jump variance factor s1), so the center re-anchors to
    the new price within a bar or two instead of over ~n bars.
    robust_scale: jump bars are down-weighted in the band-width estimate."""
    logp = np.log(close.values)
    if h_prev is None:
        h_prev = return_variance(logp)
    qm, qb = llt_ratios_for_n(n)
    pj = jump_probability(logp, h_prev, jump_params)
    m_pred, m_f, b_f = _ja_llt_filter(logp, h_prev, qm, qb, 1.0,
                                      jump_params["s1"] if qj is None else qj, pj)
    w = (1.0 - np.nan_to_num(pj)) if robust_scale else np.ones_like(pj)
    sc = _dev_scale_weighted(logp, m_pred, n_scale or n, w)
    pj_prev = np.empty_like(pj); pj_prev[0] = np.nan; pj_prev[1:] = pj[:-1]
    return _from_log_pred(close, m_pred, sc, dict(center_now=np.exp(m_f), slope_bps=1e4 * b_f,
                                                  p_jump=pj, p_jump_prev=pj_prev))


# --------------------------------------------------------------- evaluation
def containment_curve(z: pd.Series, width_per_k: pd.Series, ks=np.linspace(0.5, 4.0, 36)):
    ok = z.notna() & width_per_k.notna()
    az = z[ok].abs().values; w = width_per_k[ok].values.mean()
    return pd.DataFrame({"k": ks, "containment": [(az <= k).mean() for k in ks],
                         "mean_width_bps": ks * w})


def calibrate_k(z_train: pd.Series, target: float = 0.95) -> float:
    return float(np.nanquantile(z_train.abs(), target))


def shock_times(close: pd.Series, h_prev: np.ndarray, thresh: float = 5.0) -> pd.DatetimeIndex:
    """Bars whose 1-bar log return exceeds `thresh` x EWMA return std known at t-1."""
    r = np.diff(np.log(close.values), prepend=np.nan)
    s = np.abs(r) / np.sqrt(h_prev)
    return close.index[np.nan_to_num(s) > thresh]


def reaction_stats(z: pd.Series, k: float, shocks: pd.DatetimeIndex, post: int = 12):
    """lag = consecutive bars outside the k-band starting at the shock bar;
    post_containment = share of the next `post` bars inside the band."""
    zv = z.values; pos = z.index.get_indexer(shocks)
    lags, conts = [], []
    for p in pos:
        if p < 0 or p + post >= len(zv) or not np.isfinite(zv[p]):
            continue
        L = 0
        while p + L < len(zv) and abs(zv[p + L]) > k and L < 288:
            L += 1
        lags.append(L)
        seg = zv[p + 1:p + 1 + post]
        conts.append(np.mean(np.abs(seg) <= k))
    return dict(n_shocks=len(lags), median_lag_bars=float(np.median(lags)) if lags else np.nan,
                mean_lag_bars=float(np.mean(lags)) if lags else np.nan,
                post_containment=float(np.mean(conts)) if conts else np.nan)


def event_returns(close: pd.Series, events: pd.Series, horizon: int, cost_bps: float):
    """events: +1 / -1 at the signal bar (decided at its close). Trade taken at
    that close, held `horizon` bars, in the event direction (continuation).
    Events closer than `horizon` bars to the previous kept event are dropped
    so the samples do not overlap. Negative mean = mean reversion."""
    idx = np.flatnonzero(events.values != 0)
    kept, last = [], -10 ** 9
    for i in idx:
        if i - last >= horizon and i + horizon < len(close):
            kept.append(i); last = i
    kept = np.array(kept, dtype=int)
    if len(kept) == 0:
        return dict(n=0)
    lc = np.log(close.values)
    sgn = events.values[kept]
    fr = 1e4 * sgn * (lc[kept + horizon] - lc[kept])
    return dict(n=len(kept), mean_bps=fr.mean(), t_stat=fr.mean() / (fr.std(ddof=1) / np.sqrt(len(fr)))
                if len(fr) > 2 else np.nan, hit=(fr > 0).mean(), net_bps=fr.mean() - cost_bps,
                long_mean_bps=fr[sgn > 0].mean() if (sgn > 0).any() else np.nan,
                short_mean_bps=fr[sgn < 0].mean() if (sgn < 0).any() else np.nan)


def cross_events(z: pd.Series, k: float) -> pd.Series:
    """+1 on the first close above the upper band, -1 on the first close below the lower."""
    up = (z > k) & ~(z.shift(1) > k)
    dn = (z < -k) & ~(z.shift(1) < -k)
    return up.astype(int) - dn.astype(int)


def regime_events(p_jump: pd.Series, z: pd.Series, thresh: float = 0.5) -> pd.Series:
    """Regime signal: jump probability crosses above `thresh`, direction = sign of z."""
    cross = (p_jump > thresh) & ~(p_jump.shift(1) > thresh)
    return (cross.astype(int) * np.sign(z).fillna(0)).astype(int)


# ------------------------------------------------------------ synthetic data
def synthetic_5m(index: pd.DatetimeIndex, seed: int = 7, trends: bool = True, jumps: bool = True,
                 p0: float = 94000.0):
    """GARCH(1,1) Student-t(4) returns + optional persistent drift regimes + jumps.
    Only for checking the code. With trends=False the series is a martingale, so
    any forward-return 'edge' found on it means look-ahead leakage."""
    rng = np.random.default_rng(seed)
    n = len(index)
    nu = 4.0
    eps = rng.standard_t(nu, n) / np.sqrt(nu / (nu - 2))
    target = 0.0012 ** 2
    a, b = 0.06, 0.92
    w = target * (1 - a - b)
    r = np.empty(n); h = target
    for t in range(n):
        r[t] = np.sqrt(h) * eps[t]
        h = w + a * r[t] ** 2 + b * h
    drift = np.zeros(n)
    if trends:
        state, t = 0, 0
        levels = np.array([0.0, 1.5e-4, -1.5e-4])
        while t < n:
            L = int(rng.exponential(300)) + 20
            drift[t:t + L] = levels[state]
            state = rng.choice([s for s in range(3) if s != state]) if rng.random() < 0.7 else 0
            t += L
    jump_bars = []
    if jumps:
        t = 500
        while True:
            t += int(rng.exponential(2500)) + 50
            if t >= n - 5:
                break
            size = rng.choice([-1, 1]) * rng.uniform(0.008, 0.03)
            spread = rng.integers(1, 4)
            r[t:t + spread] += size / spread
            jump_bars.append(t)
    lr = r + drift
    close = p0 * np.exp(np.cumsum(lr))
    open_ = np.concatenate([[p0], close[:-1]])
    wig = np.abs(rng.normal(0, 0.0006, (2, n)))
    high = np.maximum(open_, close) * (1 + wig[0]); low = np.minimum(open_, close) * (1 - wig[1])
    bars = pd.DataFrame(dict(open=open_, high=high, low=low, close=close,
                             volume=rng.gamma(2.0, 50.0, n)), index=index)
    return bars, index[jump_bars]
