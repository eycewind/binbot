"""E0.3d.1: offline response analysis of Bollinger *envelopes* around
fixed, independent ATR/price impulses. No trading or forecasting claims.

Inputs are completed, regular, UTC candle-close bars. References have
onset_index/confirmation_index/direction from E0.3c and MUST be held fixed
across the (N,k) sweep. Future samples are used exclusively for scoring.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _check(bars, refs):
    if bars.empty or bars.index.has_duplicates or not bars.index.is_monotonic_increasing or bars.index.tz is None:
        raise ValueError('Require unique, monotonic, timezone-aware completed candle times')
    if not {'close','high','low'}.issubset(bars.columns):
        raise ValueError('Need close/high/low')
    if not {'onset_index','confirmation_index','direction'}.issubset(refs.columns):
        raise ValueError('References need onset_index, confirmation_index, direction')


def _atr(bars, span=14):
    prev=bars.close.shift()
    tr=pd.concat([(bars.high-bars.low).abs(),(bars.high-prev).abs(),(bars.low-prev).abs()],axis=1).max(axis=1)
    valid=bars[['high','low','close']].notna().all(axis=1)&prev.notna()
    return tr.where(valid).ewm(alpha=1/span,adjust=False,min_periods=span).mean()


def impulse_response_details(bars, references, windows=(6,8,10,12,15,18,20,24,30,40,50),
                             sigmas=(1.5,1.7,1.9,2.0,2.1,2.3,2.5),
                             follow_bars=6, response_atr=0.25, min_band_width=1e-8):
    """One row per fixed impulse per parameter pair.

    Band reaction is measured from the last band known at impulse onset
    (band at the preceding candle close) through the *confirmation* close.
    Directional band displacement >0 means upper rises/lower falls.
    Response lag is first candle close between onset and confirmation when
    that directional displacement reaches response_atr times ATR known at onset.
    It is NaN if the envelope doesn't respond by confirmation.

    Recontainment measured AFTER the reference confirmation, up to follow_bars,
    only if reference close at confirmation lies outside current envelope.
    `recontain_cause` distinguishes band-catching-up vs price-retracing by
    decomposition of changes in price and directional band value; mixed and
    ambiguous cases are explicitly marked. Later returns are not predictors.
    """
    _check(bars,references)
    if follow_bars<1 or response_atr<=0: raise ValueError('invalid follow_bars / response_atr')
    nrows=len(bars); close=bars.close.astype(float)
    atr=_atr(bars)
    out=[]
    for n in windows:
        if n<2: raise ValueError('n >= 2 required')
        middle=close.rolling(n,min_periods=n).mean()
        sd=close.rolling(n,min_periods=n).std(ddof=0)
        for k in sigmas:
            if k<=0: raise ValueError('sigma > 0 required')
            up=middle+k*sd; low=middle-k*sd
            for ref in references.itertuples(index=False):
                i=int(ref.onset_index); j=int(ref.confirmation_index)
                if i<=0 or j<i or j>=nrows: continue
                direction=str(ref.direction)
                if direction not in ('UPPER_WALK','LOWER_WALK'): raise ValueError('bad direction')
                sign=1 if direction=='UPPER_WALK' else -1
                band=up if sign==1 else low
                anchor=band.iloc[i-1]; a=atr.iloc[i-1]
                if not (np.isfinite(anchor) and np.isfinite(a) and a>0 and np.isfinite(band.iloc[j])):continue
                if close.iloc[i:j+1].isna().any() or band.iloc[i:j+1].isna().any(): continue
                price_move=sign*(close.iloc[j]-close.iloc[i-1]);band_move=sign*(band.iloc[j]-anchor)
                response=sign*(band.iloc[i:j+1].to_numpy()-anchor)
                threshold=response_atr*a
                hit=np.flatnonzero(np.isfinite(response)&(response>=threshold))
                lag=float(hit[0]) if len(hit) else np.nan
                # Signed distance: positive => price outside the active band.
                outside_at_confirm=sign*(close.iloc[j]-band.iloc[j])
                rec_lag=np.nan; cause='NOT_OUTSIDE' if outside_at_confirm<=0 else 'NOT_RECONTAINED'
                if outside_at_confirm>0:
                    for d in range(1,follow_bars+1):
                        t=j+d
                        if t>=nrows or not(np.isfinite(close.iloc[t]) and np.isfinite(band.iloc[t])):break
                        if sign*(close.iloc[t]-band.iloc[t])<=0:
                            rec_lag=float(d)
                            # D_t = sign*(P_t-B_t), delta D = price_direction - band_direction
                            # Both components can contribute, so classify only dominance.
                            price_retrace=-sign*(close.iloc[t]-close.iloc[j]);band_catch=sign*(band.iloc[t]-band.iloc[j])
                            if price_retrace>0 and price_retrace>band_catch: cause='PRICE_DOMINANT'
                            elif band_catch>0 and band_catch>price_retrace: cause='BAND_DOMINANT'
                            else: cause='MIXED_OR_OTHER'
                            break
                out.append(dict(window=n,sigma=float(k),ref_id=getattr(ref,'ref_id',i),direction=direction,
                    onset_index=i,confirmation_index=j,impulse_bars=j-i,
                    band_response_lag_bars=lag,band_move_atr=float(band_move/a),
                    price_move_atr=float(price_move/a),
                    directional_tracking_gap_atr=float((price_move-band_move)/a),
                    outside_at_confirm=bool(outside_at_confirm>0),
                    recontain_lag_bars=rec_lag,recontain_cause=cause))
    return pd.DataFrame(out)


def summarize_impulse_response(detail):
    if detail.empty: return pd.DataFrame()
    rows=[]
    for (n,k),g in detail.groupby(['window','sigma']):
        outside=g[g.outside_at_confirm]
        rows.append(dict(window=n,sigma=k,references=len(g),
            response_fraction=float(g.band_response_lag_bars.notna().mean()),
            median_band_response_lag=float(g.band_response_lag_bars.median()),
            median_band_move_atr=float(g.band_move_atr.median()),
            median_tracking_gap_atr=float(g.directional_tracking_gap_atr.median()),
            outside_at_confirmation=float(g.outside_at_confirm.mean()),
            recontain_fraction=float(outside.recontain_lag_bars.notna().mean()) if len(outside) else np.nan,
            median_recontain_bars=float(outside.recontain_lag_bars.median()),
            band_dominant_frac=float(outside.recontain_cause.eq('BAND_DOMINANT').mean()) if len(outside) else np.nan))
    return pd.DataFrame(rows)
