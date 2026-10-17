"""E0.3d: descriptive Bollinger parameter surface. No trading claims.
All inputs are completed candle-close UTC OHLCV. Scoring uses preceding
bands at each candle; retrospective impulse onsets are for audit only.
"""
import numpy as np
import pandas as pd

WINDOWS=(6,8,10,12,15,18,20,24,30,40,50)
SIGMAS=(1.5,1.7,1.9,2.0,2.1,2.3,2.5)

def surface(bb, windows=WINDOWS, sigmas=SIGMAS, impulse_lookback=3,
            impulse_atr=1.5, response_bars=3, score_start=None):
    """Band-only metrics. `bb` supplies raw completed OHLCV with full warmup.
    For each parameter pair, assess ONLY candles with complete prior bands.
    Impulse reference uses only price/ATR; its confirmation is at t, not onset.
    `response_capture` is fraction of price displacement captured by SMA
    from impulse confirmation to t+response_bars; descriptive and clipped only
    when displayed, never used to imply predictive accuracy.
    """
    if impulse_lookback<1 or response_bars<1 or impulse_atr<=0: raise ValueError('invalid horizons')
    if bb.empty or bb.index.has_duplicates or bb.index.tz is None: raise ValueError('UTC unique candle-close index required')
    df=bb.sort_index()
    required=['open','high','low','close']
    if any(c not in df for c in required): raise ValueError('Missing OHLC')
    close=df.close.astype(float)
    prev_close=close.shift(1)
    tr=pd.concat([df.high-df.low,(df.high-prev_close).abs(),(df.low-prev_close).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/14, adjust=False,min_periods=14).mean().shift(impulse_lookback)
    displacement=close-close.shift(impulse_lookback)
    refs=displacement.abs().ge(impulse_atr*atr)&atr.gt(0)
    # Avoid treating every candle of a move as an independent impulse.
    refs=refs&~refs.shift(1,fill_value=False)
    eligible_period=pd.Series(True,index=df.index)
    if score_start is not None:
        ts=pd.Timestamp(score_start)
        ts=ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')
        eligible_period=df.index>=ts
    results=[]
    for n in windows:
        if n<2: raise ValueError('window must be >=2')
        m=close.rolling(n,min_periods=n).mean()
        sd=close.rolling(n,min_periods=n).std(ddof=0)
        for k in sigmas:
            if k<=0: raise ValueError('sigma must be positive')
            upper=m+k*sd;lower=m-k*sd
            pu=upper.shift(1);pl=lower.shift(1)
            valid=close.notna()&pu.notna()&pl.notna()&pu.gt(pl)&pu.gt(0)&pl.gt(0)&eligible_period
            outside=(close.gt(pu)|close.lt(pl))&valid
            # Two-sided crossing of band previously known at candle open.
            cross=((df.open.lt(pu)&close.ge(pu)) | (df.open.gt(pl)&close.le(pl)))&valid&df.open.notna()
            # Band-width cost measured BEFORE the candle, relative to prior close.
            bw=(pu-pl)/prev_close
            # How promptly the center tracks a detected displacement, without
            # considering price/center motion over different absolute scales.
            j=np.flatnonzero((refs&valid).to_numpy())
            j=j[j+response_bars<len(df)]
            j=j[m.iloc[j+response_bars].notna().to_numpy() & m.iloc[j].notna().to_numpy()]
            delta_p=(close.iloc[j+response_bars].to_numpy()-close.iloc[j].to_numpy())
            delta_m=(m.iloc[j+response_bars].to_numpy()-m.iloc[j].to_numpy())
            same=np.sign(delta_p)==np.sign(delta_m)
            # Only meaningful future movement to avoid division by tiny deltas.
            meaningful=np.abs(delta_p)>0.1*atr.iloc[j].to_numpy()
            capture=np.abs(delta_m[meaningful])/np.maximum(np.abs(delta_p[meaningful]),1e-12)
            aligned=same[meaningful]
            results.append(dict(window=n,sigma=float(k),valid_bars=int(valid.sum()),
                containment=float(1-outside.sum()/valid.sum()) if valid.any() else np.nan,
                outside_rate=float(outside.sum()/valid.sum()) if valid.any() else np.nan,
                crossing_per_day=np.nan, # populated once resolution known
                crossings=int(cross.sum()),
                median_width_pct=float(100*bw.where(valid).median()),
                response_capture_median=float(np.median(capture[aligned])) if aligned.any() else np.nan,
                response_alignment=float(aligned.mean()) if len(aligned) else np.nan,
                response_n=int(len(aligned)),reference_starts=int((refs&valid).sum())))
    out=pd.DataFrame(results)
    if 'resolution' in bb.attrs:
        step=pd.Timedelta(bb.attrs['resolution'])
    else:
        step=df.index.to_series().diff().dropna().median()
    elapsed_days= sum(eligible_period)*step/pd.Timedelta('1D')
    out['crossing_per_day']=out.crossings/elapsed_days
    return out

def compare_shared(surfaces, objective='containment', target=0.90, max_width_pct=None):
    """Find closest target containment (penalize width as tie-break only).
    This is descriptive screening, NOT an independently validated optimum.
    Shared pair minimizes mean rank across timeframes; individual pair
    minimizes each timeframe ranking. No forward returns used.
    """
    tables=[]
    for tf, d in surfaces.items():
        a=d.copy()
        a['tf']=tf
        a['containment_error']=(a.containment-target).abs()
        if max_width_pct is not None:
            a=a[a.median_width_pct<=max_width_pct].copy()
        a['rank']=a['containment_error'].rank(method='average')+0.01*a.median_width_pct.rank(method='average')
        tables.append(a)
    all_=pd.concat(tables,ignore_index=True)
    best_each=all_.sort_values(['tf','rank']).groupby('tf',sort=True).head(1)
    complete=all_.groupby(['window','sigma']).tf.nunique()
    full_pairs=complete[complete.eq(len(surfaces))].index
    common=all_.set_index(['window','sigma']).loc[full_pairs].reset_index()
    shared=common.groupby(['window','sigma'],as_index=False).agg(mean_rank=('rank','mean'),worst_rank=('rank','max')).sort_values(['mean_rank','worst_rank']).head(1)
    return best_each,shared,all_

def plot_heatmap(df,metric,title='',ax=None):
    import matplotlib.pyplot as plt
    if ax is None: _,ax=plt.subplots(figsize=(9,5))
    pivot=df.pivot(index='window',columns='sigma',values=metric).sort_index()
    im=ax.imshow(pivot.to_numpy(),origin='lower',aspect='auto')
    ax.set_xticks(range(len(pivot.columns)),[str(v) for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)),[str(v) for v in pivot.index])
    ax.set_xlabel('SD multiplier');ax.set_ylabel('Lookback (bars)')
    ax.set_title(title or metric)
    ax.figure.colorbar(im,ax=ax,label=metric)
    return ax
