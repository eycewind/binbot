"""E03: causal cross-resolution walk-state alignment and descriptive diagnostics."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PALETTE = {'LOWER_WALK':'#d95b59','UPPER_WALK':'#38a56a','NEUTRAL':'#d1d5db'}
VALID = tuple(PALETTE)


def _utc_index(index):
    idx = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    if not idx.is_monotonic_increasing or idx.has_duplicates:
        raise ValueError('Expected sorted unique timestamps')
    return idx


def align_to_5m(runs, resolutions=None):
    """Align completed-candle states to a common 5m close-time grid.

    A higher-timeframe state is visible only once its candle closes. Missing
    scheduled source observations remain unknown (NA), not stale state.
    """
    resolutions = resolutions or {'4H':'4h','1H':'1h','15m':'15min','5m':'5min'}
    base = runs['5m']['states']
    base_idx = _utc_index(base.index)
    joined = pd.DataFrame(index=base_idx)
    for label, resolution in resolutions.items():
        src = runs[label]['states']
        src_idx = _utc_index(src.index)
        if 'state' not in src: raise KeyError(f'{label}: missing state column')
        step = pd.Timedelta(resolution)
        series = pd.Series(src['state'].to_numpy(), index=src_idx, name=label)
        # States at missing scheduled candles are unknown; forward fill permitted
        # only for the interval prior to the NEXT scheduled candle close.
        expected = pd.date_range(src_idx.min(),src_idx.max(),freq=step,tz='UTC')
        series = series.reindex(expected)
        joined[label] = series.reindex(base_idx, method='ffill')
        last_known = pd.Series(expected, index=expected).reindex(base_idx,method='ffill')
        stale = (base_idx - pd.DatetimeIndex(last_known)) >= step
        joined.loc[stale,label] = pd.NA
        # No information from before first observed state.
    joined.index.name='close_time_utc'
    return joined


def occupancy(aligned):
    rows=[]
    for label in aligned:
        s=aligned[label]
        for state in VALID:
            rows.append({'timeframe':label,'state':state,'count_5m':int(s.eq(state).sum()),
                         'fraction':float(s.eq(state).sum()/s.notna().sum()) if s.notna().any() else np.nan})
    return pd.DataFrame(rows).set_index(['timeframe','state'])


def next_walk_probabilities(states, label):
    """Next confirmed walk direction, skipping NEUTRAL time segments.

    Consecutive bars in the same confirmed walk are one occurrence, not
    repeated transition samples. WALK -> NEUTRAL -> same WALK counts as two.
    """
    s=states['state'].copy()
    idx=_utc_index(states.index)
    s.index=idx
    s=s.where(s.isin(VALID))
    starts=s[(s!=s.shift(1)) & s.isin(('LOWER_WALK','UPPER_WALK'))]
    pairs=pd.DataFrame({'previous':starts.iloc[:-1].to_numpy(),
                        'next':starts.iloc[1:].to_numpy()})
    if pairs.empty:
        return pd.DataFrame(columns=['timeframe','previous','next','count','probability'])
    counts=pairs.groupby(['previous','next'],observed=True).size().rename('count').reset_index()
    counts['probability']=counts['count']/counts.groupby('previous')['count'].transform('sum')
    counts.insert(0,'timeframe',label)
    return counts


def cross_overlap(aligned, pairs=(('4H','1H'),('1H','15m'),('15m','5m'))):
    """P(fine state | coarse state) at common candle-close timestamps.
    Descriptive occupancy only; not independent event probabilities.
    """
    rows=[]
    for coarse,fine in pairs:
        tmp=aligned[[coarse,fine]].dropna()
        for c in VALID:
            subset=tmp.loc[tmp[coarse].eq(c),fine]
            for f in VALID:
                rows.append({'coarse':coarse,'fine':fine,'coarse_state':c,
                             'fine_state':f,'n_5m':len(subset),
                             'overlap_count':int(subset.eq(f).sum()),
                             'fraction':float(subset.eq(f).mean()) if len(subset) else np.nan})
    return pd.DataFrame(rows)


def conditional_lift(overlap, occupancy_table):
    """Compare conditional state frequency with fine-timeframe baseline.
    Not a causal or statistically significant effect estimate.
    """
    out=overlap.copy()
    baseline=[]
    for r in out.itertuples():
        baseline.append(float(occupancy_table.loc[(r.fine,r.fine_state),'fraction']))
    out['baseline_fraction']=baseline
    out['lift']=out['fraction']/out['baseline_fraction'].replace(0,np.nan)
    return out


def _spans(series, start, end, step):
    s=series.loc[(series.index>=start)&(series.index<end)]
    if s.empty:return []
    vals=s.fillna('UNKNOWN')
    changes=(vals.ne(vals.shift()) | (vals.index.to_series().diff().gt(step)))
    ids=changes.cumsum()
    out=[]
    for _,g in vals.groupby(ids):
        state=g.iloc[0]
        if state in ('LOWER_WALK','UPPER_WALK'):
            out.append((max(g.index[0]-step,start),min(g.index[-1],end),state))
    return out


def plot_pair(runs, aligned, coarse, fine, start, end, figsize=(16,7)):
    """Price with fine-resolution bands; two non-overlapping state strips.
    Coarse states are aligned *causally* to the fine-timeframe close grid.
    """
    start=pd.Timestamp(start);end=pd.Timestamp(end)
    start=start.tz_localize('UTC') if start.tzinfo is None else start.tz_convert('UTC')
    end=end.tz_localize('UTC') if end.tzinfo is None else end.tz_convert('UTC')
    if end<=start:raise ValueError('end must be after start')
    bb=runs[fine]['bb']
    data=bb.loc[(bb.index>=start)&(bb.index<end)]
    if data.empty:raise ValueError('No candles in requested window')
    a=aligned.loc[(aligned.index>=start)&(aligned.index<end)]
    step=pd.Timedelta('5min')
    fig,(ax,ax_state)=plt.subplots(2,1,figsize=figsize,sharex=True,
                                     gridspec_kw={'height_ratios':[4,1]},layout='constrained')
    ax.plot(data.index,data['close'],color='black',lw=1,label='BTC close')
    ax.plot(data.index,data['bb_middle'],color='#ed8a39',lw=.8,label=f'{fine} middle')
    ax.plot(data.index,data['bb_upper'],color='#2779bf',lw=.8,label=f'{fine} upper/lower')
    ax.plot(data.index,data['bb_lower'],color='#2779bf',lw=.8)
    for lo,hi,state in _spans(a[coarse],start,end,step):
        ax.axvspan(lo,hi,color=PALETTE[state],alpha=.13,zorder=-5)
    for j,tf in enumerate((coarse,fine)):
        for lo,hi,state in _spans(a[tf],start,end,step):
            ax_state.broken_barh([(lo,hi-lo)],(1-j-.35,.7),facecolors=PALETTE[state],alpha=.85)
    ax_state.set_ylim(-.45,1.45)
    ax_state.set_yticks([1,0],labels=[coarse,fine]);ax_state.grid(axis='x',alpha=.2)
    ax.set_ylabel('BTCUSDT');ax.grid(alpha=.2);ax.legend(loc='upper right')
    ax.set_title(f'E03 walk-state comparison: {coarse} vs {fine} (UTC)')
    ax_state.legend(handles=[Patch(color=PALETTE['LOWER_WALK'],label='LOWER_WALK'),
                             Patch(color=PALETTE['UPPER_WALK'],label='UPPER_WALK')],loc='upper right',ncol=2)
    ax_state.set_xlabel('Completed candle close (UTC)');ax.set_xlim(start,end)
    return fig,(ax,ax_state)
