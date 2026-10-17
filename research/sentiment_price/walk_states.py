"""E0.2a: causal Bollinger walk-state prototype. All indices are candle CLOSE UTC."""
import numpy as np
import pandas as pd

STATES = ('NEUTRAL','LOWER_WALK','UPPER_WALK')

def bollinger_from_5m(bars, start, end, resolution='1h', window=20, num_std=2.0):
    """Complete OHLCV bars, with close-time index. Retains warm-up before start."""
    step, base = pd.Timedelta(resolution), pd.Timedelta('5min')
    if step < base or step % base: raise ValueError('resolution must be multiple of 5min')
    if window < 2: raise ValueError('window >=2 required')
    if not np.isfinite(num_std) or num_std <= 0: raise ValueError('num_std > 0 required')
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    start = start.tz_localize('UTC') if start.tz is None else start.tz_convert('UTC')
    end = end.tz_localize('UTC') if end.tz is None else end.tz_convert('UTC')
    if start >= end: raise ValueError('start >= end')
    cols=['open','high','low','close','volume']
    src=bars[cols].copy()
    src.index=pd.to_datetime(src.index, utc=True)
    src=src.sort_index()
    if src.empty or src.index.has_duplicates or (src.index.asi8%base.value != 0).any():
        raise ValueError('missing, duplicated or misaligned source timestamps')
    src=src.loc[src.index+base<=end]
    valid=src.notna().all(axis=1)
    agg=src.resample(step, closed='left',label='left',origin='epoch').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
    n=valid.astype(int).resample(step,closed='left',label='left',origin='epoch').sum()
    # Exact timestamp coverage in each bin, not just non-null count.
    present=src.assign(_present=1)['_present'].resample(step,closed='left',label='left',origin='epoch').sum()
    complete=n.eq(int(step/base)) & present.eq(int(step/base))
    agg.loc[~complete, cols]=np.nan
    mean=agg.close.rolling(window,min_periods=window).mean()
    std=agg.close.rolling(window,min_periods=window).std(ddof=0)
    agg['bb_middle']=mean
    agg['bb_upper']=mean+num_std*std
    agg['bb_lower']=mean-num_std*std
    mask=(agg.index>=start)&(agg.index+step<=end)
    out=agg.loc[mask].copy()
    out.index+=step
    out.index.name='close_time_utc'
    out.attrs.update(resolution=resolution,window=window,num_std=num_std)
    return out


def detect_walk_states(bb, near=0.15, confirm_bars=3, exit_bars=2, slope_bars=3):
    """Past-only confirmed walking states. Entry when last confirm_bars near a band,
    band slope over slope_bars is directional; exit when exit_bars consecutive
    candles cease to be near the band. No retrospective state backfilling.
    Note: fixed bar counts deliberately have different elapsed time by resolution.
    """
    if not (0 < near < .5): raise ValueError('near must be within (0,.5)')
    if any(not isinstance(v,int) or v<1 for v in (confirm_bars,exit_bars,slope_bars)):
        raise ValueError('integer counts >=1 required')
    df=bb[['close','bb_lower','bb_upper']].copy().sort_index()
    step=pd.Timedelta(bb.attrs['resolution'])
    if df.index.has_duplicates or df.empty: raise ValueError('bad index')
    # Explicit missing scheduled candles: a gap invalidates persistence.
    full_idx=pd.date_range(df.index.min(),df.index.max(),freq=step,tz='UTC')
    df=df.reindex(full_idx)
    width=df.bb_upper-df.bb_lower
    ok=(df.close.notna()&df.bb_lower.notna()&df.bb_upper.notna()&(width>0))
    pct_b=((df.close-df.bb_lower)/width).where(ok)
    lo=(pct_b<=near).where(ok,False)
    hi=(pct_b>=1-near).where(ok,False)
    lower_slope=(df.bb_lower-df.bb_lower.shift(slope_bars)).where(ok)
    upper_slope=(df.bb_upper-df.bb_upper.shift(slope_bars)).where(ok)
    state='NEUTRAL'; cand=''; streak=0; escape=0; state_age=0
    states=[]; transitions=[]; candidates=[]; ages=[]; enter_times=[]
    entered=pd.NaT
    for i,t in enumerate(df.index):
        if not bool(ok.iloc[i]):
            if state!='NEUTRAL': transitions.append((t,state,'NEUTRAL','DATA_GAP'))
            state='NEUTRAL'; cand=''; streak=0; escape=0; state_age=0; entered=pd.NaT
        else:
            proposed='LOWER_WALK' if bool(lo.iloc[i]) and pd.notna(lower_slope.iloc[i]) and lower_slope.iloc[i]<0 else ('UPPER_WALK' if bool(hi.iloc[i]) and pd.notna(upper_slope.iloc[i]) and upper_slope.iloc[i]>0 else '')
            if state=='NEUTRAL':
                if proposed and proposed==cand: streak+=1
                elif proposed: cand=proposed; streak=1
                else: cand=''; streak=0
                if streak>=confirm_bars:
                    old=state; state=cand; transitions.append((t,old,state,'CONFIRMED'))
                    entered=t; state_age=0; escape=0; cand=''; streak=0
            else:
                # A walk persists despite band slope flattening; only distance from band exits.
                still_near=bool(lo.iloc[i]) if state=='LOWER_WALK' else bool(hi.iloc[i])
                escape=0 if still_near else escape+1
                if escape>=exit_bars:
                    transitions.append((t,state,'NEUTRAL','DISTANCE_EXIT'))
                    state='NEUTRAL'; entered=pd.NaT; state_age=0;escape=0
                else: state_age+=1
        states.append(state); candidates.append(cand or None); ages.append(state_age)
        enter_times.append(entered)
    df['pct_b']=pct_b
    df['lower_slope']=lower_slope
    df['upper_slope']=upper_slope
    df['state']=states
    df['candidate']=candidates
    df['state_age_bars']=ages
    df['walk_confirmed_at']=pd.to_datetime(enter_times,utc=True)
    events=pd.DataFrame(transitions,columns=['time','from_state','to_state','reason'])
    return df,events
