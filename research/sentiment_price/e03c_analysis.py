"""E0.3c: offline movement references and causal Bollinger detector comparisons.
All indexes are completed UTC candle closes. Reference labels are RETROSPECTIVE,
not usable as real-time detector inputs. Evaluate by timeframe separately.
"""
import numpy as np
import pandas as pd

VARIANTS = ('V0_frozen','V1_one_bar','V2_near_no_slope','V3_prior_band_cross','V4_adaptive')

def _index(df):
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError('Require sorted, unique candle-close timestamps')
    if df.index.tz is None:
        raise ValueError('Require UTC timezone-aware timestamps')

def causal_atr(bb, period=14):
    """Past-only Wilder-style ATR, known at candle close. Shift for pre-candle use."""
    prev=bb.close.shift(1)
    tr=pd.concat([(bb.high-bb.low).abs(),(bb.high-prev).abs(),(bb.low-prev).abs()],axis=1).max(axis=1)
    tr=tr.where(bb[['high','low','close']].notna().all(axis=1) & prev.notna())
    return tr.ewm(alpha=1/period,adjust=False,min_periods=period,ignore_na=False).mean()

def price_impulse_references(bb, lookback=3, threshold_atr=1.5, cooldown_bars=6):
    """Retrospective direction-specific impulses independent of Bollinger bands.
    A reference is confirmed when displacement from lookback close crosses k*ATR
    known at start of movement. Its onset is the lookback timestamp, hence hindsight.
    We use confirmation_time for a conservative available-by time, onset for latency.
    Disjoint references (by cooldown after confirmation) avoid repeated labels.
    """
    _index(bb)
    if lookback<1 or threshold_atr<=0 or cooldown_bars<0: raise ValueError('bad parameters')
    atr=causal_atr(bb).shift(lookback)
    displacement=bb.close-bb.close.shift(lookback)
    ok=displacement.notna() & atr.gt(0) & bb.close.notna()
    records=[]; last=-10**10
    for i in np.flatnonzero(ok.to_numpy()):
        if i-last<=cooldown_bars: continue
        score=displacement.iloc[i]/atr.iloc[i]
        if abs(score)>=threshold_atr:
            records.append(dict(ref_id=len(records), direction='UPPER_WALK' if score>0 else 'LOWER_WALK',
                onset=bb.index[i-lookback], detected_at=bb.index[i], magnitude_atr=float(abs(score)),
                onset_index=i-lookback, confirmation_index=i))
            last=i
    return pd.DataFrame(records,columns=['ref_id','direction','onset','detected_at','magnitude_atr','onset_index','confirmation_index'])

def _runs(mask, n):
    streak=0; out=np.zeros(len(mask),dtype=bool)
    for i,v in enumerate(mask):
        streak=streak+1 if v else 0
        out[i]=streak>=n and (i==0 or not out[i-1])
    return out

def candidate_alarms(bb, variant='V0_frozen', near=.15, confirm_bars=3,
                    slope_bars=3, adaptive_lookback=200, adaptive_quantile=.9,
                    min_spacing_bars=2):
    """One alarm at each qualifying-run onset; V0 uses baseline confirm transitions.
    V4 uses trailing, SHIFTED quantile of positive band penetration (no future).
    """
    _index(bb)
    if variant not in VARIANTS: raise ValueError('Unknown variant')
    if min_spacing_bars<0: raise ValueError('min_spacing_bars')
    width=bb.bb_upper-bb.bb_lower
    valid=bb[['close','bb_upper','bb_lower']].notna().all(axis=1)&width.gt(0)
    pb=(bb.close-bb.bb_lower)/width
    lo=(pb<=near)&valid; hi=(pb>=1-near)&valid
    if variant=='V0_frozen':
        from walk_states import detect_walk_states
        baseline,ev=detect_walk_states(bb,near=near,confirm_bars=confirm_bars,slope_bars=slope_bars)
        conf=ev.loc[ev.reason.eq('CONFIRMED')].copy()
        if conf.empty: return pd.DataFrame(columns=['time','direction','variant','bar_index'])
        loc=bb.index.get_indexer(pd.DatetimeIndex(conf.time))
        return pd.DataFrame({'time':conf.time.to_numpy(),'direction':conf.to_state.to_numpy(),
                             'variant':variant,'bar_index':loc}).reset_index(drop=True)
    elif variant=='V1_one_bar':
        sl=bb.bb_lower.diff(slope_bars)<0; su=bb.bb_upper.diff(slope_bars)>0
        lo=lo&sl;hi=hi&su
    elif variant=='V2_near_no_slope':
        pass
    elif variant=='V3_prior_band_cross':
        # open/close of candle compared with previously known band
        valid=valid&bb[['open']].notna().all(axis=1)
        lo=valid & bb.open.gt(bb.bb_lower.shift(1)) & bb.close.le(bb.bb_lower.shift(1))
        hi=valid & bb.open.lt(bb.bb_upper.shift(1)) & bb.close.ge(bb.bb_upper.shift(1))
    elif variant=='V4_adaptive':
        # past-only empirical tail of excursion beyond bands in channel-width units
        upper_excess=(pb-1).clip(lower=0)
        lower_excess=(-pb).clip(lower=0)
        # Positive-only historical excursions; require >= min periods before valid
        qu=upper_excess.where(upper_excess>0).rolling(adaptive_lookback,min_periods=20).quantile(adaptive_quantile).shift(1)
        ql=lower_excess.where(lower_excess>0).rolling(adaptive_lookback,min_periods=20).quantile(adaptive_quantile).shift(1)
        lo=valid&(-pb).gt(ql)&ql.notna()
        hi=valid&(pb-1).gt(qu)&qu.notna()
    positions=[]; previous={'LOWER_WALK':-999999,'UPPER_WALK':-999999}
    for direction, flag in [('LOWER_WALK',lo),('UPPER_WALK',hi)]:
        indices=np.flatnonzero(_runs(flag.fillna(False).to_numpy(dtype=bool),1))
        for i in indices:
            if i-previous[direction]>min_spacing_bars:
                positions.append((i,direction));previous[direction]=i
    positions.sort()
    return pd.DataFrame([dict(time=bb.index[i],direction=d,variant=variant,bar_index=i)
                         for i,d in positions],columns=['time','direction','variant','bar_index'])

def match_alarms(alarms, references, max_lag_bars=6):
    """Greedy one-to-one matching in chronological order; a reference may be
    detected from retrospective onset to max_lag after onset. Early alarms that
    precede reference onset are NOT counted as matched. Matching is descriptive.
    """
    a=alarms.sort_values('bar_index').copy()
    r=references.sort_values('onset_index').copy()
    used=set(); matches=[]; false=[]
    for ix,row in a.iterrows():
        possible=r.loc[(r.direction==row.direction)&(r.onset_index<=row.bar_index)
            &(r.onset_index+max_lag_bars>=row.bar_index)&~r.ref_id.isin(used)]
        if possible.empty: false.append(ix);continue
        chosen=possible.iloc[0];used.add(chosen.ref_id)
        matches.append(dict(alarm_index=ix,ref_id=chosen.ref_id,lag_bars=int(row.bar_index-chosen.onset_index),
                     score_at_reference=float(chosen.magnitude_atr)))
    matched=pd.DataFrame(matches,columns=['alarm_index','ref_id','lag_bars','score_at_reference'])
    return matched,a.loc[false].copy()

def detector_report(bb, references, variant, max_lag_bars=6, **kwargs):
    alarms=candidate_alarms(bb,variant,**kwargs)
    match, false=match_alarms(alarms,references,max_lag_bars=max_lag_bars)
    # Distinct events, not bar-level rate. Pd and alarm precision over sample.
    n_ref=len(references);n_alarm=len(alarms); m=len(match)
    elapsed_days=(bb.index[-1]-bb.index[0])/pd.Timedelta('1D') if len(bb)>1 else np.nan
    return {'variant':variant,'references':n_ref,'alarms':n_alarm,'matched':m,
        'detection_rate':m/n_ref if n_ref else np.nan,
        'alarm_precision':m/n_alarm if n_alarm else np.nan,
        'unmatched_alarms':len(false),
        'unmatched_per_day':len(false)/elapsed_days if elapsed_days>0 else np.nan,
        'median_lag_bars':match.lag_bars.median() if m else np.nan},alarms,match,false
