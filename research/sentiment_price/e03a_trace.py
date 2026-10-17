"""E0.3a: non-invasive instrumentation for frozen E0.2 walk detector.

Calls the baseline, independently replays its exact state logic with diagnostic
counters, then asserts state/transition identity. All timestamps are candle CLOSE.
"""
import numpy as np
import pandas as pd
from walk_states import detect_walk_states


def audit_walk_states(bb, near=0.15, confirm_bars=3, exit_bars=2, slope_bars=3):
    baseline, original_events = detect_walk_states(
        bb, near=near, confirm_bars=confirm_bars,
        exit_bars=exit_bars, slope_bars=slope_bars
    )
    df = baseline.copy()
    ok = df[['close','bb_lower','bb_upper','pct_b']].notna().all(axis=1) & (df.bb_upper > df.bb_lower)
    near_lo = (df.pct_b <= near) & ok
    near_hi = (df.pct_b >= 1-near) & ok
    slope_lo = (df.lower_slope < 0) & df.lower_slope.notna()
    slope_hi = (df.upper_slope > 0) & df.upper_slope.notna()
    qualifies_lo = near_lo & slope_lo
    qualifies_hi = near_hi & slope_hi
    state = 'NEUTRAL'; cand = ''; streak = 0; escape = 0
    candidate_since = pd.NaT
    records = []
    for t in df.index:
        valid=bool(ok.loc[t]); nl=bool(near_lo.loc[t]); nh=bool(near_hi.loc[t]);
        proposed = ('LOWER_WALK' if bool(qualifies_lo.loc[t]) else
                    ('UPPER_WALK' if bool(qualifies_hi.loc[t]) else ''))
        before=state; reason=''; current_streak=0; current_escape=0
        prior_candidate_start=candidate_since
        if not valid:
            if state!='NEUTRAL': reason='DATA_GAP'
            state='NEUTRAL'; cand=''; streak=0; escape=0;candidate_since=pd.NaT
        elif state=='NEUTRAL':
            if proposed and proposed==cand:
                streak+=1
            elif proposed:
                cand=proposed; streak=1;candidate_since=t
            else:
                cand=''; streak=0;candidate_since=pd.NaT
            current_streak=streak
            if streak >= confirm_bars:
                state=cand;reason='CONFIRMED'
                prior_candidate_start=candidate_since
                cand='';streak=0;escape=0;candidate_since=pd.NaT
        else:
            still_near = nl if state=='LOWER_WALK' else nh
            escape=0 if still_near else escape+1
            current_escape=escape
            if escape>=exit_bars:
                state='NEUTRAL';reason='DISTANCE_EXIT';escape=0
        records.append({
            'time':t,'valid':valid,'near_lower':nl,'near_upper':nh,
            'lower_slope_ok':bool(slope_lo.loc[t]),'upper_slope_ok':bool(slope_hi.loc[t]),
            'qualify_lower':bool(qualifies_lo.loc[t]),'qualify_upper':bool(qualifies_hi.loc[t]),
            'proposed':proposed or 'NONE','state_before':before,'state_after':state,
            'candidate_streak':current_streak,'exit_streak':current_escape,
            'candidate_since':prior_candidate_start if reason=='CONFIRMED' else candidate_since,
            'decision':reason or ('CANDIDATE' if state=='NEUTRAL' and cand else
                                 'ACTIVE' if state!='NEUTRAL' else 'NONE'),
        })
    trace=pd.DataFrame.from_records(records).set_index('time')
    trace.index.name=df.index.name
    pd.testing.assert_series_equal(trace.state_after.rename('state'),df.state,check_names=True,check_freq=False)
    replay_events=trace.rename_axis('time').loc[trace.decision.isin(['CONFIRMED','DATA_GAP','DISTANCE_EXIT']) &
                            trace.state_before.ne(trace.state_after),
                            ['state_before','state_after','decision']].reset_index()
    replay_events.columns=['time','from_state','to_state','reason']
    pd.testing.assert_frame_equal(replay_events.reset_index(drop=True),
                                  original_events.reset_index(drop=True),check_dtype=False)
    confirmations=trace.loc[trace.decision=='CONFIRMED',
                            ['state_after','candidate_since']].rename_axis('time').reset_index()
    if not confirmations.empty:
        confirmations['confirmation_lag_hours']=(
            confirmations['time']-confirmations['candidate_since'])/pd.Timedelta('1h')
        confirmations['state']=confirmations['state_after']
    return df, original_events, trace, confirmations


def trace_summary(trace, confirmations):
    out=pd.Series({
        'valid_candles':int(trace.valid.sum()),
        'invalid_candles':int((~trace.valid).sum()),
        'lower_qualifying_candles':int(trace.qualify_lower.sum()),
        'upper_qualifying_candles':int(trace.qualify_upper.sum()),
        'confirmed_walks':int((trace.decision=='CONFIRMED').sum()),
        'distance_exits':int((trace.decision=='DISTANCE_EXIT').sum()),
        'gap_exits':int((trace.decision=='DATA_GAP').sum()),
        'median_confirmation_lag_hours':confirmations.confirmation_lag_hours.median() if len(confirmations) else np.nan,
    })
    return out.to_frame('value')