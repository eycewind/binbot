"""E0.3b: descriptive, causal-origin walk studies. NO trading signals.
Forward labels are future-looking research outcomes, never model features.
"""
import numpy as np
import pandas as pd

HORIZONS = (1, 3, 6, 12, 24)


def event_table(states, trace, *, horizons=HORIZONS):
    """Candidate starts (including unconfirmed), confirmations, and exits.

    Label directional close-to-close returns from the event candle close;
    the event close is NOT an executable fill price. Reject future gaps.
    """
    states = states.sort_index()
    trace = trace.reindex(states.index)
    if not states.index.equals(trace.index):
        raise ValueError('trace and states index mismatch')
    step = pd.tseries.frequencies.to_offset(pd.infer_freq(states.index)) if pd.infer_freq(states.index) else None
    if step is None:
        # Can pass a reindexed schedule to remove ambiguity.
        raise ValueError('states must have a regular complete timestamp index')
    valid = states['close'].notna() & states['pct_b'].notna()
    flags = (trace['decision'].eq('CANDIDATE') & trace['candidate_streak'].eq(1)) | trace['decision'].isin(['CONFIRMED', 'DISTANCE_EXIT'])
    rows = []
    for pos in np.flatnonzero(flags.to_numpy()):
        event = trace.iloc[pos]
        kind = 'CANDIDATE' if event['decision'] == 'CANDIDATE' else event['decision']
        if kind == 'DISTANCE_EXIT':
            direction = event['state_before']
        elif kind == 'CONFIRMED':
            direction = event['state_after']
        else:
            direction = event['proposed']
        if direction not in ('LOWER_WALK', 'UPPER_WALK'):
            continue
        sign = -1 if direction == 'LOWER_WALK' else 1
        t = states.index[pos]
        rec = dict(time=t, event=kind, direction=direction, close=float(states['close'].iloc[pos]), state_age_bars=int(states['state_age_bars'].iloc[pos]))
        for h in horizons:
            key = f'ret_{h}b_pct'
            if pos + h >= len(states) or not valid.iloc[pos:pos+h+1].all():
                rec[key] = np.nan
            else:
                rec[key] = sign * 100.0 * (float(states['close'].iloc[pos+h])/rec['close'] - 1)
        rows.append(rec)
    return pd.DataFrame(rows)


def walk_survival(states, *, horizons=(1, 3, 6)):
    """Each ACTIVE confirmed-walk snapshot: was it still same walk h bars later?
    No bridging across invalid data; every snapshot is overlapping/dependent.
    Exits in between count as ended, even if the same direction restarts.
    """
    states = states.sort_index()
    valid = states[['close','pct_b']].notna().all(axis=1)
    s = states['state'].to_numpy()
    # Walk run identities (new identity after every state change)
    ids = states['state'].ne(states['state'].shift(1)).cumsum().to_numpy()
    rows=[]
    for pos in np.flatnonzero(np.isin(s, ['LOWER_WALK','UPPER_WALK'])):
        rec=dict(time=states.index[pos], direction=s[pos], age_bars=int(states['state_age_bars'].iloc[pos]))
        for h in horizons:
            rec[f'survives_{h}b'] = (bool(ids[pos+h] == ids[pos] and valid.iloc[pos:pos+h+1].all()) if pos+h<len(states) and valid.iloc[pos:pos+h+1].all() else pd.NA)
        rows.append(rec)
    return pd.DataFrame(rows)


def summaries(events, survival, horizons=HORIZONS):
    ec = [f'ret_{h}b_pct' for h in horizons]
    ret = events.groupby(['event','direction'])[ec].agg(['count','median','mean']) if len(events) else pd.DataFrame()
    if survival.empty:
        return ret, pd.DataFrame()
    s = survival.copy()
    # coarse age bins for initial exploration, not optimized thresholds
    s['age_bin'] = pd.cut(s['age_bars'], [-1,2,5,11,np.inf], labels=['0-2','3-5','6-11','12+'])
    cols = [c for c in s if c.startswith('survives_')]
    for c in cols: s[c] = s[c].astype('Float64')
    surv = s.groupby(['direction','age_bin'], observed=True)[cols].agg(['count','mean'])
    return ret, surv
