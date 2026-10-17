import numpy as np
import pandas as pd


def detect_band_episodes(bb, rearm_pct_b=0.25, max_horizon=24):
    """Detect non-overlapping contacts per side against *previously known* bands.

    E0 diagnostic: fixed anchor; NO breakout/rejection claims yet. An episode
    remains active until the side-specific rearm condition at a candle close.
    If it lasts max_horizon bars, its observation is TIMEOUT and stays disarmed
    until rearm. End-of-data active observations are RIGHT_CENSORED.

    Output includes event start, time of terminal observation, rearm time,
    contact count, and maximum anchor-relative excursions. Individual
    observations per side can overlap opposite-side episodes.
    """
    if not isinstance(max_horizon, int) or max_horizon < 1:
        raise ValueError('max_horizon must be a positive integer')
    if not 0 < rearm_pct_b < 0.5:
        raise ValueError('rearm_pct_b must be in (0, 0.5)')
    step = pd.Timedelta(bb.attrs['resolution'])
    df = bb.sort_index().copy()
    if df.empty or df.index.has_duplicates:
        raise ValueError('Expected nonempty data and unique timestamps')
    idx = pd.date_range(df.index.min(), df.index.max(), freq=step, tz='UTC')
    df = df.reindex(idx)
    df.index.name = 'close_time_utc'
    df['known_lower'] = df.bb_lower.shift(1)
    df['known_upper'] = df.bb_upper.shift(1)
    valid = df[['open','high','low','close','known_lower','known_upper']].notna().all(axis=1) & (df.known_upper > df.known_lower)
    rows = []
    for side in ['Support','Resistance']:
        is_support = side == 'Support'
        level_col = 'known_lower' if is_support else 'known_upper'
        mode = 'READY'
        active = None
        episode_seq = 0
        for t, r in df.iterrows():
            if not valid.loc[t]:
                if mode == 'ACTIVE':
                    active['status'] = 'DATA_GAP_CENSORED'
                    active['observation_end'] = t - step
                    rows.append(active)
                mode, active = 'READY', None  # No linking across gaps
                continue
            level = r[level_col]
            pb = (r.close - r.known_lower) / (r.known_upper - r.known_lower)
            # Rearm is intentionally NOT an event outcome: it just marks end of one contact cycle.
            rearmed = pb >= rearm_pct_b if is_support else pb <= 1 - rearm_pct_b
            if mode == 'DISARMED':
                if rearmed:
                    mode = 'READY'
                continue
            if mode == 'READY':
                touch = (r.open > level and r.low <= level) if is_support else (r.open < level and r.high >= level)
                if not touch:
                    continue
                episode_seq += 1
                active = dict(side=side, episode_id=f'{side[0]}{episode_seq:04d}', start=t,
                              anchor=float(level), n_bars=0, n_contacts=0,
                              max_adverse_pct=0.0, max_favorable_pct=0.0,
                              status=None, observation_end=None, rearm_time=pd.NaT)
                mode = 'ACTIVE'
            if mode != 'ACTIVE':
                continue
            active['n_bars'] += 1
            anchor = active['anchor']
            touch = r.low <= level if is_support else r.high >= level
            active['n_contacts'] += int(touch)
            adverse = 100 * ((anchor-r.low)/anchor if is_support else (r.high-anchor)/anchor)
            favorable = 100 * ((r.high-anchor)/anchor if is_support else (anchor-r.low)/anchor)
            active['max_adverse_pct'] = max(active['max_adverse_pct'], float(adverse))
            active['max_favorable_pct'] = max(active['max_favorable_pct'], float(favorable))
            if rearmed:
                active.update(status='REARMED', observation_end=t, rearm_time=t)
                rows.append(active)
                mode, active = 'READY', None
            elif active['n_bars'] >= max_horizon:
                active.update(status='TIMEOUT', observation_end=t)
                rows.append(active)
                mode, active = 'DISARMED', None
        if active is not None:
            active.update(status='RIGHT_CENSORED', observation_end=df.index[-1])
            rows.append(active)
    cols=['side','episode_id','start','anchor','n_bars','n_contacts',
          'max_adverse_pct','max_favorable_pct','status','observation_end','rearm_time']
    return pd.DataFrame(rows, columns=cols).sort_values(['start','side']).reset_index(drop=True)


def episode_summary(episodes):
    return episodes.groupby('side').agg(
        episodes=('episode_id','size'),
        median_bars=('n_bars','median'),
        total_contacts=('n_contacts','sum'),
        rearmed=('status',lambda x: int((x=='REARMED').sum())),
        timeouts=('status',lambda x: int((x=='TIMEOUT').sum())),
        right_censored=('status',lambda x: int((x=='RIGHT_CENSORED').sum())),
        gap_censored=('status',lambda x: int((x=='DATA_GAP_CENSORED').sum())),
    )
