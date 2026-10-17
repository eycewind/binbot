import numpy as np
import pandas as pd
from e03a_trace import audit_walk_states
from walk_states import detect_walk_states


def sample(seed=0):
    rng=np.random.default_rng(seed)
    idx=pd.date_range('2025-01-01',periods=320,freq='1h',tz='UTC')
    x=np.cumsum(rng.normal(0,1,320))+100
    df=pd.DataFrame({'close':x,'bb_lower':x-1.5+rng.normal(0,.3,320),
                     'bb_upper':x+1.5+rng.normal(0,.3,320)},index=idx)
    df.attrs['resolution']='1h'
    return df


def test_replay_parity_multiple_seeds():
    for seed in range(12):
        df=sample(seed)
        for confirm in (1,3,5):
            states,events,trace,conf=audit_walk_states(df,confirm_bars=confirm)
            assert states.state.equals(trace.state_after.rename('state'))
            assert (conf.confirmation_lag_hours>=0).all() if len(conf) else True


def test_gap_resets_and_parity():
    df=sample(); df.iloc[42,df.columns.get_loc('close')]=np.nan
    _,_,trace,_=audit_walk_states(df)
    assert not trace.iloc[42].valid
    assert trace.iloc[42].state_after=='NEUTRAL'


def test_no_backfill_and_one_bar_candidate():
    df=sample()
    _,_,trace,conf=audit_walk_states(df,confirm_bars=1)
    if len(conf):
        assert (conf.time==conf.candidate_since).all()
