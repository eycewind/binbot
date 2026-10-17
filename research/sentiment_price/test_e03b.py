import pandas as pd
import numpy as np
from e03b_analysis import event_table, walk_survival

def fixture():
    idx=pd.date_range('2025-01-01',periods=8,freq='1h',tz='UTC',name='close_time_utc')
    states=pd.DataFrame({'close':[100,99,98,97,96,95,94,93], 'pct_b':[.1]*8,
                         'state':['NEUTRAL','NEUTRAL','LOWER_WALK','LOWER_WALK','NEUTRAL','LOWER_WALK','LOWER_WALK','NEUTRAL'],
                         'state_age_bars':[0,0,0,1,0,0,1,0]},index=idx)
    trace=pd.DataFrame({'decision':['NONE','CANDIDATE','CONFIRMED','ACTIVE','DISTANCE_EXIT','CONFIRMED','ACTIVE','DISTANCE_EXIT'],
                        'candidate_streak':[0,1,3,0,0,3,0,0],
                        'state_before':['NEUTRAL','NEUTRAL','NEUTRAL','LOWER_WALK','LOWER_WALK','NEUTRAL','LOWER_WALK','LOWER_WALK'],
                        'state_after':states.state,'proposed':['NONE','LOWER_WALK','LOWER_WALK','NONE','NONE','LOWER_WALK','NONE','NONE']},index=idx)
    return states,trace

def test_events_and_direction():
    s,t=fixture(); e=event_table(s,t,horizons=(1,3))
    assert len(e)==5
    assert e.iloc[0].event=='CANDIDATE'
    assert np.isclose(e.iloc[0]['ret_1b_pct'], 100/99)
    assert np.isnan(e.iloc[-1]['ret_1b_pct'])

def test_survival_tracks_episode_not_state():
    s,t=fixture(); w=walk_survival(s,horizons=(1,3))
    assert w.loc[w.time==s.index[2],'survives_1b'].iloc[0] == True
    assert w.loc[w.time==s.index[2],'survives_3b'].iloc[0] == False
    assert pd.isna(w.loc[w.time==s.index[6],'survives_3b'].iloc[0])

def test_gap_censors_return():
    s,t=fixture();s.loc[s.index[3],'close']=np.nan
    e=event_table(s,t,horizons=(3,))
    assert pd.isna(e.loc[e.time==s.index[1], 'ret_3b_pct'].iloc[0])
