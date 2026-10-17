import numpy as np
import pandas as pd
from e03_analysis import align_to_5m, next_walk_probabilities, cross_overlap

def fixture():
    base=pd.date_range('2025-01-01 00:05',periods=24,freq='5min',tz='UTC')
    def mk(idx,state):return {'states':pd.DataFrame({'state':state},index=idx)}
    return {'5m':mk(base,['NEUTRAL']*24),
            '15m':mk(pd.date_range('2025-01-01 00:15',periods=8,freq='15min',tz='UTC'),['LOWER_WALK']*8),
            '1H':mk(pd.DatetimeIndex(['2025-01-01 01:00Z','2025-01-01 02:00Z']),['UPPER_WALK','NEUTRAL']),
            '4H':mk(pd.DatetimeIndex(['2025-01-01 00:00Z']),['LOWER_WALK'])}

def test_completed_candles_no_lookahead():
    a=align_to_5m(fixture())
    assert pd.isna(a.loc['2025-01-01 00:55Z','1H'])
    assert a.loc['2025-01-01 01:00Z','1H']=='UPPER_WALK'
    assert pd.isna(a.loc['2025-01-01 00:10Z','15m'])
    assert a.loc['2025-01-01 00:15Z','15m']=='LOWER_WALK'

def test_no_stale_4h():
    a=align_to_5m(fixture())
    assert a.loc['2025-01-01 01:00Z','4H']=='LOWER_WALK'

def test_count_walks_not_bars():
    idx=pd.date_range('2025-01-01',periods=8,freq='1h',tz='UTC')
    df=pd.DataFrame({'state':['NEUTRAL','LOWER_WALK','LOWER_WALK','NEUTRAL',
                              'LOWER_WALK','NEUTRAL','UPPER_WALK','NEUTRAL']},index=idx)
    t=next_walk_probabilities(df,'1H')
    assert t.loc[(t.previous=='LOWER_WALK') & (t.next=='LOWER_WALK'),'count'].iloc[0]==1
    assert t.loc[(t.previous=='LOWER_WALK') & (t.next=='UPPER_WALK'),'count'].iloc[0]==1

def test_overlap_is_conditionally_normalized():
    a=align_to_5m(fixture())
    o=cross_overlap(a,pairs=[('1H','15m')])
    x=o.loc[(o.coarse_state=='UPPER_WALK')]
    assert np.isclose(x.fraction.sum(),1)
