import pandas as pd
from episode_detector import detect_band_episodes

def make_bb(close, lows=None, highs=None, opens=None):
    n=len(close)
    idx=pd.date_range('2025-01-01', periods=n, freq='1h', tz='UTC')
    b=pd.DataFrame({'open':opens if opens is not None else [95.]*n,
                    'low':lows if lows is not None else [90.]*n,
                    'high':highs if highs is not None else [100.]*n,
                    'close':close,'bb_lower':[90.]*n,'bb_upper':[110.]*n},index=idx)
    b.attrs['resolution']='1h'
    return b

def test_band_walk_timeout_no_repeats():
    b=make_bb([95]+[89]*12+[96]*3,opens=[95]*16,lows=[91]+[87]*12+[95]*3)
    e=detect_band_episodes(b,max_horizon=4)
    s=e[e.side=='Support']
    assert len(s)==1, s
    assert s.iloc[0].status=='TIMEOUT'
    assert s.iloc[0].n_bars==4

def test_rearmed_episodes():
    b=make_bb([95,91,96,94,91,97], lows=[93,88,95,93,88,96])
    e=detect_band_episodes(b,max_horizon=8)
    s=e[e.side=='Support']
    assert len(s)==2,s
    assert list(s.status)==['REARMED','REARMED']

def test_gap_censoring():
    b=make_bb([95,89,88,89,96],lows=[95,87,86,87,95])
    b.loc[b.index[2],['open','high','low','close']]=float('nan')
    e=detect_band_episodes(b,max_horizon=8)
    assert e.iloc[0].status=='DATA_GAP_CENSORED'

if __name__=='__main__':
    test_band_walk_timeout_no_repeats()
    test_rearmed_episodes()
    test_gap_censoring()
    print('3 tests passed')
