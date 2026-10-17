import numpy as np
import pandas as pd
from walk_states import detect_walk_states

def fixture(closes,lower,upper):
    idx=pd.date_range('2025-01-01',periods=len(closes),freq='1h',tz='UTC')
    x=pd.DataFrame({'close':closes,'bb_lower':lower,'bb_upper':upper},index=idx)
    x.attrs['resolution']='1h'
    return x

def test_lower_confirm_no_backdating():
    x=fixture([9]*8,np.arange(10,2,-1,dtype=float),np.arange(20,12,-1,dtype=float))
    s,e=detect_walk_states(x,near=.3,confirm_bars=2,slope_bars=1)
    assert s.iloc[0]['state']=='NEUTRAL'
    assert 'LOWER_WALK' in set(s.state)
    first=s.index[s.state=='LOWER_WALK'][0]
    assert first==e.iloc[0]['time']

def test_no_walk_if_flat_band():
    x=fixture([10]*10,[10]*10,[20]*10)
    s,e=detect_walk_states(x,confirm_bars=2,slope_bars=1)
    assert (s.state=='NEUTRAL').all()

def test_exit_requires_two_closes():
    x=fixture([9,8,7,6,7,12,12], [10,9,8,7,6,5,4], [20,19,18,17,16,15,14])
    s,e=detect_walk_states(x,near=.3,confirm_bars=2,exit_bars=2,slope_bars=1)
    assert 'LOWER_WALK' in set(s.state)
    assert (e.reason=='DISTANCE_EXIT').any()

def test_gap_resets():
    x=fixture([9,8,7,6,5,4], [10,9,8,7,6,5], [20,19,18,17,16,15]);x.loc[x.index[4],'close']=np.nan
    s,e=detect_walk_states(x,near=.3,confirm_bars=2,slope_bars=1)
    assert s.iloc[4].state=='NEUTRAL'
