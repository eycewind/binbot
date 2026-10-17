import numpy as np
import pandas as pd
from e03c_analysis import price_impulse_references,candidate_alarms,match_alarms,detector_report

def sample(n=350):
    ix=pd.date_range('2025-01-01',periods=n,freq='15min',tz='UTC',name='close_time_utc')
    close=100+np.cumsum(np.sin(np.arange(n)/8)*.45+.12)
    bb=pd.DataFrame({'open':close-.1,'high':close+.6,'low':close-.6,'close':close,
                     'bb_upper':close+1.1,'bb_lower':close-1.1},index=ix)
    bb.attrs['resolution']='15min'
    return bb

def test_no_future_change():
    bb=sample(); cut=250
    ref=price_impulse_references(bb.iloc[:cut]); whole=price_impulse_references(bb)
    # References with confirmation through cut don't change when future bars are appended
    pd.testing.assert_frame_equal(ref.reset_index(drop=True),whole.loc[whole.confirmation_index<cut].reset_index(drop=True))
    for v in ['V1_one_bar','V2_near_no_slope','V3_prior_band_cross','V4_adaptive']:
        a=candidate_alarms(bb.iloc[:cut],v);b=candidate_alarms(bb,v)
        pd.testing.assert_frame_equal(a.reset_index(drop=True),b.loc[b.bar_index<cut].reset_index(drop=True))

def test_match_one_to_one():
    ref=pd.DataFrame([{'ref_id':0,'direction':'UPPER_WALK','onset_index':10,'onset':pd.Timestamp('2025-01-01',tz='UTC'),'magnitude_atr':2}])
    alarms=pd.DataFrame([{'bar_index':11,'direction':'UPPER_WALK'}, {'bar_index':12,'direction':'UPPER_WALK'}])
    m,f=match_alarms(alarms,ref)
    assert len(m)==1 and len(f)==1 and m.lag_bars.iloc[0]==1

def test_full():
    bb=sample();ref=price_impulse_references(bb)
    for v in ['V0_frozen','V1_one_bar','V2_near_no_slope','V3_prior_band_cross','V4_adaptive']:
        row,*_=detector_report(bb,ref,v)
        assert row['matched']<=row['references'] and row['matched']<=row['alarms']
