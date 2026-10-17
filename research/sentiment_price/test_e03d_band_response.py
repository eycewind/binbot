import numpy as np
import pandas as pd
from e03d_band_response import impulse_response_details, summarize_impulse_response

def data():
    ix=pd.date_range('2025-01-01',periods=140,freq='1h',tz='UTC',name='close_time_utc')
    p=100+np.sin(np.arange(140)/8)*2+np.arange(140)*.05
    p[90:97]+=np.arange(7)*2
    b=pd.DataFrame({'open':p,'close':p,'high':p+.6,'low':p-.6},index=ix)
    r=pd.DataFrame([dict(ref_id=0,onset_index=90,confirmation_index=93,direction='UPPER_WALK')])
    return b,r

def test_independent_reference_population_and_sigma_changes():
    b,r=data()
    d=impulse_response_details(b,r,windows=(10,),sigmas=(1.5,2.0),follow_bars=6)
    assert len(d)==2 and d.ref_id.nunique()==1
    assert d.band_move_atr.notna().all()
    s=summarize_impulse_response(d)
    assert len(s)==2

def test_no_early_band_data_and_no_recontain_fabrication():
    b,r=data()
    r.onset_index=2
    r.confirmation_index=4
    d=impulse_response_details(b,r,windows=(30,),sigmas=(2.0,))
    assert d.empty

def test_future_extension_does_not_change_earlier_response():
    b,r=data()
    d1=impulse_response_details(b.iloc[:115],r,windows=(10,),sigmas=(2.,))
    b2=b.copy();b2.loc[b2.index[115]:,'close']+=10000
    b2.loc[b2.index[115]:,'high']+=10000;b2.loc[b2.index[115]:,'low']+=10000
    d2=impulse_response_details(b2,r,windows=(10,),sigmas=(2.,))
    pd.testing.assert_frame_equal(d1.reset_index(drop=True),d2.reset_index(drop=True))
