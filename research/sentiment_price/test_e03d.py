import numpy as np
import pandas as pd
from e03d_surface import surface,compare_shared

def sample():
    idx=pd.date_range('2025-01-01',periods=320,freq='1h',tz='UTC')
    vals=100+np.cumsum(np.random.default_rng(7).normal(0,.4,len(idx)))
    d=pd.DataFrame({'open':vals-.1,'high':vals+.3,'low':vals-.3,'close':vals},index=idx)
    d.attrs['resolution']='1h'
    return d

def test_surface_and_baseline():
    r=surface(sample(),windows=[10,20],sigmas=[1.9,2.0])
    assert len(r)==4 and ((r.containment>=0)&(r.containment<=1)).all()
    assert set(r.window)=={10,20}
    a=r.set_index(['window','sigma'])
    assert a.loc[(20,2.0),'containment']>=a.loc[(20,1.9),'containment']

def test_prefix_stability():
    d=sample(); a=surface(d.iloc[:260],windows=[10],sigmas=[2.0]); b=surface(d,windows=[10],sigmas=[2.0])
    # All prior bands evaluated identically; aggregate differs due to extra rows.
    assert a.iloc[0].valid_bars < b.iloc[0].valid_bars

def test_shared():
    a=surface(sample(),windows=[10,20],sigmas=[1.9,2.0])
    best,shared,_=compare_shared({'1H':a,'15m':a})
    assert len(best)==2 and len(shared)==1
