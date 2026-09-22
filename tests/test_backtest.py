import numpy as np
import pandas as pd
import pytest
from src.backtest import evaluate_signals, run_anomaly_backtest
from src.events import load_schedule


def test_fill_is_next_session_and_cannot_capture_signal_bar():
    s=load_schedule().loc['2024-02-20':'2024-03-08']
    p=pd.DataFrame({'RKLB':[100,200,300]+[330]*(len(s)-3),'SPY':100.},index=s.index)
    signals=[{'ticker':'RKLB','signal_date':s.index[1]}]
    tr=evaluate_signals(p,signals,hold_days=1,transaction_cost_bps=0,schedule=s).iloc[0]
    assert tr.entry_price==300
    assert tr['return']==pytest.approx(.1)
    assert tr.signal_timestamp < tr.entry_timestamp
    p.iloc[1,0]=10000
    assert evaluate_signals(p,signals,1,0,s).iloc[0]['return']==pytest.approx(.1)


def test_missing_fill_no_delay_and_overlapping_signals():
    s=load_schedule().loc['2024-02-20':'2024-03-08']
    p=pd.DataFrame({'RKLB':100.,'SPY':100.},index=s.index)
    signals=[{'ticker':'RKLB','signal_date':x} for x in s.index[:3]]
    assert len(evaluate_signals(p,signals,5,0,s))==1
    assert evaluate_signals(p.drop(s.index[1]),signals[:1],1,0,s).empty


def test_future_prices_do_not_change_past_signal():
    s=load_schedule().iloc[:240]
    rng=np.random.default_rng(23)
    p=pd.DataFrame(np.exp(np.cumsum(rng.normal(0,.02,(len(s),2)),axis=0))*100,
                   index=s.index,columns=['RKLB','SPY'])
    v=pd.DataFrame({'RKLB':rng.uniform(100,1000,len(s))},index=s.index)
    kwargs=dict(hold_days=1,vol_threshold=-100,price_threshold=-100,corr_threshold=2)
    a=run_anomaly_backtest(p,v,**kwargs)
    p.iloc[180:]*=30; v.iloc[180:]*=50
    b=run_anomaly_backtest(p,v,**kwargs)
    cutoff=s.index[175]
    pd.testing.assert_frame_equal(a[a.signal_date<cutoff],b[b.signal_date<cutoff])
