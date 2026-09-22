from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import exchange_calendars as xc
from src.demo import synthetic_case, selection_experiment
from src.events import Event, StudyConfig, align_event, load_catalog, load_schedule, study

@pytest.fixture
def case():
    return synthetic_case()

@pytest.mark.parametrize('timestamp,expected', [
    ('2024-02-22T15:59:00-05:00','2024-02-22'),
    ('2024-02-22T16:00:00-05:00','2024-02-23'),
    ('2024-02-22T18:23:00-05:00','2024-02-23'),
    ('2024-03-29T10:00:00-04:00','2024-04-01'),
    ('2024-11-29T13:00:00-05:00','2024-12-02'),
    ('2024-03-10T18:00:00-04:00','2024-03-11'),
])
def test_alignment(case,timestamp,expected):
    s,r,ev = case
    assert str(align_event(replace(ev[0],timestamp=timestamp),s).date()) == expected


def test_frozen_calendar_matches_dependency():
    s=load_schedule()
    cal=xc.get_calendar('XNYS',start='2022-01-01',end='2025-12-31')
    pd.testing.assert_frame_equal(s,cal.schedule[['open','close']].rename_axis('session'),check_freq=False)


def test_planted_shock_and_control(case):
    s,r,ev=case
    shock,control=study(ev,r,s)
    assert shock['car'] == pytest.approx(.05,abs=1e-12)
    assert shock['ar_day'] == pytest.approx(.05,abs=1e-12)
    assert control['car'] == pytest.approx(0,abs=1e-12)
    assert shock['beta'] == pytest.approx(1.3)
    assert shock['n_estimation']==120 and shock['n_event']==8
    assert shock['estimation_end'] < shock['event_start']
    changed=r.copy(); changed.loc['2024-02-21':'2024-03-01','SHOCK'] += 100
    after=study(ev,changed,s)[0]
    assert after['alpha']==shock['alpha'] and after['beta']==shock['beta']
    assert 'not a trading-edge' in control['limitations']

@pytest.mark.parametrize('where,reason',[('2024-02-23','missing_event_observation'),('2024-01-02','missing_estimation_observation')])
@pytest.mark.parametrize('column',['SPY','SHOCK'])
def test_missing_observation(case,where,reason,column):
    s,r,ev=case; r.loc[where,column]=np.nan
    assert study(ev,r,s)[0]['reason']==reason


def test_missing_date_does_not_shift_session(case):
    s,r,ev=case; r=r.drop(pd.Timestamp('2024-02-23'))
    out=study(ev,r,s)[0]
    assert out['aligned_session']=='2024-02-23'
    assert out['reason']=='missing_event_observation'


def test_duplicate_overlap_and_insufficient_history(case):
    s,r,ev=case; e=ev[0]
    out=study([e,replace(e,id='dup')],r,s)
    assert out[1]['reason']=='duplicate_announcement' and out[0]['reason'] is None
    overlap=replace(e,id='other',timestamp='2024-02-23T18:23:00-05:00')
    assert all(x['reason']=='overlapping_event_window' for x in study([e,overlap],r,s))
    early=replace(e,timestamp='2022-01-04T10:00:00-05:00')
    assert study([early],r,s)[0]['reason']=='insufficient_history'
    later=replace(e,id='later',timestamp='2024-04-23T10:00:00-04:00')
    assert study([e,later],r,s)[1]['reason']=='contaminated_estimation_window'


def test_schema_and_unverified_sources(case):
    s,r,ev=case
    with pytest.raises(ValueError): replace(ev[0],timestamp='2024-02-22T18:00:00')
    with pytest.raises(ValueError): replace(ev[0],timestamp='2024-02-22T18:00:00-04:00')
    with pytest.raises(ValueError): study([ev[0],ev[0]],r,s)
    out=study(load_catalog(),r,s)
    assert sum(x['reason']=='announcement_unverified' for x in out)==17
    assert next(x for x in out if x['id']=='lunr-001')['aligned_session']=='2024-02-23'


def test_selection_bias_under_null():
    result=selection_experiment()
    assert abs(result['all_dates_mean'])<.001
    assert result['hindsight_top10_mean']> .02
    assert result['hindsight_top10_mean']>result['precommitted_mean']


def test_model_interval_formula(case):
    s,r,ev=case
    r['SHOCK'] += np.random.default_rng(6).normal(0,.002,len(r))
    out=study(ev,r,s)[0]
    assert out['car_se']>0
    for p in out['path']:
        assert p['lower']<p['car']<p['upper']
