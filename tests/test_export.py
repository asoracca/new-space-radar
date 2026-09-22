import json
from src.export import export


def test_export_is_deterministic_and_fully_accounted(tmp_path):
    a=export(tmp_path/'a',tmp_path/'evidence.db')
    b=export(tmp_path/'b',tmp_path/'evidence.db')
    assert a==b
    assert len(a['events'])==20
    assert a['counts']['historical']['included']==0
    assert a['counts']['synthetic']['included']==2
    assert a['counts']['historical']['excluded_reasons']=={
        'announcement_unverified':17,'missing_stock_or_benchmark':1}
    assert json.loads((tmp_path/'a/results.json').read_text())==a


def test_saved_evidence_matches_scientific_results(tmp_path):
    from src.events import ROOT
    import pytest
    saved=json.loads((ROOT/'evidence/results.json').read_text())
    current=export(tmp_path/'current',tmp_path/'current.db')
    assert current['counts']==saved['counts']
    assert current['selection_bias']==saved['selection_bias']
    for now,old in zip(current['events'],saved['events'],strict=True):
        for key in ['id','status','reason','aligned_session','n_estimation','n_event']:
            assert now[key]==old[key]
        if now['car'] is not None:
            assert now['car']==pytest.approx(old['car'],abs=1e-12)
