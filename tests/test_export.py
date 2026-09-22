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
