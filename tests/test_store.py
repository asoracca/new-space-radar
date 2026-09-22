from dataclasses import replace
import sqlite3
import pytest
from src.demo import synthetic_case
from src.events import study
from src.store import connect,save_run,query_events,reconcile


def test_migrations_reproducibility_and_reconciliation(tmp_path):
    path=tmp_path/'study.db'; db=connect(path)
    s,r,e=synthetic_case(); rows=study(e,r,s)
    key=save_run(db,e,rows,{'fixture':'test'})
    assert save_run(db,e,rows,{'fixture':'test'})==key
    assert len(query_events(db,key,symbol='SHOCK',event_type='validation',start='2024-02-23',end='2024-02-23'))==1
    assert query_events(db,key,symbol="' OR 1=1 --")==[]
    assert len(reconcile(db,key,rows))==1
    db.close(); db=connect(path)
    assert db.execute('SELECT COUNT(*) FROM migrations').fetchone()[0]==2
    assert db.execute('SELECT COUNT(*) FROM runs').fetchone()[0]==1
    altered=[{**rows[0],'car':9},rows[1]]
    with pytest.raises(ValueError): save_run(db,e,altered,{'fixture':'test'})
    with pytest.raises(ValueError): save_run(db,[replace(e[0],title='changed'),e[1]],rows,{'fixture':'changed'})
    assert db.execute('SELECT COUNT(*) FROM runs').fetchone()[0]==1
    with pytest.raises(sqlite3.IntegrityError):
        db.execute('INSERT INTO events SELECT ?,symbol,event_type,timestamp,original_date,input_kind,announcement_status,source_url,payload FROM events LIMIT 1',('duplicate',))
    db.rollback()
