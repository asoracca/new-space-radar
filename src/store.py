"""Append-only local SQLite provenance and derived results, with numbered migrations."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3
import pandas as pd
from src.events import ROOT


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def connect(path):
    db=sqlite3.connect(path)
    db.execute('PRAGMA foreign_keys=ON')
    db.execute('CREATE TABLE IF NOT EXISTS migrations(version TEXT PRIMARY KEY, checksum TEXT NOT NULL)')
    for file in sorted((ROOT/'migrations').glob('*.sql')):
        checksum=hashlib.sha256(file.read_bytes()).hexdigest()
        row=db.execute('SELECT checksum FROM migrations WHERE version=?',(file.name,)).fetchone()
        if row:
            if row[0]!=checksum:
                raise ValueError('Applied migration changed: '+file.name)
            continue
        try:
            db.executescript('BEGIN;\n'+file.read_text())
            db.execute('INSERT INTO migrations VALUES (?,?)',(file.name,checksum))
            db.commit()
        except Exception:
            db.rollback()
            raise
    return db


def save_run(db,events,results,manifest):
    """Same inputs and software yield the same ID; reruns never overwrite evidence."""
    run_id=digest(manifest)
    if db.execute('SELECT 1 FROM runs WHERE id=?',(run_id,)).fetchone():
        existing=[json.loads(r[0]) for r in db.execute('SELECT payload FROM results WHERE run_id=? ORDER BY event_id',(run_id,))]
        if canonical(existing)!=canonical(sorted(results,key=lambda r:r['id'])):
            raise ValueError('Non-deterministic result for existing run')
        return run_id
    with db:
        db.execute('INSERT INTO runs VALUES (?,?)',(run_id,canonical(manifest)))
        for e in events:
            payload=canonical(asdict(e))
            previous=db.execute('SELECT payload FROM events WHERE id=?',(e.id,)).fetchone()
            if previous and previous[0]!=payload:
                raise ValueError('Immutable catalog record changed; use a new versioned ID: '+e.id)
            if not previous:
                if e.source_url:
                    db.execute('INSERT OR IGNORE INTO sources VALUES (?,?)',(e.source_url,e.source_note))
                db.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)',
                           (e.id,e.symbol,e.event_type,pd.Timestamp(e.timestamp).tz_convert("UTC").isoformat() if e.timestamp else None,e.original_date,e.input_kind,
                            e.announcement_status,e.source_url,payload))
        for r in results:
            db.execute('INSERT INTO results VALUES (?,?,?,?,?,?,?)',
                       (run_id,r['id'],r['aligned_session'],r['status'],r['reason'],r['car'],canonical(r)))
            db.executemany('INSERT INTO points VALUES (?,?,?,?,?,?,?,?)',
                          [(run_id,r['id'],p['offset'],p['session'],p['ar'],p['car'],p['lower'],p['upper']) for p in r['path']])
    return run_id


def query_events(db, run_id, symbol=None, event_type=None, start='0001-01-01',end='9999-12-31'):
    """Filters use aligned session, falling back to original date for unaligned exclusions."""
    return [json.loads(r[0]) for r in db.execute('''
        SELECT r.payload FROM results r JOIN events e ON e.id=r.event_id
        WHERE r.run_id=? AND (? IS NULL OR e.symbol=?)
        AND (? IS NULL OR e.event_type=?)
        AND COALESCE(r.aligned_session,e.original_date) BETWEEN ? AND ?
        ORDER BY e.id''',(run_id,symbol,symbol,event_type,event_type,start,end))]


def reconcile(db,run_id,rows):
    sql=db.execute('SELECT status,COUNT(*),AVG(car) FROM results WHERE run_id=? GROUP BY status',(run_id,)).fetchall()
    for status,n,mean in sql:
        subset=[r for r in rows if r['status']==status]
        assert len(subset)==n
        if status=='included':
            assert abs(sum(r['car'] for r in subset)/n-mean)<1e-12
    for r in rows:
        total=db.execute('SELECT SUM(ar),COUNT(*) FROM points WHERE run_id=? AND event_id=?',(run_id,r['id'])).fetchone()
        assert total[1]==len(r['path'])
        if r['car'] is not None:
            assert abs(total[0]-r['car'])<1e-12
    return sql
