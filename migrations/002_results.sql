CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    manifest TEXT NOT NULL
);
CREATE TABLE results (
    run_id TEXT NOT NULL REFERENCES runs(id),
    event_id TEXT NOT NULL REFERENCES events(id),
    aligned_session TEXT,
    status TEXT NOT NULL CHECK(status IN ('included','excluded')),
    reason TEXT,
    car REAL,
    payload TEXT NOT NULL,
    PRIMARY KEY(run_id,event_id),
    CHECK((status='included' AND reason IS NULL AND car IS NOT NULL)
       OR (status='excluded' AND reason IS NOT NULL AND car IS NULL))
);
CREATE INDEX results_session ON results(aligned_session);
CREATE TABLE points (
    run_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    offset INTEGER NOT NULL,
    session TEXT NOT NULL,
    ar REAL NOT NULL,
    car REAL NOT NULL,
    lower_bound REAL NOT NULL,
    upper_bound REAL NOT NULL,
    PRIMARY KEY(run_id,event_id,offset),
    FOREIGN KEY(run_id,event_id) REFERENCES results(run_id,event_id)
);
