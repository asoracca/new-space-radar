CREATE TABLE sources (
    url TEXT PRIMARY KEY,
    note TEXT NOT NULL
);
CREATE TABLE events (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp TEXT,
    original_date TEXT NOT NULL,
    input_kind TEXT NOT NULL CHECK(input_kind IN ('historical','synthetic','live')),
    announcement_status TEXT NOT NULL,
    source_url TEXT REFERENCES sources(url),
    payload TEXT NOT NULL,
    UNIQUE(symbol, timestamp, event_type, input_kind)
);
CREATE INDEX events_filters ON events(symbol, event_type, original_date);
