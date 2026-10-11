-- Forage core schema: watches, runs, evidence, research memory, reports, logs.
-- Timestamps are UTC ISO-8601 strings. JSON columns hold serialised objects.

CREATE TABLE watch (
    id                  TEXT PRIMARY KEY,
    title               TEXT NOT NULL,
    status              TEXT NOT NULL,
    previous_status     TEXT,
    active_revision_id  TEXT,
    pending_revision_id TEXT,
    checkpoint_policy   TEXT NOT NULL DEFAULT 'continuous',
    policy_config       TEXT NOT NULL DEFAULT '{}',
    budget              TEXT NOT NULL DEFAULT '{}',
    queue_position      INTEGER NOT NULL,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

-- Every objective/constraint version, with who proposed and who decided it.
CREATE TABLE watch_revision (
    id          TEXT PRIMARY KEY,
    watch_id    TEXT NOT NULL REFERENCES watch(id),
    revision_no INTEGER NOT NULL,
    objective   TEXT NOT NULL,
    spec        TEXT NOT NULL,
    origin      TEXT NOT NULL,
    state       TEXT NOT NULL,
    note        TEXT,
    created_at  TEXT NOT NULL,
    decided_at  TEXT,
    decided_by  TEXT,
    UNIQUE (watch_id, revision_no)
);

CREATE TABLE watch_event (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    watch_id    TEXT NOT NULL REFERENCES watch(id),
    at          TEXT NOT NULL,
    kind        TEXT NOT NULL,
    from_status TEXT,
    to_status   TEXT,
    actor       TEXT NOT NULL,
    detail      TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX watch_event_watch ON watch_event (watch_id, id);

CREATE TABLE session_run (
    id                    TEXT PRIMARY KEY,
    watch_id              TEXT NOT NULL REFERENCES watch(id),
    revision_id           TEXT NOT NULL,
    seq                   INTEGER NOT NULL,
    status                TEXT NOT NULL,
    trigger               TEXT NOT NULL,
    policy                TEXT NOT NULL,
    model                 TEXT,
    search_provider       TEXT,
    budget                TEXT NOT NULL,
    metrics               TEXT NOT NULL DEFAULT '{}',
    has_meaningful_update INTEGER,
    stop_reason           TEXT,
    error                 TEXT,
    cancel_requested      INTEGER NOT NULL DEFAULT 0,
    started_at            TEXT NOT NULL,
    heartbeat_at          TEXT NOT NULL,
    finished_at           TEXT,
    UNIQUE (watch_id, seq)
);
-- One research session at a time across the whole store (shared local compute).
CREATE UNIQUE INDEX session_run_single_running ON session_run (status) WHERE status = 'RUNNING';

CREATE TABLE search_attempt (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id             TEXT NOT NULL REFERENCES session_run(id),
    watch_id           TEXT NOT NULL,
    query              TEXT NOT NULL,
    query_norm         TEXT NOT NULL,
    rationale          TEXT,
    target_question_id TEXT,
    provider           TEXT NOT NULL,
    status             TEXT NOT NULL,
    error              TEXT,
    result_count       INTEGER NOT NULL DEFAULT 0,
    new_url_count      INTEGER NOT NULL DEFAULT 0,
    known_url_count    INTEGER NOT NULL DEFAULT 0,
    results            TEXT NOT NULL DEFAULT '[]',
    duration_ms        INTEGER,
    at                 TEXT NOT NULL
);
CREATE INDEX search_attempt_watch ON search_attempt (watch_id, id);

CREATE TABLE source (
    id                TEXT PRIMARY KEY,
    watch_id          TEXT NOT NULL REFERENCES watch(id),
    canonical_url     TEXT NOT NULL,
    url               TEXT NOT NULL,
    final_url         TEXT,
    domain            TEXT NOT NULL,
    title             TEXT,
    snippet           TEXT,
    status            TEXT NOT NULL,
    relevance         INTEGER,
    triage_note       TEXT,
    discovered_via    TEXT,
    discovered_run_id TEXT,
    first_seen_at     TEXT NOT NULL,
    last_fetched_at   TEXT,
    last_changed_at   TEXT,
    fetch_count       INTEGER NOT NULL DEFAULT 0,
    http_status       INTEGER,
    content_type      TEXT,
    content_hash      TEXT,
    text_chars        INTEGER,
    published_at      TEXT,
    page_relevance    INTEGER,
    flags             TEXT NOT NULL DEFAULT '[]',
    last_error        TEXT,
    UNIQUE (watch_id, canonical_url)
);

-- Extracted text of each distinct content version: audit trail, quote verification, change detection.
CREATE TABLE source_snapshot (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id    TEXT NOT NULL REFERENCES source(id),
    run_id       TEXT NOT NULL,
    fetched_at   TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    text         TEXT NOT NULL,
    UNIQUE (source_id, content_hash)
);

CREATE TABLE finding (
    id                  TEXT PRIMARY KEY,
    watch_id            TEXT NOT NULL REFERENCES watch(id),
    title               TEXT NOT NULL,
    subject             TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'ACTIVE',
    relevance           INTEGER,
    uncertainty         TEXT,
    first_seen_run_id   TEXT NOT NULL,
    first_seen_at       TEXT NOT NULL,
    last_changed_run_id TEXT NOT NULL,
    last_changed_at     TEXT NOT NULL,
    last_confirmed_at   TEXT NOT NULL
);
CREATE INDEX finding_watch ON finding (watch_id);

-- Accepted claims belong to a finding. Rejected claims keep finding_id NULL and are never reported.
CREATE TABLE claim (
    id                TEXT PRIMARY KEY,
    watch_id          TEXT NOT NULL,
    finding_id        TEXT REFERENCES finding(id),
    source_id         TEXT NOT NULL REFERENCES source(id),
    snapshot_id       INTEGER REFERENCES source_snapshot(id),
    run_id            TEXT NOT NULL,
    text              TEXT NOT NULL,
    text_norm         TEXT NOT NULL,
    quote             TEXT NOT NULL,
    quote_offset      INTEGER,
    verification      TEXT NOT NULL,
    support           TEXT NOT NULL,
    status            TEXT NOT NULL,
    note              TEXT,
    checked_at        TEXT NOT NULL,
    last_confirmed_at TEXT NOT NULL
);
CREATE INDEX claim_finding ON claim (finding_id);
CREATE INDEX claim_watch ON claim (watch_id, status);

-- What each run did to each finding. Drives reports and before/after comparison.
CREATE TABLE finding_change (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    watch_id    TEXT NOT NULL,
    finding_id  TEXT NOT NULL,
    change_type TEXT NOT NULL,
    detail      TEXT NOT NULL DEFAULT '{}',
    at          TEXT NOT NULL,
    UNIQUE (run_id, finding_id)
);

CREATE TABLE open_question (
    id              TEXT PRIMARY KEY,
    watch_id        TEXT NOT NULL,
    text            TEXT NOT NULL,
    text_norm       TEXT NOT NULL,
    origin          TEXT NOT NULL,
    status          TEXT NOT NULL,
    attempts        INTEGER NOT NULL DEFAULT 0,
    created_run_id  TEXT,
    created_at      TEXT NOT NULL,
    resolved_run_id TEXT,
    resolved_at     TEXT,
    resolution      TEXT,
    resolved_by     TEXT NOT NULL DEFAULT '[]',
    UNIQUE (watch_id, text_norm)
);

-- Next research directions proposed at the end of a session.
CREATE TABLE lead (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    watch_id       TEXT NOT NULL,
    text           TEXT NOT NULL,
    text_norm      TEXT NOT NULL,
    rationale      TEXT,
    status         TEXT NOT NULL,
    created_run_id TEXT,
    created_at     TEXT NOT NULL,
    used_run_id    TEXT,
    UNIQUE (watch_id, text_norm)
);

CREATE TABLE report (
    run_id                TEXT PRIMARY KEY REFERENCES session_run(id),
    watch_id              TEXT NOT NULL,
    schema_version        TEXT NOT NULL,
    has_meaningful_update INTEGER NOT NULL,
    json                  TEXT NOT NULL,
    markdown              TEXT NOT NULL,
    json_path             TEXT,
    markdown_path         TEXT,
    created_at            TEXT NOT NULL
);

CREATE TABLE llm_call (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT,
    watch_id      TEXT,
    purpose       TEXT NOT NULL,
    model         TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    duration_ms   INTEGER NOT NULL DEFAULT 0,
    attempts      INTEGER NOT NULL DEFAULT 1,
    ok            INTEGER NOT NULL,
    error         TEXT,
    at            TEXT NOT NULL
);
CREATE INDEX llm_call_run ON llm_call (run_id);

CREATE TABLE run_event (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   TEXT,
    watch_id TEXT,
    at       TEXT NOT NULL,
    kind     TEXT NOT NULL,
    data     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX run_event_run ON run_event (run_id, id);
