-- Control plane: checkpoints, queue rotation, scheduler lease, delivery outbox.

-- Every policy evaluation, including CONTINUE, so declined checkpoints can be analysed.
CREATE TABLE checkpoint_decision (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   TEXT,
    watch_id TEXT NOT NULL,
    policy   TEXT NOT NULL,
    decision TEXT NOT NULL,
    signal   TEXT,
    reason   TEXT NOT NULL,
    signals  TEXT NOT NULL DEFAULT '{}',
    at       TEXT NOT NULL
);
CREATE INDEX checkpoint_decision_watch ON checkpoint_decision (watch_id, id);

CREATE TABLE checkpoint_request (
    id            TEXT PRIMARY KEY,
    watch_id      TEXT NOT NULL REFERENCES watch(id),
    run_id        TEXT,
    policy        TEXT NOT NULL,
    signal        TEXT NOT NULL,
    status        TEXT NOT NULL,
    what_happened TEXT NOT NULL,
    why_human     TEXT NOT NULL,
    question      TEXT NOT NULL,
    choices       TEXT NOT NULL,
    safe_default  TEXT NOT NULL,
    urgency       TEXT NOT NULL,
    evidence      TEXT NOT NULL DEFAULT '[]',
    created_at    TEXT NOT NULL,
    answered_at   TEXT
);
-- A watch can wait on at most one question at a time.
CREATE UNIQUE INDEX checkpoint_request_one_pending ON checkpoint_request (watch_id) WHERE status = 'PENDING';

CREATE TABLE checkpoint_response (
    id                     TEXT PRIMARY KEY,
    request_id             TEXT NOT NULL REFERENCES checkpoint_request(id),
    watch_id               TEXT NOT NULL,
    choice_id              TEXT,
    note                   TEXT,
    actor                  TEXT NOT NULL,
    scope_change_confirmed INTEGER NOT NULL DEFAULT 0,
    resulting_revision_id  TEXT,
    resulting_status       TEXT,
    detail                 TEXT NOT NULL DEFAULT '{}',
    at                     TEXT NOT NULL
);

-- Round-robin cursor: the queue_position of the watch served most recently.
CREATE TABLE queue_state (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    cursor_position INTEGER NOT NULL DEFAULT 0,
    cursor_watch_id TEXT,
    admitted_total  INTEGER NOT NULL DEFAULT 0,
    updated_at      TEXT NOT NULL
);

CREATE TABLE watch_queue (
    watch_id             TEXT PRIMARY KEY REFERENCES watch(id),
    turns                INTEGER NOT NULL DEFAULT 0,
    last_admitted_at     TEXT,
    last_run_id          TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    cooldown_until       TEXT
);

-- Every scheduling decision, for auditing fairness and starvation.
CREATE TABLE queue_admission (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    at       TEXT NOT NULL,
    watch_id TEXT,
    run_id   TEXT,
    decision TEXT NOT NULL,
    reason   TEXT,
    trigger  TEXT
);

-- Single scheduler authority: only the lease holder may admit queued runs.
CREATE TABLE scheduler_lease (
    id           INTEGER PRIMARY KEY CHECK (id = 1),
    owner        TEXT NOT NULL,
    acquired_at  TEXT NOT NULL,
    heartbeat_at TEXT NOT NULL,
    expires_at   TEXT NOT NULL
);

-- Reports and review requests waiting for an adapter to deliver them. Unused in standalone mode.
CREATE TABLE outbox (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    watch_id     TEXT NOT NULL,
    kind         TEXT NOT NULL,
    payload      TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    delivered_at TEXT,
    delivery_ref TEXT,
    UNIQUE (run_id, kind)
);
