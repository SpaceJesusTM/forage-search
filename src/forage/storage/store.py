"""Repository over the SQLite schema. All persistence goes through this class.

Rows are returned as plain dicts with JSON columns decoded. Callers that need
several writes to land together wrap them in `with store.tx():`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from forage.storage.db import Database, loads
from forage.types import ChangeType, ClaimStatus, QuestionStatus, RunStatus, SourceStatus
from forage.util import iso, normalize_text, parse_iso, stable_id, utcnow

_JSON_COLUMNS = {
    "watch": ("policy_config", "budget"),
    "watch_revision": ("spec",),
    "watch_event": ("detail",),
    "session_run": ("budget", "metrics"),
    "search_attempt": ("results",),
    "source": ("flags",),
    "finding_change": ("detail",),
    "open_question": ("resolved_by",),
    "run_event": ("data",),
    "checkpoint_request": ("choices", "evidence"),
    "checkpoint_response": ("detail",),
    "checkpoint_decision": ("signals",),
    "outbox": ("payload",),
}


def _decode(table: str, row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    for col in _JSON_COLUMNS.get(table, ()):
        if col in row:
            row[col] = loads(row[col], default=None)
    return row


class Store(Database):
    def __init__(self, path: str | Path, clock=utcnow):
        super().__init__(path)
        self.clock = clock

    def now(self) -> str:
        return iso(self.clock())

    def _rows(self, table: str, sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
        return [_decode(table, row) for row in self.all(sql, params)]  # type: ignore[misc]

    def _row(self, table: str, sql: str, params: tuple | dict = ()) -> dict[str, Any] | None:
        return _decode(table, self.one(sql, params))

    # -- watches ----------------------------------------------------------

    def get_watch(self, watch_id: str) -> dict[str, Any] | None:
        return self._row("watch", "SELECT * FROM watch WHERE id = ?", (watch_id,))

    def find_watch(self, ref: str) -> dict[str, Any] | None:
        """Resolve an exact id or a unique id prefix."""
        watch = self.get_watch(ref)
        if watch:
            return watch
        matches = self._rows("watch", "SELECT * FROM watch WHERE id LIKE ? ORDER BY id", (ref.replace("%", "") + "%",))
        return matches[0] if len(matches) == 1 else None

    def list_watches(self, statuses: list[str] | None = None) -> list[dict[str, Any]]:
        if statuses:
            marks = ",".join("?" * len(statuses))
            return self._rows(
                "watch", f"SELECT * FROM watch WHERE status IN ({marks}) ORDER BY queue_position, id", tuple(statuses)
            )
        return self._rows("watch", "SELECT * FROM watch ORDER BY queue_position, id")

    def next_queue_position(self) -> int:
        return int(self.scalar("SELECT COALESCE(MAX(queue_position), 0) + 1 FROM watch"))

    def get_revision(self, revision_id: str | None) -> dict[str, Any] | None:
        if not revision_id:
            return None
        return self._row("watch_revision", "SELECT * FROM watch_revision WHERE id = ?", (revision_id,))

    def list_revisions(self, watch_id: str) -> list[dict[str, Any]]:
        return self._rows(
            "watch_revision", "SELECT * FROM watch_revision WHERE watch_id = ? ORDER BY revision_no", (watch_id,)
        )

    def add_watch_event(
        self, watch_id: str, kind: str, actor: str, from_status: str | None = None, to_status: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.insert(
            "watch_event",
            {"watch_id": watch_id, "at": self.now(), "kind": kind, "from_status": from_status,
             "to_status": to_status, "actor": actor, "detail": detail or {}},
        )

    def list_watch_events(self, watch_id: str) -> list[dict[str, Any]]:
        return self._rows("watch_event", "SELECT * FROM watch_event WHERE watch_id = ? ORDER BY id", (watch_id,))

    # -- runs -------------------------------------------------------------

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self._row("session_run", "SELECT * FROM session_run WHERE id = ?", (run_id,))

    def list_runs(self, watch_id: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM session_run"
        params: tuple = ()
        if watch_id:
            sql += " WHERE watch_id = ?"
            params = (watch_id,)
        sql += " ORDER BY started_at, seq"
        rows = self._rows("session_run", sql, params)
        return rows[-limit:] if limit else rows

    def running_run(self) -> dict[str, Any] | None:
        return self._row("session_run", "SELECT * FROM session_run WHERE status = ?", (RunStatus.RUNNING,))

    def next_seq(self, watch_id: str) -> int:
        return int(self.scalar("SELECT COALESCE(MAX(seq), 0) + 1 FROM session_run WHERE watch_id = ?", (watch_id,)))

    def heartbeat(self, run_id: str) -> None:
        self.update("session_run", {"id": run_id}, {"heartbeat_at": self.now()})

    def recover_stale_runs(self, stale_after_seconds: int, now: datetime | None = None) -> list[str]:
        """Mark RUNNING runs with an old heartbeat as INTERRUPTED so a new run can be admitted."""
        now = now or self.clock()
        recovered = []
        for run in self._rows("session_run", "SELECT * FROM session_run WHERE status = ?", (RunStatus.RUNNING,)):
            age = (now - parse_iso(run["heartbeat_at"])).total_seconds()
            if age >= stale_after_seconds:
                self.update(
                    "session_run", {"id": run["id"]},
                    {"status": RunStatus.INTERRUPTED, "finished_at": iso(now),
                     "stop_reason": "interrupted", "error": f"no heartbeat for {int(age)}s; process presumed dead"},
                )
                self.add_event(run["id"], run["watch_id"], "run_interrupted", {"heartbeat_age_seconds": int(age)})
                recovered.append(run["id"])
        return recovered

    def completed_runs(self, watch_id: str) -> list[dict[str, Any]]:
        return self._rows(
            "session_run", "SELECT * FROM session_run WHERE watch_id = ? AND status = ? ORDER BY seq",
            (watch_id, RunStatus.COMPLETED),
        )

    # -- logs -------------------------------------------------------------

    def add_event(self, run_id: str | None, watch_id: str | None, kind: str, data: dict[str, Any] | None = None) -> None:
        self.insert("run_event", {"run_id": run_id, "watch_id": watch_id, "at": self.now(), "kind": kind, "data": data or {}})

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        return self._rows("run_event", "SELECT * FROM run_event WHERE run_id = ? ORDER BY id", (run_id,))

    def add_llm_call(self, **values: Any) -> None:
        values.setdefault("at", self.now())
        self.insert("llm_call", values)

    def list_llm_calls(self, run_id: str) -> list[dict[str, Any]]:
        return self.all("SELECT * FROM llm_call WHERE run_id = ? ORDER BY id", (run_id,))

    # -- search -----------------------------------------------------------

    def add_search_attempt(self, **values: Any) -> int:
        values.setdefault("at", self.now())
        values["query_norm"] = normalize_text(values["query"])
        return int(self.insert("search_attempt", values).lastrowid or 0)

    def list_search_attempts(self, watch_id: str, run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id:
            return self._rows(
                "search_attempt", "SELECT * FROM search_attempt WHERE watch_id = ? AND run_id = ? ORDER BY id",
                (watch_id, run_id),
            )
        return self._rows("search_attempt", "SELECT * FROM search_attempt WHERE watch_id = ? ORDER BY id", (watch_id,))

    # -- sources ----------------------------------------------------------

    @staticmethod
    def source_id(watch_id: str, canonical_url: str) -> str:
        return stable_id("s", watch_id, canonical_url)

    def get_source(self, source_id: str) -> dict[str, Any] | None:
        return self._row("source", "SELECT * FROM source WHERE id = ?", (source_id,))

    def get_source_by_url(self, watch_id: str, canonical_url: str) -> dict[str, Any] | None:
        return self._row(
            "source", "SELECT * FROM source WHERE watch_id = ? AND canonical_url = ?", (watch_id, canonical_url)
        )

    def upsert_discovered_source(
        self, watch_id: str, canonical_url: str, url: str, domain: str, title: str | None, snippet: str | None,
        discovered_via: str | None, run_id: str | None,
    ) -> tuple[dict[str, Any], bool]:
        """Return (source, created). Existing sources are left untouched."""
        existing = self.get_source_by_url(watch_id, canonical_url)
        if existing:
            return existing, False
        sid = self.source_id(watch_id, canonical_url)
        self.insert(
            "source",
            {"id": sid, "watch_id": watch_id, "canonical_url": canonical_url, "url": url, "domain": domain,
             "title": title, "snippet": snippet, "status": SourceStatus.DISCOVERED, "discovered_via": discovered_via,
             "discovered_run_id": run_id, "first_seen_at": self.now()},
        )
        return self.get_source(sid), True  # type: ignore[return-value]

    def update_source(self, source_id: str, **values: Any) -> None:
        self.update("source", {"id": source_id}, values)

    def list_sources(self, watch_id: str, statuses: list[str] | None = None) -> list[dict[str, Any]]:
        if statuses:
            marks = ",".join("?" * len(statuses))
            return self._rows(
                "source", f"SELECT * FROM source WHERE watch_id = ? AND status IN ({marks}) ORDER BY first_seen_at, rowid",
                (watch_id, *statuses),
            )
        return self._rows("source", "SELECT * FROM source WHERE watch_id = ? ORDER BY first_seen_at, rowid", (watch_id,))

    def add_snapshot(self, source_id: str, run_id: str, content_hash: str, text: str) -> tuple[int, bool]:
        """Store a content version. Return (snapshot_id, is_new_version)."""
        existing = self.one(
            "SELECT id FROM source_snapshot WHERE source_id = ? AND content_hash = ?", (source_id, content_hash)
        )
        if existing:
            return int(existing["id"]), False
        cur = self.insert(
            "source_snapshot",
            {"source_id": source_id, "run_id": run_id, "fetched_at": self.now(), "content_hash": content_hash, "text": text},
        )
        return int(cur.lastrowid or 0), True

    def get_snapshot(self, snapshot_id: int) -> dict[str, Any] | None:
        return self.one("SELECT * FROM source_snapshot WHERE id = ?", (snapshot_id,))

    def latest_snapshot(self, source_id: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM source_snapshot WHERE source_id = ? ORDER BY id DESC LIMIT 1", (source_id,))

    # -- findings and claims ----------------------------------------------

    def get_finding(self, finding_id: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM finding WHERE id = ?", (finding_id,))

    def list_findings(self, watch_id: str) -> list[dict[str, Any]]:
        return self.all("SELECT * FROM finding WHERE watch_id = ? ORDER BY first_seen_at, rowid", (watch_id,))

    def create_finding(self, watch_id: str, run_id: str, title: str, subject: str, relevance: int | None,
                       uncertainty: str | None, seed: str) -> str:
        fid = stable_id("f", watch_id, seed)
        now = self.now()
        self.insert(
            "finding",
            {"id": fid, "watch_id": watch_id, "title": title, "subject": subject, "relevance": relevance,
             "uncertainty": uncertainty, "first_seen_run_id": run_id, "first_seen_at": now,
             "last_changed_run_id": run_id, "last_changed_at": now, "last_confirmed_at": now},
            or_ignore=True,
        )
        return fid

    def update_finding(self, finding_id: str, **values: Any) -> None:
        self.update("finding", {"id": finding_id}, values)

    @staticmethod
    def claim_id(watch_id: str, source_id: str, text: str) -> str:
        return stable_id("c", watch_id, source_id, normalize_text(text), length=14)

    def get_claim(self, claim_id: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM claim WHERE id = ?", (claim_id,))

    def add_claim(self, **values: Any) -> tuple[str, bool]:
        """Insert a claim; return (claim_id, created). A repeat of the same text from the same source is a no-op."""
        cid = self.claim_id(values["watch_id"], values["source_id"], values["text"])
        now = self.now()
        values = {"id": cid, "text_norm": normalize_text(values["text"]), "checked_at": now, "last_confirmed_at": now} | values
        values["id"] = cid
        created = self.insert("claim", values, or_ignore=True).rowcount > 0
        return cid, created

    def update_claim(self, claim_id: str, **values: Any) -> None:
        self.update("claim", {"id": claim_id}, values)

    def list_claims(self, *, watch_id: str | None = None, finding_id: str | None = None, source_id: str | None = None,
                    run_id: str | None = None, status: str | None = ClaimStatus.ACCEPTED) -> list[dict[str, Any]]:
        clauses, params = [], []
        for col, val in (("watch_id", watch_id), ("finding_id", finding_id), ("source_id", source_id),
                         ("run_id", run_id), ("status", status)):
            if val is not None:
                clauses.append(f"{col} = ?")
                params.append(str(val))
        where = " AND ".join(clauses) or "1 = 1"
        return self.all(f"SELECT * FROM claim WHERE {where} ORDER BY checked_at, rowid", tuple(params))

    def record_change(self, run_id: str, watch_id: str, finding_id: str, change: ChangeType,
                      detail: dict[str, Any] | None = None) -> None:
        """Keep the strongest change per (run, finding) and merge claim-id lists in the detail."""
        existing = self._row(
            "finding_change", "SELECT * FROM finding_change WHERE run_id = ? AND finding_id = ?", (run_id, finding_id)
        )
        detail = detail or {}
        if existing is None:
            self.insert(
                "finding_change",
                {"run_id": run_id, "watch_id": watch_id, "finding_id": finding_id, "change_type": change,
                 "detail": detail, "at": self.now()},
            )
            return
        merged = dict(existing["detail"] or {})
        for key, value in detail.items():
            if isinstance(value, list):
                merged[key] = list(dict.fromkeys([*merged.get(key, []), *value]))
            else:
                merged[key] = value
        best = change if change.rank > ChangeType(existing["change_type"]).rank else ChangeType(existing["change_type"])
        self.update("finding_change", {"id": existing["id"]}, {"change_type": best, "detail": merged})

    def list_changes(self, run_id: str) -> list[dict[str, Any]]:
        return self._rows("finding_change", "SELECT * FROM finding_change WHERE run_id = ? ORDER BY id", (run_id,))

    # -- open questions and leads -----------------------------------------

    def add_question(self, watch_id: str, text: str, origin: str, run_id: str | None = None) -> tuple[str, bool]:
        norm = normalize_text(text)
        qid = stable_id("q", watch_id, norm, length=10)
        created = self.insert(
            "open_question",
            {"id": qid, "watch_id": watch_id, "text": text, "text_norm": norm, "origin": origin,
             "status": QuestionStatus.OPEN, "created_run_id": run_id, "created_at": self.now()},
            or_ignore=True,
        ).rowcount > 0
        return qid, created

    def list_questions(self, watch_id: str, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            return self._rows(
                "open_question", "SELECT * FROM open_question WHERE watch_id = ? AND status = ? ORDER BY created_at, rowid",
                (watch_id, status),
            )
        return self._rows(
            "open_question", "SELECT * FROM open_question WHERE watch_id = ? ORDER BY created_at, rowid", (watch_id,)
        )

    def update_question(self, question_id: str, **values: Any) -> None:
        self.update("open_question", {"id": question_id}, values)

    def add_lead(self, watch_id: str, text: str, rationale: str | None, run_id: str | None) -> bool:
        return self.insert(
            "lead",
            {"watch_id": watch_id, "text": text, "text_norm": normalize_text(text), "rationale": rationale,
             "status": "OPEN", "created_run_id": run_id, "created_at": self.now()},
            or_ignore=True,
        ).rowcount > 0

    def list_leads(self, watch_id: str, status: str | None = "OPEN") -> list[dict[str, Any]]:
        if status:
            return self.all("SELECT * FROM lead WHERE watch_id = ? AND status = ? ORDER BY id", (watch_id, status))
        return self.all("SELECT * FROM lead WHERE watch_id = ? ORDER BY id", (watch_id,))

    def update_lead(self, lead_id: int, **values: Any) -> None:
        self.update("lead", {"id": lead_id}, values)

    # -- reports ----------------------------------------------------------

    def save_report(self, **values: Any) -> None:
        values.setdefault("created_at", self.now())
        self.execute("DELETE FROM report WHERE run_id = ?", (values["run_id"],))
        self.insert("report", values)

    def get_report(self, run_id: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM report WHERE run_id = ?", (run_id,))

    def latest_report(self, watch_id: str) -> dict[str, Any] | None:
        return self.one(
            "SELECT r.* FROM report r JOIN session_run s ON s.id = r.run_id WHERE r.watch_id = ? ORDER BY s.seq DESC LIMIT 1",
            (watch_id,),
        )
