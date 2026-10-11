"""Export stored logs and derived metrics for analysis outside Forage.

Everything comes from the database, so an export can be regenerated at any time and
two exports of the same database are identical. Page text and report bodies are left
out by default; the export is about behaviour and cost, not content.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from forage.storage.store import Store
from forage.types import ClaimStatus

TABLES: dict[str, str] = {
    "watches": "SELECT id, title, status, checkpoint_policy, policy_config, budget, queue_position, created_at FROM watch ORDER BY queue_position",
    "watch_revisions": "SELECT * FROM watch_revision ORDER BY watch_id, revision_no",
    "watch_events": "SELECT * FROM watch_event ORDER BY id",
    "runs": "SELECT * FROM session_run ORDER BY started_at, rowid",
    "search_attempts": "SELECT id, run_id, watch_id, query, rationale, target_question_id, provider, status, error, "
                       "result_count, new_url_count, known_url_count, duration_ms, at FROM search_attempt ORDER BY id",
    "llm_calls": "SELECT * FROM llm_call ORDER BY id",
    "sources": "SELECT id, watch_id, canonical_url, domain, status, relevance, triage_note, discovered_via, discovered_run_id, "
               "first_seen_at, last_fetched_at, fetch_count, http_status, text_chars, page_relevance, flags FROM source ORDER BY rowid",
    "findings": "SELECT * FROM finding ORDER BY rowid",
    "claims": "SELECT id, watch_id, finding_id, source_id, run_id, text, verification, support, status, note, checked_at "
              "FROM claim ORDER BY rowid",
    "finding_changes": "SELECT * FROM finding_change ORDER BY id",
    "questions": "SELECT * FROM open_question ORDER BY rowid",
    "checkpoint_decisions": "SELECT * FROM checkpoint_decision ORDER BY id",
    "checkpoint_requests": "SELECT * FROM checkpoint_request ORDER BY rowid",
    "checkpoint_responses": "SELECT * FROM checkpoint_response ORDER BY rowid",
    "queue_admissions": "SELECT * FROM queue_admission ORDER BY id",
    "run_events": "SELECT * FROM run_event ORDER BY id",
}

_JSON_FIELDS = {"policy_config", "budget", "metrics", "spec", "detail", "flags", "signals", "choices", "evidence",
                "resolved_by", "data"}


def _decode(row: dict[str, Any]) -> dict[str, Any]:
    for key in _JSON_FIELDS & row.keys():
        if isinstance(row[key], str):
            try:
                row[key] = json.loads(row[key])
            except ValueError:
                pass
    return row


def _ratio(numerator: float, denominator: float, per: float = 1.0) -> float | None:
    return round(per * numerator / denominator, 3) if denominator else None


def watch_summary(store: Store, watch_id: str) -> dict[str, Any]:
    """Per-watch totals and the efficiency ratios named in docs/CSC2524_EVALUATION.md, with raw numerators."""
    runs = store.list_runs(watch_id)
    metrics = [r["metrics"] or {} for r in runs]

    def total(key: str) -> float:
        return sum(m.get(key, 0) or 0 for m in metrics)

    accepted = int(store.scalar("SELECT COUNT(*) FROM claim WHERE watch_id = ? AND status = ?", (watch_id, ClaimStatus.ACCEPTED)) or 0)
    rejected = int(store.scalar("SELECT COUNT(*) FROM claim WHERE watch_id = ? AND status = ?", (watch_id, ClaimStatus.REJECTED)) or 0)
    findings = int(store.scalar("SELECT COUNT(*) FROM finding WHERE watch_id = ?", (watch_id,)) or 0)
    requests = store.all("SELECT signal, status FROM checkpoint_request WHERE watch_id = ?", (watch_id,))
    decisions = store.all("SELECT decision, COUNT(*) AS n FROM checkpoint_decision WHERE watch_id = ? GROUP BY decision", (watch_id,))
    skipped = store.all("SELECT reason, COUNT(*) AS n FROM queue_admission WHERE watch_id = ? AND decision = 'SKIPPED' GROUP BY reason",
                        (watch_id,))
    searches = total("search_requests")
    tokens = total("llm_output_tokens")
    watch = store.get_watch(watch_id) or {}
    return {
        "watch_id": watch_id, "status": watch.get("status"), "policy": watch.get("checkpoint_policy"),
        "sessions": len(runs),
        "sessions_by_status": {s: sum(1 for r in runs if r["status"] == s) for s in sorted({r["status"] for r in runs})},
        "sessions_with_meaningful_update": sum(1 for r in runs if r["has_meaningful_update"]),
        "search_requests": int(searches), "search_failures": int(total("search_failures")),
        "pages_fetched": int(total("fetched_pages")), "pages_read": int(total("pages_read")),
        "llm_calls": int(total("llm_calls")), "llm_prompt_tokens": int(total("llm_prompt_tokens")),
        "llm_output_tokens": int(tokens), "llm_seconds": round(total("llm_seconds"), 1),
        "wall_seconds": round(total("wall_seconds"), 1),
        "findings_total": findings, "findings_new_reported": int(total("findings_new")),
        "findings_updated_reported": int(total("findings_updated")), "findings_stale_reported": int(total("findings_stale")),
        "claims_verified": accepted, "claims_rejected": rejected,
        "queries_rejected_as_repeats": int(total("queries_rejected_as_repeats")),
        "known_urls_skipped": int(total("duplicate_urls_skipped")),
        "duplicate_claims_suppressed": int(total("duplicate_claims_suppressed")),
        "out_of_scope_candidates": int(total("out_of_scope_candidates")),
        "reviews_requested": len(requests),
        "reviews_by_signal": {s: sum(1 for r in requests if r["signal"] == s) for s in sorted({r["signal"] for r in requests})},
        "checkpoint_decisions": {d["decision"]: d["n"] for d in decisions},
        "queue_turns_skipped": {s["reason"]: s["n"] for s in skipped},
        "new_findings_per_100_searches": _ratio(total("findings_new"), searches, 100),
        "new_findings_per_1k_output_tokens": _ratio(total("findings_new"), tokens, 1000),
        "verified_claims_per_100_searches": _ratio(accepted, searches, 100),
    }


def export_all(store: Store, out_dir: Path, include_text: bool = False) -> dict[str, Any]:
    """Write one JSONL file per table plus summary.json. Returns the summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    tables = dict(TABLES)
    if include_text:
        tables["source_snapshots"] = "SELECT * FROM source_snapshot ORDER BY id"
        tables["reports"] = "SELECT run_id, watch_id, schema_version, has_meaningful_update, json, created_at FROM report ORDER BY created_at"
    for name, sql in tables.items():
        rows = store.all(sql)
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(_decode(row), ensure_ascii=False, sort_keys=True, default=str) + "\n")
        counts[name] = len(rows)
    summary = {
        "schema_version": store.schema_version(),
        "row_counts": counts,
        "watches": [watch_summary(store, w["id"]) for w in store.list_watches()],
        "notes": [
            "Ratios are given with their raw numerators and denominators in the same record.",
            "'new findings' counts findings the engine classified NEW; relevance to the objective is not judged here.",
            "Runs whose model is 'simulated' or whose search provider is 'fixture' are not real research.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return summary
