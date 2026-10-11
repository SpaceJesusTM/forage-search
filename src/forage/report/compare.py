"""Before/after comparison of two sessions of one watch, built from their stored reports and logs."""

from __future__ import annotations

import json
from typing import Any

from forage.engine.safety import md_text, md_url
from forage.storage.store import Store
from forage.util import normalize_text


def _load(store: Store, run_id: str) -> dict[str, Any]:
    row = store.get_report(run_id)
    if row is None:
        raise ValueError(f"run '{run_id}' has no stored report")
    return json.loads(row["json"])


def _session(report: dict[str, Any]) -> dict[str, Any]:
    kinds = [f["change_type"] for f in report["findings"]]
    m = report["metrics"]
    return {
        "session": report["session_number"], "run_id": report["run_id"], "status": report["status"],
        "started_at": report["started_at"], "headline": report["headline"],
        "searches": [s["query"] for s in report["searches"]],
        "pages_fetched": m.get("fetched_pages", 0), "pages_read": m.get("pages_read", 0),
        "wall_seconds": m.get("wall_seconds", 0), "llm_output_tokens": m.get("llm_output_tokens", 0),
        "findings_new": kinds.count("NEW"), "findings_updated": kinds.count("UPDATED"),
        "findings_corroborated": kinds.count("CORROBORATED"), "findings_stale": kinds.count("STALE"),
        "findings_unchanged": report["unchanged_findings"], "claims_kept": m.get("claims_accepted", 0),
        "claims_rejected": m.get("claims_rejected", 0), "questions_answered": len(report["questions_answered"]),
        "open_questions_after": len(report["questions_open"]),
    }


def compare_runs(store: Store, run_a: str, run_b: str) -> dict[str, Any]:
    a, b = _load(store, run_a), _load(store, run_b)
    if a["watch_id"] != b["watch_id"]:
        raise ValueError("both runs must belong to the same watch")
    if a["session_number"] > b["session_number"]:
        a, b = b, a
    queries_a = {normalize_text(q): q for q in (s["query"] for s in a["searches"])}
    urls_a = {s["url"] for s in a["sources"]}
    plan_b = next((e["data"] for e in store.list_events(b["run_id"]) if e["kind"] == "plan"), {})
    recall_b = next((e["data"] for e in store.list_events(b["run_id"]) if e["kind"] == "recall"), {})
    known_before_b = {f["finding_id"] for f in a["findings"]}

    def brief(finding: dict[str, Any], only_new: bool) -> dict[str, Any]:
        claims = [c for c in finding["claims"] if c["new_this_session"]] if only_new else finding["claims"]
        return {"finding_id": finding["finding_id"], "title": finding["title"], "change_type": finding["change_type"],
                "claims": [{"text": c["text"], "quote": c["quote"], "evidence_url": c["evidence_url"]} for c in claims[:4]]}

    return {
        "watch_id": a["watch_id"], "watch_title": a["watch_title"], "simulated": a["simulated"] or b["simulated"],
        "before": _session(a), "after": _session(b),
        "state_recalled_by_later_session": recall_b,
        "queries_repeated_verbatim": [q for q in (s["query"] for s in b["searches"]) if normalize_text(q) in queries_a],
        "queries_rejected_as_repeats": plan_b.get("rejected_as_repeats", []),
        "urls_fetched_in_both": sorted(urls_a & {s["url"] for s in b["sources"]}),
        "known_urls_skipped_in_later_session": b["comparison"]["known_urls_skipped"],
        "saved_leads_followed_in_later_session": b["comparison"]["saved_leads_followed"],
        "duplicate_claims_suppressed_in_later_session": b["comparison"]["duplicate_claims_suppressed"],
        "new_in_later_session": [brief(f, False) for f in b["findings"] if f["change_type"] == "NEW"],
        "updated_in_later_session": [brief(f, True) for f in b["findings"] if f["change_type"] == "UPDATED"],
        "corroborated_in_later_session": [brief(f, True) for f in b["findings"] if f["change_type"] == "CORROBORATED"],
        "earlier_findings_touched_again": sorted(known_before_b & {f["finding_id"] for f in b["findings"]}),
        "questions_answered_in_later_session": [q["text"] for q in b["questions_answered"]],
        "known_findings": {"before": b["comparison"]["known_findings_before"], "after": b["comparison"]["known_findings_after"]},
    }


def render_comparison(c: dict[str, Any]) -> str:
    a, b = c["before"], c["after"]
    lines = [f"# {md_text(c['watch_title'], 160)} — session {a['session']} vs session {b['session']}", ""]
    if c["simulated"]:
        lines += ["> **SIMULATED.** At least one of these sessions used fixture pages or a simulated model.", ""]
    lines += [
        f"| | Session {a['session']} | Session {b['session']} |", "| --- | ---: | ---: |",
        f"| Searches run | {len(a['searches'])} | {len(b['searches'])} |",
        f"| Pages fetched | {a['pages_fetched']} | {b['pages_fetched']} |",
        f"| Claims kept / rejected | {a['claims_kept']} / {a['claims_rejected']} | {b['claims_kept']} / {b['claims_rejected']} |",
        f"| New findings | {a['findings_new']} | {b['findings_new']} |",
        f"| Updated findings | {a['findings_updated']} | {b['findings_updated']} |",
        f"| Corroborated findings | {a['findings_corroborated']} | {b['findings_corroborated']} |",
        f"| Seen again, nothing new | {a['findings_unchanged']} | {b['findings_unchanged']} |",
        f"| Questions answered | {a['questions_answered']} | {b['questions_answered']} |",
        f"| Open questions afterwards | {a['open_questions_after']} | {b['open_questions_after']} |",
        f"| Model output tokens | {a['llm_output_tokens']} | {b['llm_output_tokens']} |",
        f"| Wall time (s) | {a['wall_seconds']} | {b['wall_seconds']} |", "",
        f"Known findings: {c['known_findings']['before']} before session {b['session']}, {c['known_findings']['after']} after.", "",
        f"## What session {b['session']} started from", "",
    ]
    recall = c["state_recalled_by_later_session"]
    lines += [f"- {key.replace('_', ' ')}: {value}" for key, value in recall.items()] or ["- (no recall record)"]
    lines += ["", "## Did it repeat itself?", "",
              f"- Searches repeated word for word: {len(c['queries_repeated_verbatim'])}",
              f"- Proposed searches rejected as repeats before running: {len(c['queries_rejected_as_repeats'])}"]
    lines += [f"  - {md_text(q, 160)}" for q in c["queries_rejected_as_repeats"]]
    lines += [f"- Search results already visited and skipped: {c['known_urls_skipped_in_later_session']}",
              f"- Saved leads from earlier sessions followed up: {c['saved_leads_followed_in_later_session']}",
              f"- Pages fetched in both sessions: {len(c['urls_fetched_in_both'])}",
              f"- Duplicate claims suppressed: {c['duplicate_claims_suppressed_in_later_session']}", "",
              f"### Searches in session {a['session']}", ""]
    lines += [f"- {md_text(q, 160)}" for q in a["searches"]] or ["- none"]
    lines += ["", f"### Searches in session {b['session']}", ""]
    lines += [f"- {md_text(q, 160)}" for q in b["searches"]] or ["- none"]
    for key, title in (("new_in_later_session", "New"), ("updated_in_later_session", "Updated (new claims only)"),
                       ("corroborated_in_later_session", "Corroborated")):
        lines += ["", f"## {title} in session {b['session']}", ""]
        if not c[key]:
            lines.append("None.")
        for finding in c[key]:
            lines.append(f"- **{md_text(finding['title'], 160)}** `{finding['finding_id']}`")
            for claim in finding["claims"]:
                url = md_url(claim["evidence_url"])
                lines.append(f"  - {md_text(claim['text'], 400)} — <{url}>" if url else f"  - {md_text(claim['text'], 400)}")
    if c["questions_answered_in_later_session"]:
        lines += ["", f"## Questions answered in session {b['session']} (model judgement)", ""]
        lines += [f"- {md_text(q, 300)}" for q in c["questions_answered_in_later_session"]]
    if not (c["new_in_later_session"] or c["updated_in_later_session"]):
        lines += ["", f"Session {b['session']} verified nothing new. That reflects what the searches returned, "
                      "not proof that nothing new exists."]
    return "\n".join(lines) + "\n"
