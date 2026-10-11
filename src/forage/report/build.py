"""Build a session report from stored state. Deterministic: the same rows give the same report.

Nothing here calls a model. Model-written text (summary, answers) is read back
from the run log and placed in fields that are labelled as inference.
"""

from __future__ import annotations

import json
import re
from datetime import timedelta
from typing import Any

from forage.config import Settings
from forage.fetch.urls import registrable_domain
from forage.report.markdown import render_markdown
from forage.report.schema import (
    PendingReview, ReportClaim, ReportComparison, ReportDirection, ReportFinding, ReportQuestion, ReportSearch,
    ReportSource, ResearchReport,
)
from forage.storage.store import Store
from forage.types import ChangeType, ClaimStatus, QuestionStatus, Verification
from forage.util import normalize_text, parse_iso

MAX_CLAIMS_PER_FINDING = 12
_REJECTION_TEXT = {
    "quote_not_found": "the quoted passage could not be found on the page",
    "number_not_in_source": "a number in the claim does not appear on the page",
    "unsupported_by_quote": "the quote did not support the claim",
    "not_relevant_to_objective": "the claim did not bear on the objective",
    "instruction_like_text": "the text read like an instruction rather than a fact",
    "claim_unrelated_to_quote": "claim and quote shared no vocabulary",
    "name_not_in_source": "a name in the claim does not appear on the page",
    "claim_length": "the claim was too short or too long",
}


def _question(row: dict[str, Any]) -> ReportQuestion:
    return ReportQuestion(
        question_id=row["id"], text=row["text"], origin=row["origin"], status=row["status"],
        searches_so_far=row["attempts"], answer=row["resolution"], answered_by=row["resolved_by"] or [],
    )


def build_report(store: Store, run_id: str, settings: Settings | None = None) -> ResearchReport:
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"unknown run '{run_id}'")
    watch = store.get_watch(run["watch_id"])
    revision = store.get_revision(run["revision_id"])
    assert watch is not None and revision is not None
    watch_id = watch["id"]
    metrics: dict[str, Any] = run["metrics"] or {}
    events = store.list_events(run_id)
    sources = {s["id"]: s for s in store.list_sources(watch_id)}
    seq_by_run = {r["id"]: r["seq"] for r in store.list_runs(watch_id)}
    started = parse_iso(run["started_at"])

    # -- findings touched by this run ------------------------------------
    changes = store.list_changes(run_id)
    findings: list[ReportFinding] = []
    unchanged = 0
    for change in sorted(changes, key=lambda c: (-ChangeType(c["change_type"]).rank, c["id"])):
        if change["change_type"] == ChangeType.UNCHANGED:
            unchanged += 1
            continue
        row = store.get_finding(change["finding_id"])
        if row is None:
            continue
        detail = change["detail"] or {}
        accepted = store.list_claims(finding_id=row["id"], status=ClaimStatus.ACCEPTED)
        shown = accepted
        if change["change_type"] == ChangeType.STALE and not accepted:
            shown = store.list_claims(finding_id=row["id"], status=ClaimStatus.SUPERSEDED)
        new_ids = set(detail.get("new_claim_ids", [])) | set(detail.get("corroborating_claim_ids", []))
        ordered = sorted(shown, key=lambda c: (c["id"] not in new_ids, c["checked_at"]))
        shown, hidden = ordered[:MAX_CLAIMS_PER_FINDING], max(0, len(ordered) - MAX_CLAIMS_PER_FINDING)

        claims, notes = [], []
        domains = {registrable_domain(sources[c["source_id"]]["domain"]) for c in accepted if c["source_id"] in sources}
        for claim in shown:
            source = sources.get(claim["source_id"])
            if source is None:
                continue
            claims.append(ReportClaim(
                claim_id=claim["id"], text=claim["text"], quote=claim["quote"],
                evidence_url=source["final_url"] or source["url"], source_id=source["id"], source_title=source["title"],
                source_domain=source["domain"], checked_at=claim["last_confirmed_at"],
                verification_status="quote_verified" if claim["verification"] == Verification.QUOTE_EXACT
                else "quote_verified_approximate",
                support={"SUPPORTED": "supported", "PARTIAL": "partial"}.get(claim["support"], "unchecked"),
                new_this_session=claim["id"] in new_ids, note=claim["note"],
            ))
        if row["uncertainty"]:
            notes.append(f"Model note: {row['uncertainty']}")
        if len(domains) <= 1 and change["change_type"] != ChangeType.STALE:
            notes.append("Reported by a single source; not independently corroborated.")
        if any(c.support == "partial" for c in claims):
            notes.append("At least one claim is only partly backed by its quote.")
        if any(c.support == "unchecked" for c in claims):
            notes.append("At least one claim was not put through the support check.")
        if any(c.verification_status == "quote_verified_approximate" for c in claims):
            notes.append("At least one quote was matched approximately rather than word for word.")
        for source_id in {c.source_id for c in claims}:
            source = sources[source_id]
            if "instruction_like_text" in (source["flags"] or []):
                notes.append(f"Source {source['domain']} contains instruction-like text and was handled as untrusted.")
            published = (source["published_at"] or "")[:10]
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", published):
                try:
                    if started - parse_iso(published + "T00:00:00Z") > timedelta(days=365):
                        notes.append(f"Source {source['domain']} is dated {published}; it may be out of date.")
                except ValueError:
                    pass
        retired = [c["text"] for c in (store.get_claim(cid) for cid in detail.get("superseded_claim_ids", [])) if c]
        findings.append(ReportFinding(
            finding_id=row["id"], change_type=change["change_type"], title=row["title"], subject=row["subject"],
            claims=claims, claims_not_shown=hidden, independent_sources=len(domains),
            uncertainties=list(dict.fromkeys(notes)),
            change_note=detail.get("note"), retired_claims=retired if accepted else [],
            first_seen_session=seq_by_run.get(row["first_seen_run_id"]),
            first_seen_at=row["first_seen_at"], last_confirmed_at=row["last_confirmed_at"],
        ))

    # -- questions, directions, model-written summary --------------------
    questions = store.list_questions(watch_id)
    answered = [_question(q) for q in questions if q["resolved_run_id"] == run_id]
    opened = [_question(q) for q in questions if q["created_run_id"] == run_id and q["status"] == QuestionStatus.OPEN]
    still_open = [_question(q) for q in questions if q["status"] == QuestionStatus.OPEN]
    reflection = next((e["data"] for e in reversed(events) if e["kind"] == "reflection"), {}) or {}
    directions = [ReportDirection(query=d["text"], rationale=d["rationale"])
                  for d in store.list_leads(watch_id, status=None) if d["created_run_id"] == run_id]

    # -- sources and searches --------------------------------------------
    counts: dict[str, dict[str, int]] = {}
    for row in store.all("SELECT source_id, status, COUNT(*) AS n FROM claim WHERE run_id = ? GROUP BY source_id, status", (run_id,)):
        counts.setdefault(row["source_id"], {})[row["status"]] = row["n"]
    report_sources = []
    for event in events:
        if event["kind"] != "source_result":
            continue
        data = event["data"]
        source = sources.get(data["source_id"])
        if source is None:
            continue
        report_sources.append(ReportSource(
            source_id=source["id"], url=source["final_url"] or source["url"], title=source["title"], domain=source["domain"],
            outcome=data["outcome"], fetched_at=event["at"], http_status=data.get("http_status"),
            claims_kept=counts.get(source["id"], {}).get(ClaimStatus.ACCEPTED, 0),
            claims_rejected=counts.get(source["id"], {}).get(ClaimStatus.REJECTED, 0),
            flags=[f for f in (source["flags"] or []) if not f.startswith("same_content_as")], note=data.get("note"),
        ))
    attempts = store.list_search_attempts(watch_id, run_id)
    searches = [ReportSearch(query=a["query"], rationale=a["rationale"], provider=a["provider"], status=a["status"],
                             results=a["result_count"], new_urls=a["new_url_count"], known_urls=a["known_url_count"])
                for a in attempts]

    # -- comparison with the state this session started from --------------
    recall = next((e["data"] for e in events if e["kind"] == "recall"), None) or metrics.get("recall", {})
    earlier_queries: dict[str, str] = {}  # normalised query -> when it last ran before this session
    for a in store.list_search_attempts(watch_id):
        if a["run_id"] != run_id and a["status"] in ("OK", "EMPTY") and seq_by_run.get(a["run_id"], 0) < run["seq"]:
            earlier_queries[normalize_text(a["query"])] = a["at"]
    ran_now = [normalize_text(a["query"]) for a in attempts if a["status"] in ("OK", "EMPTY")]
    reused = [q for q in ran_now if q in earlier_queries]
    interval = timedelta(hours=settings.requery_after_hours if settings else 20.0)
    too_soon = [q for q in reused if started - parse_iso(earlier_queries[q]) < interval]
    previous = [r for r in store.list_runs(watch_id) if r["seq"] < run["seq"]]
    comparison = ReportComparison(
        previous_run_id=previous[-1]["id"] if previous else None,
        prior_sessions=recall.get("prior_sessions", len(previous)),
        known_findings_before=recall.get("known_findings", 0),
        known_findings_after=recall.get("known_findings", 0) + metrics.get("findings_new", 0),
        open_questions_before=recall.get("open_questions", 0),
        open_questions_after=recall.get("open_questions", 0) - len(answered) + len(opened),
        queries_already_tried_before=recall.get("queries_already_tried", 0),
        queries_repeated=len(too_soon),
        queries_rerun_after_interval=len(reused) - len(too_soon),
        queries_rejected_as_repeats=metrics.get("queries_rejected_as_repeats", 0),
        known_urls_skipped=metrics.get("duplicate_urls_skipped", 0),
        saved_leads_followed=metrics.get("backlog_sources_read", 0),
        sources_rechecked=metrics.get("rechecked_sources", 0),
        duplicate_claims_suppressed=metrics.get("duplicate_claims_suppressed", 0),
    )

    # -- report-level caveats --------------------------------------------
    caveats = list(metrics.get("notes", []))
    rejected = metrics.get("claims_rejected_by_reason", {})
    if rejected:
        parts = [f"{n} because {_REJECTION_TEXT.get(reason, reason)}" for reason, n in rejected.items()]
        caveats.append(f"{sum(rejected.values())} extracted claim(s) were discarded and are not reported: " + "; ".join(parts) + ".")
    failed_searches = [a for a in attempts if a["status"] not in ("OK", "EMPTY")]
    if failed_searches:
        caveats.append(f"{len(failed_searches)} search request(s) failed or were throttled, so coverage is incomplete.")
    if attempts and not ran_now:
        caveats.append("No search succeeded this session; the absence of new findings says nothing about the topic.")
    if metrics.get("fetch_failures"):
        caveats.append(f"{metrics['fetch_failures']} page(s) could not be retrieved.")
    if metrics.get("budget_exhausted"):
        caveats.append("The session reached its budget limit for: " + ", ".join(metrics["budget_exhausted"])
                       + ". More may have been found with a larger budget.")
    if metrics.get("llm_failures"):
        caveats.append(f"{metrics['llm_failures']} model call(s) returned unusable output and were skipped.")
    if metrics.get("sources_flagged_instruction_like"):
        caveats.append(f"{metrics['sources_flagged_instruction_like']} page(s) contained instruction-like text; "
                       "those passages were withheld from the model and the pages are flagged.")
    if run["status"] != "COMPLETED":
        caveats.append(f"This session ended with status {run['status']}; its results are partial.")
    caveats.append("Verification confirms that a source states a claim. It does not establish that the claim is true.")

    kinds = [f.change_type for f in findings]
    pieces = []
    for kind, label in (("NEW", "new"), ("UPDATED", "updated"), ("STALE", "no longer confirmed"), ("CORROBORATED", "corroborated")):
        if kinds.count(kind):
            pieces.append(f"{kinds.count(kind)} {label}")
    if answered:
        pieces.append(f"{len(answered)} question(s) answered")
    meaningful = bool(run["has_meaningful_update"])
    if meaningful:
        headline = "Findings: " + ", ".join(pieces) + "."
    elif pieces:
        headline = "No new information; existing findings " + ", ".join(pieces) + "."
    else:
        headline = "No meaningful update: nothing new was verified this session."

    pending = store.one("SELECT * FROM checkpoint_request WHERE run_id = ? AND status = 'PENDING'", (run_id,))
    pending_review = None
    if pending:
        pending_review = PendingReview(
            request_id=pending["id"], policy=pending["policy"], signal=pending["signal"], question=pending["question"],
            choices=json.loads(pending["choices"]), safe_default=pending["safe_default"], created_at=pending["created_at"])

    model = run["model"]
    provider = run["search_provider"]
    return ResearchReport(
        run_id=run_id, watch_id=watch_id, watch_title=watch["title"], objective=revision["objective"],
        session_number=run["seq"], status=run["status"].lower(), stop_reason=run["stop_reason"], error=run["error"],
        started_at=run["started_at"], finished_at=run["finished_at"], model=model, search_provider=provider,
        simulated=(model == "simulated" or provider == "fixture"), has_meaningful_update=meaningful, headline=headline,
        summary=_clean_summary(reflection.get("summary") or "", {f.finding_id for f in findings}) or None,
        findings=findings, unchanged_findings=unchanged,
        known_findings_total=int(store.scalar("SELECT COUNT(*) FROM finding WHERE watch_id = ?", (watch_id,)) or 0),
        questions_answered=answered, questions_opened=opened, questions_open=still_open, next_directions=directions,
        uncertainties=caveats, sources=report_sources, searches=searches, comparison=comparison,
        pending_review=pending_review,
        metrics={k: v for k, v in metrics.items() if k not in ("recall", "notes", "out_of_scope_examples", "guidance_proposed")},
        budget=run["budget"] or {},
    )


def _clean_summary(summary: str, valid_ids: set[str]) -> str:
    """Drop bracketed finding ids that do not exist, so the summary cannot cite something unreal."""
    def keep(match: re.Match[str]) -> str:
        return match.group(0) if match.group(1) in valid_ids else ""

    return " ".join(re.sub(r"\[(f-[0-9a-f]{6,})\]", keep, summary).split())


def write_report(store: Store, settings: Settings, run_id: str) -> ResearchReport:
    report = build_report(store, run_id, settings)
    payload = report.model_dump_json(indent=2)
    ResearchReport.model_validate_json(payload)  # the stored artifact must satisfy its own contract
    markdown = render_markdown(report)
    folder = settings.reports_dir / report.watch_id
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{report.session_number:03d}_{run_id}"
    json_path = folder / f"{stem}.json"
    md_path = folder / f"{stem}.md"
    json_path.write_text(payload, encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    store.save_report(
        run_id=run_id, watch_id=report.watch_id, schema_version=report.schema_version,
        has_meaningful_update=report.has_meaningful_update, json=payload, markdown=markdown,
        json_path=str(json_path), markdown_path=str(md_path),
    )
    return report
