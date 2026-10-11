"""Recall: a bounded view of everything earlier sessions left behind for a watch."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from forage.fetch.urls import registrable_domain
from forage.storage.store import Store
from forage.types import ClaimStatus, QuestionStatus, RunStatus, SourceStatus, WatchSpec
from forage.util import parse_iso

MAX_KNOWN_FOR_PLANNER = 24
MAX_TRIED_FOR_PLANNER = 40
MAX_QUESTIONS_FOR_PLANNER = 8


@dataclass
class FindingView:
    id: str
    title: str
    subject: str
    status: str
    claims: list[dict[str, Any]] = field(default_factory=list)
    domains: set[str] = field(default_factory=set)

    @property
    def text(self) -> str:
        return " ".join([self.title, self.subject, *[c["text"] for c in self.claims]])


@dataclass
class Memory:
    session_number: int
    prior_sessions: int
    last_session_at: str | None
    findings: dict[str, FindingView]
    open_questions: list[dict[str, Any]]
    tried_queries: list[str]  # every query that actually reached a provider, oldest first
    recent_queries: list[str]  # run within the re-query interval: must not be repeated yet
    rerunnable: list[dict[str, Any]]  # older queries that produced kept facts, most productive first
    tried_for_planner: list[dict[str, Any]]
    leads: list[dict[str, Any]]
    backlog: list[dict[str, Any]]  # relevant sources discovered earlier but never read
    recheck: list[dict[str, Any]]  # read sources whose evidence is older than the freshness window

    def known_for_planner(self) -> list[dict[str, Any]]:
        views = sorted(self.findings.values(), key=lambda f: -len(f.claims))[:MAX_KNOWN_FOR_PLANNER]
        return [{"id": f.id, "title": f.title, "claim": f.claims[0]["text"] if f.claims else ""} for f in views]

    def questions_for_planner(self) -> list[dict[str, Any]]:
        ordered = sorted(self.open_questions, key=lambda q: (q["attempts"], q["created_at"]))
        return [{"id": q["id"], "text": q["text"], "attempts": q["attempts"]} for q in ordered[:MAX_QUESTIONS_FOR_PLANNER]]

    def summary(self) -> dict[str, int]:
        return {
            "prior_sessions": self.prior_sessions,
            "known_findings": len(self.findings),
            "open_questions": len(self.open_questions),
            "queries_already_tried": len(self.tried_queries),
            "queries_too_recent_to_repeat": len(self.recent_queries),
            "open_leads": len(self.leads),
            "unread_relevant_sources": len(self.backlog),
            "sources_due_for_recheck": len(self.recheck),
        }


def load_findings(store: Store, watch_id: str) -> dict[str, FindingView]:
    views: dict[str, FindingView] = {}
    for row in store.list_findings(watch_id):
        views[row["id"]] = FindingView(row["id"], row["title"], row["subject"], row["status"])
    domains: dict[str, str] = {s["id"]: s["domain"] for s in store.list_sources(watch_id)}
    for claim in store.list_claims(watch_id=watch_id, status=ClaimStatus.ACCEPTED):
        view = views.get(claim["finding_id"])
        if view is not None:
            view.claims.append(claim)
            view.domains.add(registrable_domain(domains.get(claim["source_id"], "")))
    return views


def build_memory(store: Store, watch_id: str, spec: WatchSpec, now: datetime, requery_after_hours: float = 20.0) -> Memory:
    runs = store.list_runs(watch_id)
    prior = [r for r in runs if r["status"] != RunStatus.RUNNING]
    current = next((r for r in runs if r["status"] == RunStatus.RUNNING), None)
    session_number = current["seq"] if current else len(runs) + 1

    useful: dict[str, int] = {}
    for row in store.all(
        "SELECT s.discovered_via AS q, COUNT(c.id) AS n FROM source s JOIN claim c ON c.source_id = s.id "
        "WHERE s.watch_id = ? AND c.status = ? GROUP BY s.discovered_via",
        (watch_id, ClaimStatus.ACCEPTED),
    ):
        useful[row["q"] or ""] = row["n"]

    tried: list[str] = []
    tried_view: list[dict[str, Any]] = []
    last_run: dict[str, tuple[datetime, str, int]] = {}  # normalised query -> (last run, text, facts kept)
    cutoff = now - timedelta(hours=requery_after_hours)
    for attempt in store.list_search_attempts(watch_id):
        if attempt["status"] not in ("OK", "EMPTY"):
            continue  # throttled or failed queries never ran, so they may be tried again
        tried.append(attempt["query"])
        kept = useful.get(attempt["query"], 0)
        if attempt["status"] == "EMPTY":
            outcome = "no results"
        else:
            outcome = f"{attempt['result_count']} results, {attempt['new_url_count']} new pages, {kept} facts kept"
        tried_view.append({"query": attempt["query"], "outcome": outcome})
        last_run[attempt["query_norm"]] = (parse_iso(attempt["at"]), attempt["query"], kept)
    # Repeating a search is pointless within the interval and necessary after it: new pages appear over time.
    recent = [text for at, text, _ in last_run.values() if at > cutoff]
    rerunnable = sorted(
        ({"query": text, "facts_kept": kept, "last_run": at.strftime("%Y-%m-%d")}
         for at, text, kept in last_run.values() if at <= cutoff and kept > 0),
        key=lambda entry: -entry["facts_kept"])

    backlog = [
        s for s in store.list_sources(watch_id, [SourceStatus.DISCOVERED]) if (s["relevance"] or 0) >= 2
    ]
    backlog += [
        s for s in store.list_sources(watch_id, [SourceStatus.FAILED])
        if "transient_failure" in (s["flags"] or []) and s["fetch_count"] < 2 and (s["relevance"] or 0) >= 2
    ]
    backlog.sort(key=lambda s: (-(s["relevance"] or 0), s["first_seen_at"]))

    recheck: list[dict[str, Any]] = []
    if spec.freshness_days:
        cutoff = now - timedelta(days=spec.freshness_days)
        with_claims = {
            row["source_id"] for row in store.all(
                "SELECT DISTINCT source_id FROM claim WHERE watch_id = ? AND status = ?", (watch_id, ClaimStatus.ACCEPTED))
        }
        for source in store.list_sources(watch_id, [SourceStatus.FETCHED]):
            if source["id"] in with_claims and source["last_fetched_at"] and parse_iso(source["last_fetched_at"]) <= cutoff:
                recheck.append(source)
        recheck.sort(key=lambda s: s["last_fetched_at"])

    return Memory(
        session_number=session_number,
        prior_sessions=len(prior),
        last_session_at=prior[-1]["started_at"][:10] if prior else None,
        findings=load_findings(store, watch_id),
        open_questions=store.list_questions(watch_id, QuestionStatus.OPEN),
        tried_queries=tried,
        recent_queries=recent,
        rerunnable=rerunnable,
        tried_for_planner=tried_view[-MAX_TRIED_FOR_PLANNER:],
        leads=store.list_leads(watch_id, "OPEN"),
        backlog=backlog,
        recheck=recheck,
    )


def select_passages(text: str, focus: set[str], budget_chars: int) -> str:
    """Pick the paragraphs most related to the focus terms, in page order, within a character budget."""
    from forage.util import token_set

    if len(text) <= budget_chars:
        return text
    paragraphs: list[str] = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        if paragraphs and len(paragraphs[-1]) < 200:
            paragraphs[-1] += "\n" + block
        else:
            paragraphs.append(block)
    # Very long paragraphs are cut at sentence-ish boundaries so one block cannot use the whole budget.
    pieces: list[str] = []
    for para in paragraphs:
        while len(para) > 1500:
            cut = para.rfind(". ", 600, 1500)
            cut = cut + 1 if cut > 0 else 1500
            pieces.append(para[:cut].strip())
            para = para[cut:].strip()
        if para:
            pieces.append(para)

    scores = []
    for i, piece in enumerate(pieces):
        words = token_set(piece)
        hits = len(words & focus)
        density = hits / (len(words) ** 0.5 + 1)
        has_number = any(ch.isdigit() for ch in piece)
        scores.append((hits + density + (0.3 if has_number else 0.0), i))

    chosen: set[int] = set()
    used = 0
    lead = 0
    while lead < len(pieces) and used + len(pieces[lead]) <= min(700, budget_chars):
        chosen.add(lead)
        used += len(pieces[lead]) + 1
        lead += 1
    for score, i in sorted(scores, key=lambda pair: (-pair[0], pair[1])):
        if i in chosen or score <= 0:
            continue
        if used + len(pieces[i]) + 8 > budget_chars:
            continue
        chosen.add(i)
        used += len(pieces[i]) + 8
    if not chosen:
        return text[:budget_chars]
    out: list[str] = []
    previous = -1
    for i in sorted(chosen):
        if previous >= 0 and i != previous + 1:
            out.append("[...]")
        out.append(pieces[i])
        previous = i
    return "\n".join(out)
