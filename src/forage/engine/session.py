"""One bounded research session (Boundary A).

Input: an admitted run row for an approved watch revision. Output: updated research
memory in the store, a structured report, metrics, and at most a *candidate*
guidance proposal. The engine cannot change a watch's objective or status and
knows nothing about schedulers, checkpoint policies or user interfaces.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

from forage.config import Settings
from forage.engine import prompts
from forage.engine.budget import BudgetTracker
from forage.engine.matching import (
    MAYBE_CLAIM, MAYBE_FINDING, SAME_CLAIM, SAME_FINDING, is_repeat_query, rank_candidates, text_similarity,
)
from forage.engine.memory import FindingView, Memory, build_memory, select_passages
from forage.engine.safety import redact_injection, scan_for_injection
from forage.engine.schemas import (
    Extraction, ExtractedItem, GuidanceProposal, MatchBatch, NoveltyCheck, Plan, PlannedQuery, Reflection,
    SupportCheck, Triage,
)
from forage.engine.verify import ClaimCheck, check_claim, claim_coverage, locate_quote, normalize_with_map
from forage.fetch.base import Fetcher, FetchResult
from forage.fetch.urls import UrlError, canonicalize_url, check_fetchable, registrable_domain
from forage.llm.base import LLMClient, LLMOutputError, LLMUnavailable
from forage.search.base import SearchError, SearchProvider, SearchRateLimited
from forage.storage.store import Store
from forage.types import (
    Budget, ChangeType, ClaimStatus, QuestionStatus, RunStatus, SourceStatus, Support, Verification, WatchSpec,
)
from forage.util import iso, normalize_text, sha256_hex, token_set, tokenize, utcnow

MAX_ITEMS_PER_PAGE = 3
MAX_CLAIMS_PER_ITEM = 4
MAX_OPEN_QUESTIONS = 12
TRANSIENT_OUTCOMES = {"TIMEOUT", "NETWORK_ERROR"}


class SessionCancelled(Exception):
    """The run was cancelled or its watch stopped being active."""


@dataclass
class SessionOutcome:
    run_id: str
    watch_id: str
    status: RunStatus
    has_meaningful_update: bool = False
    guidance: GuidanceProposal | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class _Accepted:
    text: str
    check: ClaimCheck
    support: Support = Support.UNCHECKED
    dropped: bool = False


class ResearchEngine:
    def __init__(
        self, store: Store, llm: LLMClient, search: SearchProvider, fetcher: Fetcher, settings: Settings, *,
        clock: Callable = utcnow, sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic, progress: Callable[[str], None] | None = None,
    ):
        self.store = store
        self.llm = llm
        self.search = search
        self.fetcher = fetcher
        self.settings = settings
        self.clock = clock
        self.sleep = sleep
        self.monotonic = monotonic
        self.progress = progress or (lambda message: None)

    def execute(self, run_id: str, should_cancel: Callable[[], bool] | None = None,
                allow_guidance: bool = True) -> SessionOutcome:
        return _Session(self, run_id, should_cancel, allow_guidance).run()


class _Session:
    def __init__(self, engine: ResearchEngine, run_id: str, should_cancel: Callable[[], bool] | None, allow_guidance: bool):
        self.e = engine
        self.store = engine.store
        self.settings = engine.settings
        run = self.store.get_run(run_id)
        if run is None or run["status"] != RunStatus.RUNNING:
            raise ValueError(f"run '{run_id}' is not an admitted running session")
        self.run_row = run
        self.run_id = run_id
        self.watch_id = run["watch_id"]
        revision = self.store.get_revision(run["revision_id"])
        if revision is None or revision["state"] != "APPROVED":
            raise ValueError(f"run '{run_id}' does not reference an approved revision")
        # The engine works from a private copy of the approved objective; nothing below can write it back.
        self.objective: str = revision["objective"]
        self.spec = WatchSpec.model_validate(revision["spec"])
        self.budget = BudgetTracker(Budget.model_validate(run["budget"]), engine.monotonic)
        self.should_cancel = should_cancel or (lambda: False)
        self.allow_guidance = allow_guidance
        self.today = engine.clock().strftime("%Y-%m-%d")
        self.c: dict[str, Any] = {
            "results_seen": 0, "new_urls": 0, "duplicate_urls_skipped": 0, "queries_planned": 0,
            "queries_rejected_as_repeats": 0, "fetch_failures": 0, "pages_unchanged": 0, "pages_changed": 0,
            "pages_read": 0, "duplicate_pages_skipped": 0, "claims_extracted": 0, "claims_accepted": 0,
            "claims_rejected": 0, "duplicate_claims_suppressed": 0, "sources_flagged_instruction_like": 0,
            "llm_failures": 0, "search_failures": 0, "rechecked_sources": 0, "backlog_sources_read": 0,
            "out_of_scope_candidates": 0,
        }
        self.rejections: dict[str, int] = {}
        self.out_of_scope: list[dict[str, Any]] = []
        self.domain_count: dict[str, int] = {}
        self.notes: list[str] = []
        self.guidance: GuidanceProposal | None = None
        self.memory: Memory | None = None
        self.findings: dict[str, FindingView] = {}
        self.focus: set[str] = set()
        self.recall: dict[str, int] = {}
        self.hint_domains: set[str] = set()
        self.work_limits: list[str] | None = None
        self._last_attempt_id = 0

    # ------------------------------------------------------------------ run

    def run(self) -> SessionOutcome:
        status, error, stop_reason = RunStatus.COMPLETED, None, "completed"
        self._event("session_started", {"budget": self.budget.budget.model_dump(), "model": self.e.llm.model,
                                        "search_provider": self.e.search.name})
        try:
            self._prepare()
            self._recheck_stale()
            self._read_backlog(limit=max(1, self.budget.budget.max_fetched_pages // 3))
            self._search_phase()
            self._read_backlog(limit=self.budget.budget.max_fetched_pages)
            self.work_limits = self.budget.limits_reached()  # measured before wrap-up spends its reserve
            self._reflect()
        except SessionCancelled as exc:
            status, stop_reason, error = RunStatus.CANCELLED, "cancelled", str(exc) or None
        except LLMUnavailable as exc:
            status, stop_reason, error = RunStatus.FAILED, "llm_unavailable", str(exc)
        except Exception as exc:  # a session must always leave a truthful run record behind
            status, stop_reason = RunStatus.FAILED, "error"
            error = f"{type(exc).__name__}: {exc}"
            self._event("session_error", {"error": error, "traceback": traceback.format_exc()[-3000:]})
        if self.work_limits is None:  # the session ended before its work phase finished
            self.work_limits = self.budget.limits_reached()
        if status is RunStatus.COMPLETED and self.work_limits:
            stop_reason = "budget:" + ",".join(self.work_limits)
        return self._finalize(status, stop_reason, error)

    # ------------------------------------------------------------ plumbing

    def _event(self, kind: str, data: dict[str, Any] | None = None) -> None:
        self.store.add_event(self.run_id, self.watch_id, kind, data)

    def _say(self, message: str) -> None:
        self.e.progress(message)

    def _domain_full(self, domain: str, pending: int = 0) -> bool:
        """Per-site cap for one session, so one site cannot absorb the page budget. User-named sites are exempt."""
        if domain in self.hint_domains:
            return False
        return self.domain_count.get(domain, 0) + pending >= self.settings.max_per_domain_per_session

    def _source_result(self, source: dict[str, Any], outcome: str, note: str | None = None,
                       http_status: int | None = None) -> None:
        self._event("source_result", {"source_id": source["id"], "outcome": outcome, "note": note,
                                      "http_status": http_status if http_status is not None else source.get("http_status")})

    def _checkpoint(self) -> None:
        """Heartbeat and cancellation point between steps."""
        self.store.heartbeat(self.run_id)
        run = self.store.get_run(self.run_id)
        if (run and run["cancel_requested"]) or self.should_cancel():
            raise SessionCancelled("cancel requested")

    def _reject(self, reason: str) -> None:
        self.c["claims_rejected"] += 1
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    def _llm(self, purpose: str, prompt: tuple[str, str, dict[str, Any]], schema: type[BaseModel], max_tokens: int,
             *, wrapup: bool = False) -> Any | None:
        """Budgeted, logged model call. Returns None when the budget forbids it or the output was unusable."""
        allowed = self.budget.max_tokens_for_call(max_tokens, wrapup=wrapup)
        if allowed < min(max_tokens, 48):
            self.budget.flag("llm_output_tokens")
            self._event("llm_skipped", {"purpose": purpose, "reason": "token budget"})
            return None
        if not self.budget.can_llm(0, wrapup=wrapup):
            self._event("llm_skipped", {"purpose": purpose, "reason": "time budget"})
            return None
        system, user, payload = prompt
        try:
            result, usage = self.e.llm.generate(
                purpose=purpose, system=system, user=user, schema=schema, max_tokens=allowed, payload=payload)
            ok, err = True, None
        except LLMOutputError as exc:
            usage = getattr(exc, "usage", None)
            result, ok, err = None, False, str(exc)[:300]
            self.c["llm_failures"] += 1
        if usage is not None:
            self.budget.note_llm(usage)
        self.store.add_llm_call(
            run_id=self.run_id, watch_id=self.watch_id, purpose=purpose, model=self.e.llm.model,
            prompt_tokens=usage.prompt_tokens if usage else 0, output_tokens=usage.output_tokens if usage else 0,
            duration_ms=usage.duration_ms if usage else 0, attempts=usage.attempts if usage else 1, ok=ok, error=err,
            load_ms=usage.load_ms if usage else 0, prompt_eval_ms=usage.prompt_eval_ms if usage else 0,
            eval_ms=usage.eval_ms if usage else 0,
        )
        self._checkpoint()
        return result

    # ------------------------------------------------------------- prepare

    def _prepare(self) -> None:
        for question in self.spec.key_questions:
            self.store.add_question(self.watch_id, question, "SPEC")
        if not self.store.list_questions(self.watch_id):
            self.store.add_question(self.watch_id, self.objective, "SPEC")
        for hint in self.spec.source_hints:
            self._register_hint(hint)
        self.memory = build_memory(self.store, self.watch_id, self.spec, self.e.clock())
        self.findings = self.memory.findings
        self.focus = token_set(" ".join([self.objective, *self.spec.key_questions, *self.spec.include, *self.spec.priorities]))
        self.recall = self.memory.summary()  # what this session started from, before it adds anything
        self._event("recall", self.recall)
        self._say(f"session {self.memory.session_number}: recalled {len(self.findings)} findings, "
                  f"{len(self.memory.open_questions)} open questions, {len(self.memory.tried_queries)} earlier queries")

    def _register_hint(self, hint: str) -> None:
        if "/" not in hint and "." in hint and " " not in hint:
            hint = f"https://{hint}/"
        if check_fetchable(hint) is not None:
            return
        try:
            canonical = canonicalize_url(hint)
        except UrlError:
            return
        self.hint_domains.add(registrable_domain(hint))
        source, created = self.store.upsert_discovered_source(
            self.watch_id, canonical, hint, registrable_domain(hint), None, None, "user hint", self.run_id)
        if created:
            self.store.update_source(source["id"], relevance=3, triage_note="suggested by the user")

    # -------------------------------------------------------------- recheck

    def _recheck_stale(self) -> None:
        assert self.memory is not None
        for source in self.memory.recheck[: self.settings.max_rechecks_per_session]:
            if not self.budget.can_fetch():
                break
            self.c["rechecked_sources"] += 1
            self._say(f"recheck: {source['canonical_url']}")
            self._process_source(source, recheck=True)

    def _read_backlog(self, limit: int) -> None:
        """Read relevant pages that earlier sessions found but had no budget to open."""
        assert self.memory is not None
        read = 0
        while self.memory.backlog and read < limit and self.budget.can_fetch():
            source = self.memory.backlog.pop(0)
            current = self.store.get_source(source["id"])
            if current is None or current["status"] not in (SourceStatus.DISCOVERED, SourceStatus.FAILED):
                continue
            if self._domain_full(current["domain"]):
                continue
            self.c["backlog_sources_read"] += 1
            self._say(f"reading saved lead: {current['canonical_url']}")
            self._process_source(current)
            read += 1

    # --------------------------------------------------------------- search

    def _search_phase(self) -> None:
        if self.budget.budget.max_search_requests <= 0:
            return
        queries = self._plan()
        for i, planned in enumerate(queries):
            pages_left = self.budget.budget.max_fetched_pages - self.budget.fetched_pages
            if pages_left <= 0 or not self.budget.can_search():
                break
            # Spread the page budget across the remaining queries so one query cannot use it all.
            quota = max(1, min(self.settings.fetch_per_query, -(-pages_left // (len(queries) - i))))
            picks = self._search_and_triage(planned, quota)
            for source in picks:
                if not self.budget.can_fetch():
                    break
                self._process_source(source)

    def _plan(self) -> list[PlannedQuery]:
        assert self.memory is not None
        memory = self.memory
        wanted = min(self.settings.queries_per_session, self.budget.budget.max_search_requests)
        open_ids = {q["id"] for q in memory.open_questions}
        tried = list(memory.tried_queries)
        accepted: list[PlannedQuery] = []
        rejected: list[str] = []

        def consider(candidate: PlannedQuery) -> None:
            query = " ".join(candidate.query.replace('"', " ").split()[:12])
            if len(tokenize(query)) < 2 or scan_for_injection(query):
                return
            earlier = is_repeat_query(query, tried + [a.query for a in accepted])
            if earlier is not None:
                if earlier in tried:
                    self.c["queries_rejected_as_repeats"] += 1
                    rejected.append(query)
                    self._event("query_rejected_repeat", {"query": query, "repeats": earlier})
                return
            candidate.query = query
            if candidate.question_id not in open_ids:
                candidate.question_id = None
            if len(accepted) < wanted:
                accepted.append(candidate)

        def ask(extra_rejected: list[str] | None = None) -> None:
            plan = self._llm("plan", prompts.plan(
                objective=self.objective, spec=self.spec, today=self.today, session_number=memory.session_number,
                max_queries=wanted, known=memory.known_for_planner(), open_questions=memory.questions_for_planner(),
                leads=[lead["text"] for lead in memory.leads][:6], tried=memory.tried_for_planner,
                rejected=extra_rejected, last_session=memory.last_session_at), Plan, 420)
            for candidate in (plan.queries if plan else []):
                consider(candidate)

        ask()
        if len(accepted) < wanted and rejected:
            ask(list(rejected))
        # Deterministic fallbacks: saved leads, then open questions reduced to keywords.
        for lead in memory.leads:
            if len(accepted) >= wanted:
                break
            consider(PlannedQuery(query=lead["text"], rationale="direction saved by the previous session"))
        for question in memory.questions_for_planner():
            if len(accepted) >= wanted:
                break
            consider(PlannedQuery(query=" ".join(tokenize(question["text"])[:8]),
                                  rationale="keywords of an open question", question_id=question["id"]))
        # Last resort, so a watch with nothing open still looks for developments: standing questions plus a recency cue.
        year = self.today[:4]
        for question in self.spec.key_questions or [self.objective]:
            if len(accepted) >= min(wanted, 2):
                break
            for cue in ("latest", f"{year} update", "news announcement"):
                before = len(accepted)
                consider(PlannedQuery(query=" ".join(tokenize(question)[:6]) + " " + cue,
                                      rationale="standing question, checking for newer developments"))
                if len(accepted) > before:
                    break
        for lead in memory.leads:
            self.store.update_lead(lead["id"], status="USED", used_run_id=self.run_id)
        self.c["queries_planned"] = len(accepted)
        self._event("plan", {"queries": [q.model_dump() for q in accepted], "rejected_as_repeats": rejected})
        if not accepted:
            self.notes.append("No new search query could be formed that differs from earlier sessions.")
        return accepted

    def _search_and_triage(self, planned: PlannedQuery, quota: int) -> list[dict[str, Any]]:
        results = self._run_search(planned)
        if not results:
            return []
        candidates: list[dict[str, Any]] = []
        new_urls = known_urls = 0
        seen: set[str] = set()
        for result in results:
            self.c["results_seen"] += 1
            if check_fetchable(result.url) is not None:
                continue
            try:
                canonical = canonicalize_url(result.url)
            except UrlError:
                continue
            if canonical in seen:
                continue
            seen.add(canonical)
            source, created = self.store.upsert_discovered_source(
                self.watch_id, canonical, result.url, registrable_domain(result.url), result.title, result.snippet,
                planned.query, self.run_id)
            if created:
                new_urls += 1
                self.c["new_urls"] += 1
            else:
                known_urls += 1
                if source["status"] != SourceStatus.DISCOVERED:
                    self.c["duplicate_urls_skipped"] += 1
                    continue
            candidates.append(source | {"rank": result.rank, "title": result.title, "snippet": result.snippet})
        self.store.execute(
            "UPDATE search_attempt SET new_url_count = ?, known_url_count = ? WHERE id = ?",
            (new_urls, known_urls, self._last_attempt_id))
        self._say(f"search '{planned.query}': {len(results)} results, {new_urls} new, {known_urls} already known")
        if not candidates:
            return []
        return self._triage(planned, candidates, quota)

    def _run_search(self, planned: PlannedQuery) -> list:
        for attempt in range(2):
            if not self.budget.can_search():
                return []
            self.budget.note_search()
            started = self.e.monotonic()
            status, error, results = "OK", None, []
            retry_after = 0.0
            try:
                results = self.e.search.search(planned.query, self.settings.search_results_per_query)
                if not results:
                    status = "EMPTY"
            except SearchRateLimited as exc:
                status, error, retry_after = "RATE_LIMITED", str(exc)[:300], exc.retry_after
            except SearchError as exc:
                status, error = "ERROR", str(exc)[:300]
            self._last_attempt_id = self.store.add_search_attempt(
                run_id=self.run_id, watch_id=self.watch_id, query=planned.query, rationale=planned.rationale,
                target_question_id=planned.question_id, provider=getattr(self.e.search, "last_used", None) or self.e.search.name,
                status=status, error=error, result_count=len(results),
                results=[r.as_dict() for r in results], duration_ms=int((self.e.monotonic() - started) * 1000),
            )
            self._checkpoint()
            if status in ("OK", "EMPTY"):
                if planned.question_id:
                    self.store.execute("UPDATE open_question SET attempts = attempts + 1 WHERE id = ?", (planned.question_id,))
                return results
            self.c["search_failures"] += 1
            self._say(f"search '{planned.query}': {status} ({error})")
            wait = min(retry_after, 30.0)
            if status != "RATE_LIMITED" or attempt == 1 or self.budget.remaining_seconds() < wait + self.budget.wrapup_reserve_seconds + 30:
                return []
            self.e.sleep(wait)
        return []

    def _triage(self, planned: PlannedQuery, candidates: list[dict[str, Any]], quota: int) -> list[dict[str, Any]]:
        listing = [
            {"index": i + 1, "title": c["title"] or "", "snippet": c["snippet"] or "", "domain": c["domain"]}
            for i, c in enumerate(candidates)
        ]
        triage = self._llm("triage", prompts.triage(
            objective=self.objective, spec=self.spec, query=planned.query, candidates=listing,
            focus=self.spec.key_questions), Triage, 50 + 36 * len(listing))
        ratings = {r.index: r for r in (triage.ratings if triage else [])}
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for i, cand in enumerate(candidates):
            rating = ratings.get(i + 1)
            if rating is None:
                # No judgement available: fall back to the provider's ranking for the first few results.
                score, note, out_of_scope = (2 if cand["rank"] <= 3 else 1), "not triaged; ranked by search position", False
            else:
                score, note, out_of_scope = rating.score, rating.reason, rating.out_of_scope
            if out_of_scope:
                self.c["out_of_scope_candidates"] += 1
                self.out_of_scope.append({"title": cand["title"], "url": cand["canonical_url"], "reason": note, "score": score})
                self.store.update_source(cand["id"], status=SourceStatus.IRRELEVANT, relevance=score,
                                         triage_note=f"out of approved scope: {note}"[:300])
                continue
            if score == 0:
                self.store.update_source(cand["id"], status=SourceStatus.IRRELEVANT, relevance=0, triage_note=note)
                continue
            self.store.update_source(cand["id"], relevance=score, triage_note=note)
            if score >= 2:
                scored.append((score, cand["rank"], cand))
        scored.sort(key=lambda t: (-t[0], t[1]))
        picks = []
        for _, _, cand in scored:
            if len(picks) >= quota:
                break
            if self._domain_full(cand["domain"], pending=sum(1 for p in picks if p["domain"] == cand["domain"])):
                continue
            picks.append(cand)
        self._event("triage", {"query": planned.query, "rated": len(ratings), "selected": [p["canonical_url"] for p in picks],
                               "left_for_later": len(scored) - len(picks)})
        return picks

    # ------------------------------------------------------- read one page

    def _process_source(self, source: dict[str, Any], recheck: bool = False) -> None:
        if not self.budget.can_fetch():
            return
        self.budget.note_fetch()
        self.domain_count[source["domain"]] = self.domain_count.get(source["domain"], 0) + 1
        now = iso(self.e.clock())
        result: FetchResult = self.e.fetcher.fetch(source["url"])
        self._event("fetch", {"source_id": source["id"], "url": source["url"], "outcome": result.outcome,
                              "http_status": result.http_status, "chars": len(result.text), "ms": result.duration_ms,
                              "recheck": recheck})
        self._checkpoint()
        if not result.ok:
            self._fetch_failed(source, result, recheck, now)
            return

        flags = set(source["flags"] or []) - {"transient_failure"} | set(result.flags)
        patterns = scan_for_injection(result.text)
        if patterns:
            flags.add("instruction_like_text")
            self.c["sources_flagged_instruction_like"] += 1
            self._event("instruction_like_text", {"source_id": source["id"], "patterns": patterns})
        content_hash = sha256_hex(normalize_text(result.text))
        previous_hash = source["content_hash"]
        snapshot_id, new_version = self.store.add_snapshot(source["id"], self.run_id, content_hash, result.text)
        update = dict(
            status=SourceStatus.FETCHED, final_url=result.final_url, http_status=result.http_status,
            content_type=result.content_type, title=result.title or source["title"], published_at=result.published_at,
            content_hash=content_hash, text_chars=len(result.text), last_fetched_at=now,
            fetch_count=source["fetch_count"] + 1, flags=sorted(flags), last_error=None,
        )
        if previous_hash != content_hash:
            update["last_changed_at"] = now
        self.store.update_source(source["id"], **update)
        source = self.store.get_source(source["id"]) or source

        if previous_hash == content_hash:
            self._confirm_unchanged(source, now)
            return
        duplicate = self.store.one(
            "SELECT id FROM source WHERE watch_id = ? AND content_hash = ? AND id != ? AND status = ?",
            (self.watch_id, content_hash, source["id"], SourceStatus.FETCHED))
        if duplicate and previous_hash is None:
            self.c["duplicate_pages_skipped"] += 1
            self.store.update_source(source["id"], flags=sorted(flags | {f"same_content_as:{duplicate['id']}"}))
            self._source_result(source, "duplicate_content", f"same text as source {duplicate['id']}")
            self._say(f"  identical content to an earlier source; skipped ({source['canonical_url']})")
            return

        self.c["pages_read"] += 1
        pre = normalize_with_map(result.text)
        if previous_hash is not None:
            self.c["pages_changed"] += 1
            self._retire_vanished_claims(source, result.text, pre, now)
        self._extract_and_integrate(source, snapshot_id, result, pre, patterns)

    def _fetch_failed(self, source: dict[str, Any], result: FetchResult, recheck: bool, now: str) -> None:
        self.c["fetch_failures"] += 1
        transient = result.outcome in TRANSIENT_OUTCOMES or (result.http_status or 0) in (429, 500, 502, 503, 504)
        self._say(f"  fetch failed: {result.outcome} {result.error or ''} ({source['canonical_url']})")
        self._source_result(source, "blocked" if result.blocked else "failed", f"{result.outcome}: {result.error}",
                            result.http_status)
        if source["status"] == SourceStatus.FETCHED:
            # A page we relied on can no longer be read. Only a definite "gone" makes findings stale.
            gone = (result.http_status or 0) in (404, 410)
            self.store.update_source(
                source["id"], last_error=f"{result.outcome}: {result.error}", fetch_count=source["fetch_count"] + 1,
                **({"status": SourceStatus.FAILED, "http_status": result.http_status,
                    "flags": sorted(set(source["flags"] or []) | {"gone"})} if gone else {}))
            if gone:
                self._mark_source_gone(source, f"source returned HTTP {result.http_status} on {now[:10]}")
            return
        flags = set(source["flags"] or [])
        flags = (flags | {"transient_failure"}) if transient else (flags - {"transient_failure"})
        self.store.update_source(
            source["id"], status=SourceStatus.BLOCKED if result.blocked else SourceStatus.FAILED,
            http_status=result.http_status, final_url=result.final_url, last_error=f"{result.outcome}: {result.error}",
            fetch_count=source["fetch_count"] + 1, flags=sorted(flags),
        )

    def _confirm_unchanged(self, source: dict[str, Any], now: str) -> None:
        self.c["pages_unchanged"] += 1
        touched = set()
        for claim in self.store.list_claims(source_id=source["id"], status=ClaimStatus.ACCEPTED):
            self.store.update_claim(claim["id"], last_confirmed_at=now)
            if claim["finding_id"]:
                touched.add(claim["finding_id"])
        for finding_id in touched:
            self.store.update_finding(finding_id, last_confirmed_at=now)
            self.store.record_change(self.run_id, self.watch_id, finding_id, ChangeType.UNCHANGED,
                                     {"reconfirmed_from": [source["id"]]})
        self._source_result(source, "unchanged", f"{len(touched)} finding(s) reconfirmed")
        self._say(f"  unchanged since last visit; {len(touched)} finding(s) reconfirmed ({source['canonical_url']})")

    def _retire_vanished_claims(self, source: dict[str, Any], text: str, pre: tuple[str, list[int]], now: str) -> None:
        """The page changed: claims whose quotes are gone are superseded; the rest are reconfirmed."""
        affected: dict[str, list[str]] = {}
        for claim in self.store.list_claims(source_id=source["id"], status=ClaimStatus.ACCEPTED):
            if locate_quote(claim["quote"], text, pre).status is Verification.QUOTE_NOT_FOUND:
                self.store.update_claim(claim["id"], status=ClaimStatus.SUPERSEDED,
                                        note=f"quote no longer on the page as of {now[:10]}")
                if claim["finding_id"]:
                    affected.setdefault(claim["finding_id"], []).append(claim["id"])
            else:
                self.store.update_claim(claim["id"], last_confirmed_at=now)
        for finding_id, claim_ids in affected.items():
            remaining = self.store.list_claims(finding_id=finding_id, status=ClaimStatus.ACCEPTED)
            view = self.findings.get(finding_id)
            if view is not None:
                view.claims = remaining
            if remaining:
                self.store.update_finding(finding_id, last_changed_run_id=self.run_id, last_changed_at=now)
                self.store.record_change(self.run_id, self.watch_id, finding_id, ChangeType.UPDATED,
                                         {"superseded_claim_ids": claim_ids, "note": "source page changed"})
            else:
                self.store.update_finding(finding_id, status="STALE", last_changed_run_id=self.run_id, last_changed_at=now)
                self.store.record_change(self.run_id, self.watch_id, finding_id, ChangeType.STALE,
                                         {"superseded_claim_ids": claim_ids, "note": "source page no longer states this"})

    def _mark_source_gone(self, source: dict[str, Any], note: str) -> None:
        by_finding: dict[str, list[str]] = {}
        for claim in self.store.list_claims(source_id=source["id"], status=ClaimStatus.ACCEPTED):
            if claim["finding_id"]:
                by_finding.setdefault(claim["finding_id"], []).append(claim["id"])
        now = iso(self.e.clock())
        for finding_id in by_finding:
            others = [c for c in self.store.list_claims(finding_id=finding_id, status=ClaimStatus.ACCEPTED)
                      if c["source_id"] != source["id"]]
            already_stale = (self.store.get_finding(finding_id) or {}).get("status") == "STALE"
            if not others and not already_stale:
                self.store.update_finding(finding_id, status="STALE", last_changed_run_id=self.run_id, last_changed_at=now)
                self.store.record_change(self.run_id, self.watch_id, finding_id, ChangeType.STALE, {"note": note})

    def _extract_and_integrate(self, source: dict[str, Any], snapshot_id: int, result: FetchResult,
                               pre: tuple[str, list[int]], patterns: list[str]) -> None:
        focus_questions = [q["text"] for q in (self.memory.questions_for_planner() if self.memory else [])][:5]
        focus = self.focus | token_set(source.get("discovered_via") or "")
        shown = result.text
        if patterns:
            shown, _ = redact_injection(shown)
        shown = select_passages(shown, focus, self.settings.page_chars_per_call)
        extraction: Extraction | None = self._llm("extract", prompts.extract(
            objective=self.objective, spec=self.spec, today=self.today, focus_questions=focus_questions,
            title=result.title or source.get("title") or "", domain=source["domain"], published=result.published_at,
            text=shown, max_items=MAX_ITEMS_PER_PAGE, max_claims=MAX_CLAIMS_PER_ITEM), Extraction, 900)
        if extraction is None:
            self.store.update_source(source["id"], flags=sorted(set(source["flags"] or []) | {"extraction_failed"}))
            self._source_result(source, "read", "extraction did not run (budget) or returned unusable output")
            return
        self.store.update_source(source["id"], page_relevance=extraction.relevance)
        if extraction.relevance <= 0 or not extraction.items:
            self._source_result(source, "not_relevant", "the model found nothing relevant to the objective")
            self._say(f"  nothing relevant on the page ({source['canonical_url']})")
            return

        staged: list[tuple[ExtractedItem, list[_Accepted]]] = []
        for item in extraction.items[:MAX_ITEMS_PER_PAGE]:
            kept: list[_Accepted] = []
            for claim in item.claims[:MAX_CLAIMS_PER_ITEM]:
                self.c["claims_extracted"] += 1
                check = check_claim(claim.text, claim.quote, result.text, pre,
                                    context=f"{result.title or ''} {source['domain']} {source['canonical_url']}")
                if not check.accepted:
                    self._store_rejected(source, snapshot_id, claim.text, claim.quote, check.reason or "rejected", check)
                    continue
                kept.append(_Accepted(" ".join(claim.text.split()), check))
            if kept:
                staged.append((item, kept))

        self._support_check(source, snapshot_id, staged)
        ready = [(item, [k for k in kept if not k.dropped]) for item, kept in staged]
        ready = [(item, kept) for item, kept in ready if kept]
        targets = self._match_items(source, ready)
        accepted_total = 0
        for (item, kept), target in zip(ready, targets):
            accepted_total += len(kept)
            self._integrate(source, snapshot_id, item, kept, extraction.relevance, target)
        self._source_result(source, "read")
        self._say(f"  read {source['canonical_url']}: {accepted_total} claim(s) kept")

    def _store_rejected(self, source: dict[str, Any], snapshot_id: int, text: str, quote: str, reason: str,
                        check: ClaimCheck | None = None, support: Support = Support.UNCHECKED) -> None:
        self._reject(reason)
        if not text.strip():
            return
        note = reason + (": " + "; ".join(check.notes) if check and check.notes else "")
        self.store.add_claim(
            watch_id=self.watch_id, finding_id=None, source_id=source["id"], snapshot_id=snapshot_id, run_id=self.run_id,
            text=text[:500], quote=(quote or "")[:700], quote_offset=None,
            verification=(check.verification if check else Verification.QUOTE_NOT_FOUND), support=support,
            status=ClaimStatus.REJECTED, note=note[:300],
        )

    def _support_check(self, source: dict[str, Any], snapshot_id: int,
                       staged: list[tuple[ExtractedItem, list[_Accepted]]]) -> None:
        flat = [k for _, kept in staged for k in kept]
        if not flat or not self.settings.support_check:
            return
        pairs = [{"index": i + 1, "claim": k.text, "quote": k.check.quote} for i, k in enumerate(flat)]
        verdicts = self._llm("support", prompts.support(pairs, self.objective, self.spec), SupportCheck,
                             30 + 22 * len(pairs))
        by_index = {v.index: v for v in (verdicts.verdicts if verdicts else [])}
        for i, kept in enumerate(flat):
            verdict = by_index.get(i + 1)
            if verdict is not None and verdict.verdict == "unsupported":
                kept.support, kept.dropped = Support.UNSUPPORTED, True
                self._store_rejected(source, snapshot_id, kept.text, kept.check.quote, "unsupported_by_quote",
                                     kept.check, Support.UNSUPPORTED)
                continue
            if verdict is not None and verdict.bearing != "direct":
                kept.dropped = True
                self._store_rejected(source, snapshot_id, kept.text, kept.check.quote, "not_relevant_to_objective",
                                     kept.check)
                continue
            if verdict is not None:
                kept.support = Support.SUPPORTED if verdict.verdict == "supported" else Support.PARTIAL
            # Deterministic downgrades: whatever the model judged, a claim that carries names or numbers its
            # quote lacks, or says much more than its quote, is at best partly supported.
            if kept.check.details_outside_quote:
                kept.support = Support.PARTIAL
            elif kept.support is not Support.PARTIAL and claim_coverage(kept.text, kept.check.quote) < 0.5:
                kept.support = Support.PARTIAL if verdict is not None else kept.support
                kept.check.notes.append("claim includes detail beyond the quoted passage")

    # ------------------------------------------------------------ integrate

    @staticmethod
    def _item_identity(source: dict[str, Any], item: ExtractedItem, kept: list[_Accepted]) -> tuple[str, str, str]:
        title = item.title or item.subject or (source.get("title") or "Untitled finding")
        subject = item.subject or item.title or title
        return title, subject, " ".join([title, subject, *[k.text for k in kept]])

    def _integrate(self, source: dict[str, Any], snapshot_id: int, item: ExtractedItem, kept: list[_Accepted],
                   relevance: int, target: str | None) -> None:
        title, subject, _ = self._item_identity(source, item, kept)
        now = iso(self.e.clock())
        domain = registrable_domain(source["domain"])
        uncertainty = item.uncertainty

        if target is None:
            first_id = self.store.claim_id(self.watch_id, source["id"], kept[0].text)
            finding_id = self.store.create_finding(self.watch_id, self.run_id, title, subject, relevance, uncertainty, first_id)
            view = self.findings.setdefault(finding_id, FindingView(finding_id, title, subject, "ACTIVE"))
            added = self._add_claims(finding_id, view, source, snapshot_id, kept)
            view.domains.add(domain)
            if added:
                self.store.record_change(self.run_id, self.watch_id, finding_id, ChangeType.NEW, {"new_claim_ids": added})
            return

        view = self.findings[target]
        known_texts = [c["text"] for c in view.claims]
        known_ids = {c["id"] for c in view.claims}
        novel: list[_Accepted] = []
        repeats: list[_Accepted] = []
        undecided: list[_Accepted] = []
        for k in kept:
            if self.store.claim_id(self.watch_id, source["id"], k.text) in known_ids:
                self.c["duplicate_claims_suppressed"] += 1
                continue
            best = max((text_similarity(k.text, t) for t in known_texts), default=0.0)
            (repeats if best >= SAME_CLAIM else undecided if best >= MAYBE_CLAIM else novel).append(k)
        if undecided:
            verdict = self._llm("novelty", prompts.novelty(
                title=view.title, known_claims=known_texts[:12],
                new_claims=[{"index": i + 1, "text": k.text} for i, k in enumerate(undecided)]), NoveltyCheck,
                30 + 14 * len(undecided))
            decided = {v.index: v.adds_information for v in (verdict.verdicts if verdict else [])}
            for i, k in enumerate(undecided):
                # Without a judgement, lean on the lexical score: closer to a repeat than to something new.
                adds = decided.get(i + 1, text_similarity(k.text, " ".join(known_texts)) < 0.6)
                (novel if adds else repeats).append(k)

        same_source_before = any(c["source_id"] == source["id"] for c in view.claims)
        new_domain = domain not in view.domains
        added_novel = self._add_claims(target, view, source, snapshot_id, novel)
        added_repeat: list[str] = []
        if repeats:
            if same_source_before:
                self.c["duplicate_claims_suppressed"] += len(repeats)
            else:
                # The same fact from another source is kept once as corroborating evidence.
                added_repeat = self._add_claims(target, view, source, snapshot_id, repeats[:1])
                self.c["duplicate_claims_suppressed"] += len(repeats) - len(added_repeat)
        view.domains.add(domain)
        self.store.update_finding(target, last_confirmed_at=now, status="ACTIVE")
        if added_novel:
            self.store.update_finding(target, last_changed_run_id=self.run_id, last_changed_at=now,
                                      **({"uncertainty": uncertainty} if uncertainty and not self.store.get_finding(target)["uncertainty"] else {}))
            self.store.record_change(self.run_id, self.watch_id, target, ChangeType.UPDATED, {"new_claim_ids": added_novel})
        elif added_repeat and new_domain:
            self.store.record_change(self.run_id, self.watch_id, target, ChangeType.CORROBORATED,
                                     {"corroborating_claim_ids": added_repeat, "new_domain": domain})
        else:
            self.store.record_change(self.run_id, self.watch_id, target, ChangeType.UNCHANGED, {"seen_again_in": [source["id"]]})

    def _match_items(self, source: dict[str, Any], ready: list[tuple[ExtractedItem, list[_Accepted]]]) -> list[str | None]:
        """Decide, for each item from one page, which existing finding it belongs with (or none).

        Clear lexical matches are decided here. Everything in the uncertain band goes to the
        model in a single call per page, because wording differs too much between sources for
        word overlap alone to recognise the same subject.
        """
        targets: list[str | None] = [None] * len(ready)
        existing = [(f.id, f"{f.title} {f.subject}", f.text) for f in self.findings.values() if f.claims]
        if not existing or not ready:
            return targets
        texts: dict[int, str] = {}
        shortlists: dict[int, list] = {}
        pool: dict[str, int] = {}
        known_claims = [(f.id, c["text"]) for f in self.findings.values() for c in f.claims]
        for i, (item, kept) in enumerate(ready):
            title, subject, text = self._item_identity(source, item, kept)
            # A claim that restates one already on record places the item with that finding.
            restated: dict[str, float] = {}
            for k in kept:
                for finding_id, known in known_claims:
                    similarity = text_similarity(k.text, known)
                    if similarity > restated.get(finding_id, 0.0):
                        restated[finding_id] = similarity
            if restated:
                finding_id, similarity = max(restated.items(), key=lambda pair: pair[1])
                if similarity >= SAME_CLAIM:
                    targets[i] = finding_id
                    continue
            ranked = rank_candidates(text, f"{title} {subject}", existing, limit=len(existing))
            for candidate in ranked:
                candidate.score = max(candidate.score, 0.75 * restated.get(candidate.key, 0.0))
            ranked.sort(key=lambda c: -c.score)
            if ranked and ranked[0].score >= SAME_FINDING:
                targets[i] = ranked[0].key
                continue
            shortlist = [c for c in ranked if c.score >= MAYBE_FINDING][:3]
            if shortlist:
                texts[i], shortlists[i] = text, shortlist
                for candidate in shortlist:
                    pool.setdefault(candidate.key, len(pool) + 1)
        if not shortlists:
            return targets
        decision = self._llm("match", prompts.match(
            items=[{"index": i + 1, "text": text} for i, text in texts.items()],
            candidates=[{"index": number, "text": self.findings[key].text} for key, number in pool.items()]),
            MatchBatch, 30 + 16 * len(shortlists))
        by_number = {number: key for key, number in pool.items()}
        chosen = {m.item: m.same_as for m in (decision.matches if decision else [])}
        for i, shortlist in shortlists.items():
            if decision is None:
                # No judgement available: merge only on a strong lexical score, otherwise file as new.
                targets[i] = shortlist[0].key if shortlist[0].score >= 0.5 else None
            else:
                targets[i] = by_number.get(chosen.get(i + 1))  # type: ignore[arg-type]
        return targets

    def _add_claims(self, finding_id: str, view: FindingView, source: dict[str, Any], snapshot_id: int,
                    kept: list[_Accepted]) -> list[str]:
        added = []
        for k in kept:
            claim_id, created = self.store.add_claim(
                watch_id=self.watch_id, finding_id=finding_id, source_id=source["id"], snapshot_id=snapshot_id,
                run_id=self.run_id, text=k.text, quote=k.check.quote, quote_offset=k.check.offset,
                verification=k.check.verification, support=k.support, status=ClaimStatus.ACCEPTED,
                note="; ".join(k.check.notes)[:300] or None,
            )
            if not created:
                existing = self.store.get_claim(claim_id)
                if existing and existing["status"] != ClaimStatus.ACCEPTED:
                    # Previously rejected or superseded text that now verifies against the current page.
                    self.store.update_claim(
                        claim_id, finding_id=finding_id, snapshot_id=snapshot_id, run_id=self.run_id, quote=k.check.quote,
                        quote_offset=k.check.offset, verification=k.check.verification, support=k.support,
                        status=ClaimStatus.ACCEPTED, note="; ".join(k.check.notes)[:300] or None,
                        last_confirmed_at=iso(self.e.clock()))
                    created = True
            if created:
                self.c["claims_accepted"] += 1
                added.append(claim_id)
                row = self.store.get_claim(claim_id)
                if row:
                    view.claims.append(row)
            else:
                self.c["duplicate_claims_suppressed"] += 1
        return added

    # -------------------------------------------------------------- reflect

    def _reflect(self) -> None:
        changes = self.store.list_changes(self.run_id)
        searches = self.store.list_search_attempts(self.watch_id, self.run_id)
        if not changes and not searches and self.budget.fetched_pages == 0:
            self._event("reflection", {"summary": "", "skipped": "nothing was searched or read"})
            return
        shown = []
        for change in sorted(changes, key=lambda c: -ChangeType(c["change_type"]).rank):
            view = self.findings.get(change["finding_id"])
            if view is None or change["change_type"] == ChangeType.UNCHANGED:
                continue
            new_ids = set((change["detail"] or {}).get("new_claim_ids", []))
            claims = [c for c in view.claims if c["id"] in new_ids] or view.claims
            shown.append({"id": view.id, "change": change["change_type"], "title": view.title,
                          "text": " ".join(c["text"] for c in claims[:4])})
        open_questions = [{"id": q["id"], "text": q["text"]}
                          for q in self.store.list_questions(self.watch_id, QuestionStatus.OPEN)][:10]
        counts = {
            "new_findings": sum(1 for c in changes if c["change_type"] == ChangeType.NEW),
            "updated_findings": sum(1 for c in changes if c["change_type"] == ChangeType.UPDATED),
            "pages_read": self.c["pages_read"], "claims_kept": self.c["claims_accepted"],
            "claims_rejected": self.c["claims_rejected"],
        }
        reflection: Reflection | None = self._llm("reflect", prompts.reflect(
            objective=self.objective, spec=self.spec, today=self.today,
            session_number=self.memory.session_number if self.memory else 1, findings=shown[:14],
            open_questions=open_questions,
            searches=[{"query": s["query"], "outcome": f"{s['status'].lower()}, {s['result_count']} results, {s['new_url_count']} new"}
                      for s in searches],
            counts=counts, allow_guidance=self.allow_guidance), Reflection, 800, wrapup=True)
        if reflection is None:
            self._event("reflection", {"summary": "", "skipped": "model call unavailable or over budget"})
            self.notes.append("The end-of-session review could not be run, so open questions were not re-assessed.")
            return
        self._apply_reflection(reflection, {q["id"] for q in open_questions})

    def _apply_reflection(self, reflection: Reflection, open_ids: set[str]) -> None:
        now = iso(self.e.clock())
        resolved = []
        for entry in reflection.resolved:
            if entry.question_id not in open_ids:
                continue
            finding_ids = [fid for fid in entry.finding_ids if fid in self.findings and self.findings[fid].claims]
            if not finding_ids or not entry.answer:
                continue  # an answer without verified evidence does not close a question
            self.store.update_question(
                entry.question_id, status=QuestionStatus.RESOLVED, resolution=entry.answer, resolved_by=finding_ids,
                resolved_run_id=self.run_id, resolved_at=now)
            resolved.append(entry.question_id)
        existing = [q["text"] for q in self.store.list_questions(self.watch_id)]
        open_count = len(self.store.list_questions(self.watch_id, QuestionStatus.OPEN))
        opened = []
        for text in reflection.new_questions:
            if open_count >= MAX_OPEN_QUESTIONS:
                break
            if scan_for_injection(text) or any(text_similarity(text, old) >= 0.7 for old in existing):
                continue
            question_id, created = self.store.add_question(self.watch_id, text, "REFLECTION", self.run_id)
            if created:
                opened.append(question_id)
                existing.append(text)
                open_count += 1
        tried = (self.memory.tried_queries if self.memory else []) + [
            s["query"] for s in self.store.list_search_attempts(self.watch_id, self.run_id)]
        directions = []
        for direction in reflection.next_directions[:4]:
            query = " ".join(direction.query.replace('"', " ").split()[:12])
            if len(tokenize(query)) < 2 or scan_for_injection(query) or is_repeat_query(query, tried + [d["query"] for d in directions]):
                continue
            if self.store.add_lead(self.watch_id, query, direction.rationale, self.run_id):
                directions.append({"query": query, "rationale": direction.rationale})
        guidance = reflection.guidance
        if guidance is not None and guidance.needed and guidance.question and self.allow_guidance \
                and not scan_for_injection(guidance.question + " " + " ".join(guidance.options)):
            self.guidance = guidance
        self._event("reflection", {
            "summary": reflection.summary, "resolved": resolved, "opened": opened, "next_directions": directions,
            "guidance": self.guidance.model_dump() if self.guidance else None,
        })

    # ------------------------------------------------------------- finalize

    def _finalize(self, status: RunStatus, stop_reason: str, error: str | None) -> SessionOutcome:
        from forage.report.build import write_report

        changes = self.store.list_changes(self.run_id)
        kinds = [c["change_type"] for c in changes]
        resolved = self.store.scalar(
            "SELECT COUNT(*) FROM open_question WHERE watch_id = ? AND resolved_run_id = ?", (self.watch_id, self.run_id))
        meaningful = any(k in (ChangeType.NEW, ChangeType.UPDATED, ChangeType.STALE) for k in kinds) or bool(resolved)
        metrics = self.budget.snapshot() | self.c | {
            "budget_exhausted": self.work_limits or [],
            "claims_rejected_by_reason": dict(sorted(self.rejections.items())),
            "findings_new": kinds.count(ChangeType.NEW), "findings_updated": kinds.count(ChangeType.UPDATED),
            "findings_corroborated": kinds.count(ChangeType.CORROBORATED),
            "findings_unchanged": kinds.count(ChangeType.UNCHANGED), "findings_stale": kinds.count(ChangeType.STALE),
            "questions_resolved": int(resolved or 0),
            "out_of_scope_examples": self.out_of_scope[:6],
            "recall": self.recall,
            "notes": self.notes,
            "guidance_proposed": self.guidance.model_dump() if self.guidance else None,
        }
        self.store.update(
            "session_run", {"id": self.run_id},
            {"status": status, "finished_at": iso(self.e.clock()), "metrics": metrics, "stop_reason": stop_reason,
             "error": error, "has_meaningful_update": meaningful},
        )
        self._event("session_finished", {"status": status, "stop_reason": stop_reason, "meaningful": meaningful})
        try:
            write_report(self.store, self.settings, self.run_id)
        except Exception as exc:  # never lose the run record because rendering failed
            self._event("report_error", {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:]})
            error = error or f"report generation failed: {type(exc).__name__}: {exc}"
            self.store.update("session_run", {"id": self.run_id}, {"error": error})
        return SessionOutcome(self.run_id, self.watch_id, status, meaningful, self.guidance, metrics, error)
