"""A session whose position and tallies live in a small dict.

`_Session` keeps its working state in Python attributes for the life of one call
to `run()`. To let any step run in a fresh process, `StepSession` can be rebuilt
from, and exported to, a JSON-safe dict. Research data is never copied into that
dict: it holds ids, counters and short strings, and everything else is re-read from
the store.

It also makes the three steps with external or multi-statement effects safe to run
again after an attempt that died part-way, which a step-checkpointing runtime
requires of its nodes.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from forage.engine.memory import build_memory
from forage.engine.safety import scan_for_injection
from forage.engine.schemas import GuidanceProposal, PlannedQuery
from forage.engine.session import ResearchEngine, _Page, _Session
from forage.search.base import SearchResult
from forage.types import SourceStatus
from forage.util import normalize_text

# Fields of the session-start recall that the planner reads. They are kept until the plan step has run,
# because the linear engine plans from this snapshot rather than from a later re-read of the store.
_MEMORY_VIEW = ("session_number", "prior_sessions", "last_session_at", "open_questions", "tried_queries",
                "recent_queries", "rerunnable", "tried_for_planner", "leads")


def _compact(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    return [{k: row.get(k) for k in keys} for row in rows]


class StepSession(_Session):
    def __init__(self, engine: ResearchEngine, state: dict[str, Any], should_cancel: Callable[[], bool] | None = None):
        super().__init__(engine, state["run_id"], should_cancel, True)
        self._pinned_focus: list[str] | None = state.get("focus_questions")
        self._restore(state)

    # ------------------------------------------------------- restore / export

    def _restore(self, state: dict[str, Any]) -> None:
        self.c.update(state.get("counters") or {})
        self.rejections = dict(state.get("rejections") or {})
        self.out_of_scope = list(state.get("out_of_scope") or [])
        self.domain_count = dict(state.get("domain_count") or {})
        self.notes = list(state.get("notes") or [])
        self.recall = dict(state.get("recall") or {})
        self.work_limits = state.get("work_limits")
        self.guidance = GuidanceProposal.model_validate(state["guidance"]) if state.get("guidance") else None

        tally = state.get("budget_tally") or {}
        budget = self.budget
        budget.search_requests = int(tally.get("search_requests", 0))
        budget.fetched_pages = int(tally.get("fetched_pages", 0))
        budget.llm_calls = int(tally.get("llm_calls", 0))
        budget.llm_prompt_tokens = int(tally.get("llm_prompt_tokens", 0))
        budget.llm_output_tokens = int(tally.get("llm_output_tokens", 0))
        budget.llm_ms = int(tally.get("llm_ms", 0))
        budget.exhausted = list(state.get("budget_flags") or [])
        budget._started = self.e.monotonic() - float(state.get("seconds_used") or 0.0)
        self._floor_budget_from_store()

        if state.get("recalled"):
            self._restore_memory(state)

    def _floor_budget_from_store(self) -> None:
        """Spending recorded in the store counts even if the step that did it never finished.

        The dict is written only when a step completes. A step that died after calling the model or
        the search provider left rows behind; reading them back keeps the hard budgets hard.
        """
        budget, run = self.budget, (self.run_id,)
        budget.search_requests = max(budget.search_requests, int(self.store.scalar(
            "SELECT COUNT(*) FROM search_attempt WHERE run_id = ?", run) or 0))
        budget.fetched_pages = max(budget.fetched_pages, int(self.store.scalar(
            "SELECT COUNT(*) FROM run_event WHERE run_id = ? AND kind = 'fetch'", run) or 0))
        row = self.store.one(
            "SELECT COALESCE(SUM(prompt_tokens), 0) AS p, COALESCE(SUM(output_tokens), 0) AS o, COALESCE(SUM(duration_ms), 0) AS ms "
            "FROM llm_call WHERE run_id = ?", run) or {"p": 0, "o": 0, "ms": 0}
        budget.llm_prompt_tokens = max(budget.llm_prompt_tokens, int(row["p"]))
        budget.llm_output_tokens = max(budget.llm_output_tokens, int(row["o"]))
        budget.llm_ms = max(budget.llm_ms, int(row["ms"]))

    def _restore_memory(self, state: dict[str, Any]) -> None:
        for hint in self.spec.source_hints:
            self._register_hint(hint)  # idempotent; rebuilds the set of user-named sites
        memory = build_memory(self.store, self.watch_id, self.spec, self.e.clock(), self.settings.requery_after_hours)
        for name, value in (state.get("memory_view") or {}).items():
            setattr(memory, name, value)
        memory.backlog = [row for row in map(self.store.get_source, state.get("lead_ids") or []) if row]
        memory.recheck = [row for row in map(self.store.get_source, state.get("recheck_ids") or []) if row]
        self._attach_memory(memory)

    def memory_view(self) -> dict[str, Any]:
        assert self.memory is not None
        view = {name: getattr(self.memory, name) for name in _MEMORY_VIEW}
        view["open_questions"] = _compact(view["open_questions"], ("id", "text", "attempts", "created_at"))
        view["leads"] = _compact(view["leads"], ("id", "text"))
        return view

    def export(self) -> dict[str, Any]:
        budget = self.budget
        out: dict[str, Any] = {
            "counters": dict(self.c), "rejections": dict(self.rejections), "out_of_scope": list(self.out_of_scope),
            "domain_count": dict(self.domain_count), "notes": list(self.notes), "recall": dict(self.recall),
            "work_limits": self.work_limits, "guidance": self.guidance.model_dump() if self.guidance else None,
            "budget_flags": list(budget.exhausted), "seconds_used": budget.elapsed(),
            "budget_tally": {
                "search_requests": budget.search_requests, "fetched_pages": budget.fetched_pages,
                "llm_calls": budget.llm_calls, "llm_prompt_tokens": budget.llm_prompt_tokens,
                "llm_output_tokens": budget.llm_output_tokens, "llm_ms": budget.llm_ms,
            },
        }
        if self.memory is not None:
            out["lead_ids"] = [row["id"] for row in self.memory.backlog]
        return out

    # ------------------------------------------------------------ page refs

    def page_from_ref(self, ref: dict[str, Any]) -> _Page:
        """Rebuild a page from its reference. The text comes from the snapshot table, not from the dict."""
        source = self.store.get_source(ref["source_id"])
        snapshot = self.store.get_snapshot(ref["snapshot_id"])
        assert source is not None and snapshot is not None, "page reference points at missing rows"
        return _Page(source, snapshot["id"], snapshot["text"], ref.get("title"), ref.get("published_at"),
                     list(ref.get("patterns") or []))

    # ----------------------------------------------------------- overrides

    def _focus_questions(self) -> list[str]:
        return list(self._pinned_focus) if self._pinned_focus is not None else super()._focus_questions()

    def _run_search(self, planned: PlannedQuery) -> list:
        """Reuse a search this run already completed instead of sending it to the provider again."""
        prior = self.store.one(
            "SELECT id, results FROM search_attempt WHERE run_id = ? AND query_norm = ? AND status IN ('OK', 'EMPTY') "
            "ORDER BY id DESC LIMIT 1", (self.run_id, normalize_text(planned.query)))
        if prior is None:
            return super()._run_search(planned)
        self._last_attempt_id = int(prior["id"])
        self._event("step_replayed", {"step": "search", "query": planned.query})
        return [SearchResult(**row) for row in json.loads(prior["results"])]

    def _fetch_source(self, source: dict[str, Any], recheck: bool = False) -> _Page | None:
        """Pick up a page this run already retrieved and stored but never finished reading.

        Without this, a second attempt at the step would find the stored content hash, conclude the
        page is "unchanged since the last visit", and never extract it.
        """
        pending = self._stored_but_unread(source)
        if pending is None:
            return super()._fetch_source(source, recheck)
        self.domain_count[source["domain"]] = self.domain_count.get(source["domain"], 0) + 1
        self.c["pages_read"] += 1
        self._event("step_replayed", {"step": "fetch", "source_id": source["id"]})
        return pending

    def _stored_but_unread(self, source: dict[str, Any]) -> _Page | None:
        snapshot = self.store.latest_snapshot(source["id"])
        if snapshot is None or snapshot["run_id"] != self.run_id:
            return None
        if source["status"] != SourceStatus.FETCHED or source["content_hash"] != snapshot["content_hash"]:
            return None  # the earlier attempt died before recording the fetch; do it again
        finished = self.store.scalar(
            "SELECT COUNT(*) FROM run_event WHERE run_id = ? AND kind = 'source_result' "
            "AND json_extract(data, '$.source_id') = ?", (self.run_id, source["id"]))
        if finished:
            return None
        return _Page(source, snapshot["id"], snapshot["text"], source.get("title"), source.get("published_at"),
                     scan_for_injection(snapshot["text"]))
