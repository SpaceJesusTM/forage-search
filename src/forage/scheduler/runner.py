"""Run manager: admits one bounded session at a time and records its outcome.

Admission is transactional and backed by a unique index, so two processes cannot
both start a session. Checkpoint policy evaluation happens here, after the engine
returns, so every policy sees the same engine behaviour.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Callable

from forage.checkpoint.base import CheckpointContext, CheckpointDecision, evaluate, get_policy
from forage.config import Settings
from forage.controller.reviews import ReviewController
from forage.controller.watches import WatchController
from forage.engine.session import ResearchEngine, SessionOutcome
from forage.report.build import write_report
from forage.storage.store import Store
from forage.types import Budget, RunStatus, WatchStatus
from forage.util import random_id, utcnow


class RunRefused(Exception):
    """The run was not admitted. Nothing was started."""


class RunManager:
    def __init__(self, store: Store, settings: Settings, engine: ResearchEngine, *, clock: Callable = utcnow):
        self.store = store
        self.settings = settings
        self.engine = engine
        self.clock = clock
        self.watches = WatchController(store)
        self.reviews = ReviewController(store)

    # -- recovery ---------------------------------------------------------

    @property
    def stale_after_seconds(self) -> int:
        return int(max(600, self.settings.llm_timeout_seconds * 2 + 120))

    def recover(self, force: bool = False) -> list[str]:
        """Close out sessions whose process died, and give each a report of what it had stored."""
        recovered = self.store.recover_stale_runs(0 if force else self.stale_after_seconds, self.clock())
        for run_id in recovered:
            try:
                write_report(self.store, self.settings, run_id)
            except Exception as exc:  # recovery must not fail because one report could not be rendered
                self.store.add_event(run_id, None, "report_error", {"error": f"{type(exc).__name__}: {exc}"})
        return recovered

    # -- admission --------------------------------------------------------

    def resolve_budget(self, watch: dict[str, Any], overrides: dict[str, Any] | None = None) -> Budget:
        base = {
            "max_duration_seconds": self.settings.max_duration_seconds,
            "max_search_requests": self.settings.max_search_requests,
            "max_fetched_pages": self.settings.max_fetched_pages,
            "max_llm_output_tokens": self.settings.max_llm_output_tokens,
        }
        base.update({k: v for k, v in (watch.get("budget") or {}).items() if v is not None})
        base.update({k: v for k, v in (overrides or {}).items() if v is not None})
        return Budget.model_validate(base)

    def admit(self, watch_id: str, trigger: str, overrides: dict[str, Any] | None = None,
              run_id: str | None = None) -> dict[str, Any]:
        run_id = run_id or random_id("run", 12)
        with self.store.tx():
            watch = self.store.get_watch(watch_id)
            if watch is None:
                raise RunRefused(f"unknown watch '{watch_id}'")
            if not self.watches.is_researchable(watch):
                raise RunRefused(f"watch '{watch_id}' is {watch['status']}; only confirmed ACTIVE watches are researched")
            if self.store.get_run(run_id) is not None:
                raise RunRefused(f"run id '{run_id}' already exists; run ids are single-use")
            running = self.store.running_run()
            if running is not None:
                raise RunRefused(f"session '{running['id']}' is still running; one session runs at a time")
            budget = self.resolve_budget(watch, overrides)
            now = self.store.now()
            try:
                self.store.insert("session_run", {
                    "id": run_id, "watch_id": watch_id, "revision_id": watch["active_revision_id"],
                    "seq": self.store.next_seq(watch_id), "status": RunStatus.RUNNING, "trigger": trigger,
                    "policy": watch["checkpoint_policy"], "model": self.engine.llm.model,
                    "search_provider": self.engine.search.name, "budget": budget.model_dump(),
                    "started_at": now, "heartbeat_at": now,
                })
            except sqlite3.IntegrityError as exc:
                raise RunRefused(f"could not admit run: {exc}") from exc
        return self.store.get_run(run_id)  # type: ignore[return-value]

    # -- execution --------------------------------------------------------

    def run_watch(self, ref: str, *, trigger: str = "manual", overrides: dict[str, Any] | None = None,
                  run_id: str | None = None, should_stop: Callable[[], bool] | None = None) -> SessionOutcome:
        """Admit and run one session for a named watch (manual use; the queue calls admit/execute itself)."""
        self.recover()
        watch = self.watches.get(ref)
        run = self.admit(watch["id"], trigger, overrides, run_id)
        return self.execute_admitted(run["id"], should_stop)

    def execute_admitted(self, run_id: str, should_stop: Callable[[], bool] | None = None) -> SessionOutcome:
        run = self.store.get_run(run_id)
        if run is None:
            raise RunRefused(f"unknown run '{run_id}'")
        watch_id = run["watch_id"]

        def should_cancel() -> bool:
            current = self.store.get_watch(watch_id)
            return current is None or current["status"] != WatchStatus.ACTIVE or bool(should_stop and should_stop())

        # Every policy runs the identical engine; only what happens after the session differs.
        outcome = self.engine.execute(run_id, should_cancel=should_cancel)
        if outcome.status is RunStatus.COMPLETED:
            decision = self.evaluate_checkpoint(watch_id, outcome)
            if decision.request_review:
                write_report(self.store, self.settings, outcome.run_id)  # include the pending question
            if outcome.has_meaningful_update:
                # For a future delivery adapter. Sessions with nothing new are recorded but queue no message,
                # and cancelled or failed sessions are never offered for publication.
                report = self.store.get_report(outcome.run_id) or {}
                self.store.insert("outbox", {
                    "run_id": outcome.run_id, "watch_id": watch_id, "kind": "REPORT",
                    "payload": {"run_id": outcome.run_id, "json_path": report.get("json_path"),
                                "markdown_path": report.get("markdown_path")},
                    "created_at": self.store.now()}, or_ignore=True)
        return outcome

    def evaluate_checkpoint(self, watch_id: str, outcome: SessionOutcome) -> CheckpointDecision:
        """Ask the watch's policy whether to keep going or hand a decision to the user."""
        watch = self.store.get_watch(watch_id)
        assert watch is not None
        policy = get_policy(watch["checkpoint_policy"])
        context = CheckpointContext.load(self.store, watch, outcome)
        decision = evaluate(policy, context)
        self.reviews.record_decision(watch_id, outcome.run_id, policy.name, decision)
        if decision.request_review and watch["status"] == WatchStatus.ACTIVE:
            self.reviews.open_request(watch_id, outcome.run_id, policy.name, decision)
        return decision
