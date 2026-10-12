"""Comparison driver: the same steps and control flow, run by a plain loop with its own checkpoints.

This exists to answer one question in the LangGraph evaluation: once the session is a set
of resumable steps, how much does the framework add over the least code that could drive
them? It supports what the comparison exercises (run, crash/cancel resume, a question to
the user) and nothing else. It has no parallel steps, subgraphs, retry policies, streaming
or time-travel API.
"""

from __future__ import annotations

import json
import sqlite3
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from forage.engine.schemas import GuidanceProposal
from forage.engine.session import ResearchEngine, SessionCancelled, SessionOutcome
from forage.experimental.langgraph_engine.nodes import ENTRY, EngineContext, next_step, run_step
from forage.experimental.langgraph_engine.steps import StepSession
from forage.llm.base import LLMUnavailable
from forage.types import RunStatus
from forage.util import iso

CHECKPOINT_FILE = "step_checkpoints.sqlite3"


class NeedsInput(Exception):
    """A step asked the user something. The session stops here until `resume(answer=...)`."""

    def __init__(self, payload: dict[str, Any]):
        super().__init__("session is waiting for input")
        self.payload = payload


class PlainStepEngine:
    def __init__(self, base: ResearchEngine, *, checkpoint_path: str | Path | None = None,
                 ask_before_follow_up: bool = False, keep_history: bool = True):
        self.base = base
        self.ask_before_follow_up = ask_before_follow_up
        self.keep_history = keep_history
        path = Path(checkpoint_path or base.settings.data_dir / CHECKPOINT_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), isolation_level=None)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS step_checkpoint (run_id TEXT NOT NULL, step INTEGER NOT NULL, next TEXT, "
            "question TEXT, state TEXT NOT NULL, PRIMARY KEY (run_id, step))")

    store = property(lambda self: self.base.store)
    llm = property(lambda self: self.base.llm)
    search = property(lambda self: self.base.search)
    settings = property(lambda self: self.base.settings)

    @property
    def monotonic(self) -> Callable[[], float]:
        return self.base.monotonic

    @monotonic.setter
    def monotonic(self, value: Callable[[], float]) -> None:
        self.base.monotonic = value

    def close(self) -> None:
        self._conn.close()

    # --------------------------------------------------------- checkpoints

    def _latest(self, run_id: str) -> tuple[int, str | None, dict[str, Any] | None, dict[str, Any]] | None:
        row = self._conn.execute(
            "SELECT step, next, question, state FROM step_checkpoint WHERE run_id = ? ORDER BY step DESC LIMIT 1",
            (run_id,)).fetchone()
        if row is None:
            return None
        return row[0], row[1], json.loads(row[2]) if row[2] else None, json.loads(row[3])

    def _save(self, run_id: str, step: int, upcoming: str | None, state: dict[str, Any],
              question: dict[str, Any] | None = None) -> None:
        if not self.keep_history:
            self._conn.execute("DELETE FROM step_checkpoint WHERE run_id = ?", (run_id,))
        self._conn.execute(
            "INSERT OR REPLACE INTO step_checkpoint (run_id, step, next, question, state) VALUES (?, ?, ?, ?, ?)",
            (run_id, step, upcoming, json.dumps(question) if question else None, json.dumps(state, default=str)))

    def history(self, run_id: str) -> list[tuple[int, tuple[str, ...]]]:
        rows = self._conn.execute("SELECT step, next FROM step_checkpoint WHERE run_id = ? ORDER BY step", (run_id,))
        return [(step, (upcoming,) if upcoming else ()) for step, upcoming in rows]

    def can_resume(self, run_id: str) -> bool:
        latest = self._latest(run_id)
        return latest is not None and latest[1] is not None

    def pending_question(self, run_id: str) -> dict[str, Any] | None:
        latest = self._latest(run_id)
        return latest[2] if latest else None

    # -------------------------------------------------------------- running

    def execute(self, run_id: str, should_cancel: Callable[[], bool] | None = None,
                allow_guidance: bool = True) -> SessionOutcome:
        self._save(run_id, 0, ENTRY, {"run_id": run_id})
        return self._drive(run_id, should_cancel, None)

    def resume(self, run_id: str, should_cancel: Callable[[], bool] | None = None, answer: Any = None) -> SessionOutcome:
        run = self.store.get_run(run_id)
        if run is None:
            raise ValueError(f"unknown run '{run_id}'")
        if not self.can_resume(run_id):
            raise ValueError(f"run '{run_id}' has nothing left to resume")
        other = self.store.running_run()
        if other is not None and other["id"] != run_id:
            raise ValueError(f"session '{other['id']}' is running; one session runs at a time")
        watch = self.store.get_watch(run["watch_id"])
        if watch is None or watch["status"] != "ACTIVE" or not watch["active_revision_id"]:
            raise ValueError(f"watch '{run['watch_id']}' is not active; a session cannot continue")
        revision_id = run["revision_id"]
        if watch["active_revision_id"] != revision_id:
            started, current = self.store.get_revision(revision_id), self.store.get_revision(watch["active_revision_id"])
            if not (started and current and (started["objective"], started["spec"]) == (current["objective"], current["spec"])):
                raise ValueError("the approved objective or scope changed since this session stopped; start a new session instead")
            revision_id = current["id"]
        self.store.update("session_run", {"id": run_id}, {
            "status": RunStatus.RUNNING, "finished_at": None, "error": None, "stop_reason": None, "cancel_requested": 0,
            "heartbeat_at": iso(self.base.clock()), "revision_id": revision_id})
        self.store.add_event(run_id, run["watch_id"], "session_resumed", {"next": [self._latest(run_id)[1]]})
        return self._drive(run_id, should_cancel, answer)

    def _drive(self, run_id: str, should_cancel: Callable[[], bool] | None, answer: Any) -> SessionOutcome:
        step, name, question, state = self._latest(run_id)  # type: ignore[misc]
        replies = [answer] if question is not None else []

        def ask(payload: dict[str, Any]) -> Any:
            if replies:
                return replies.pop(0)
            raise NeedsInput(payload)

        context = EngineContext(self.base, should_cancel or (lambda: False), self.ask_before_follow_up, ask)
        try:
            while name is not None:
                state = state | run_step(name, state, context)
                name = next_step(name, state)
                step += 1
                self._save(run_id, step, name, state)  # the checkpoint: written only after the step finished
        except NeedsInput as exc:
            self._save(run_id, step, name, state, question=exc.payload)
            run = self.store.get_run(run_id) or {}
            self.store.add_event(run_id, run.get("watch_id"), "awaiting_input", {"question": exc.payload})
            return self._stop(run_id, state, RunStatus.CANCELLED, "awaiting_input", None)
        except SessionCancelled as exc:
            return self._stop(run_id, state, RunStatus.CANCELLED, "cancelled", str(exc) or None)
        except LLMUnavailable as exc:
            return self._stop(run_id, state, RunStatus.FAILED, "llm_unavailable", str(exc))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            run = self.store.get_run(run_id) or {}
            self.store.add_event(run_id, run.get("watch_id"), "session_error",
                                 {"error": error, "traceback": traceback.format_exc()[-3000:]})
            return self._stop(run_id, state, RunStatus.FAILED, "error", error)
        return self._outcome(run_id)

    def _stop(self, run_id: str, state: dict[str, Any], status: RunStatus, stop_reason: str, error: str | None) -> SessionOutcome:
        session = StepSession(self.base, state)
        if session.work_limits is None:
            session.work_limits = session.budget.limits_reached()
        return session._finalize(status, stop_reason, error)

    def _outcome(self, run_id: str) -> SessionOutcome:
        run = self.store.get_run(run_id)
        assert run is not None
        metrics = run["metrics"] or {}
        proposed = metrics.get("guidance_proposed")
        return SessionOutcome(
            run_id, run["watch_id"], RunStatus(run["status"]), bool(run["has_meaningful_update"]),
            GuidanceProposal.model_validate(proposed) if proposed else None, metrics, run["error"])


def use_engine(app, kind: str, **options: Any):
    """Swap an `App`'s engine in place: 'langgraph', 'plain' (this module) or 'linear' (no change)."""
    if kind == "linear":
        return app.engine
    if kind == "langgraph":
        from forage.experimental.langgraph_engine.engine import LangGraphEngine

        engine: Any = LangGraphEngine(app.engine, **options)
    elif kind == "plain":
        engine = PlainStepEngine(app.engine, **{k: v for k, v in options.items() if k in ("ask_before_follow_up", "checkpoint_path")})
    else:
        raise ValueError(f"unknown engine '{kind}'")
    app.engine = engine
    app.runner.engine = engine
    return engine
