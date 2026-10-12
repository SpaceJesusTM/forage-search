"""Drop-in engine that runs a session through the LangGraph graph.

It exposes the same `execute(run_id, should_cancel)` call as `ResearchEngine`, so the
run manager, scheduler, watch controller and checkpoint policies work with it unchanged.
What it adds is what step checkpointing makes possible: a session that stopped part-way
(crash, cancellation, or a question for the user) can be continued with `resume()`.
"""

from __future__ import annotations

import sqlite3
import traceback
from pathlib import Path
from typing import Any, Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, RetryPolicy

from forage.engine.schemas import GuidanceProposal
from forage.engine.session import ResearchEngine, SessionCancelled, SessionOutcome
from forage.experimental.langgraph_engine.graph import build_graph
from forage.experimental.langgraph_engine.nodes import EngineContext
from forage.experimental.langgraph_engine.steps import StepSession
from forage.llm.base import LLMUnavailable
from forage.types import RunStatus
from forage.util import iso

CHECKPOINT_FILE = "langgraph_checkpoints.sqlite3"


class LangGraphEngine:
    def __init__(self, base: ResearchEngine, *, checkpointer: BaseCheckpointSaver | None = None,
                 checkpoint_path: str | Path | None = None, durability: str = "sync",
                 ask_before_follow_up: bool = False, keep_checkpoints: bool = True,
                 retry: dict[str, RetryPolicy] | None = None):
        self.base = base
        self.durability = durability
        self.ask_before_follow_up = ask_before_follow_up
        self.keep_checkpoints = keep_checkpoints
        self._conn: sqlite3.Connection | None = None
        if checkpointer is None:
            path = Path(checkpoint_path or base.settings.data_dir / CHECKPOINT_FILE)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
            checkpointer = SqliteSaver(self._conn)
        self.checkpointer = checkpointer
        self.graph = build_graph(retry).compile(checkpointer=checkpointer)

    # The run manager and tests read these from whatever engine they are given.
    store = property(lambda self: self.base.store)
    llm = property(lambda self: self.base.llm)
    search = property(lambda self: self.base.search)
    fetcher = property(lambda self: self.base.fetcher)
    settings = property(lambda self: self.base.settings)
    clock = property(lambda self: self.base.clock)
    progress = property(lambda self: self.base.progress)

    @property
    def monotonic(self) -> Callable[[], float]:
        return self.base.monotonic

    @monotonic.setter
    def monotonic(self, value: Callable[[], float]) -> None:
        self.base.monotonic = value

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------ inspection

    @staticmethod
    def _config(run_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": run_id}}

    def snapshot(self, run_id: str):
        """LangGraph's view of the session: state values, the node(s) due next, pending interrupts."""
        return self.graph.get_state(self._config(run_id))

    def history(self, run_id: str) -> list[tuple[int, tuple[str, ...]]]:
        """(step, next nodes) for every checkpoint of the session, oldest first."""
        items = [(s.metadata.get("step", -1), s.next) for s in self.graph.get_state_history(self._config(run_id))]
        return list(reversed(items))

    def pending_question(self, run_id: str) -> dict[str, Any] | None:
        for task in self.snapshot(run_id).tasks:
            for item in getattr(task, "interrupts", ()) or ():
                return item.value
        return None

    def can_resume(self, run_id: str) -> bool:
        return bool(self.snapshot(run_id).next)

    # -------------------------------------------------------------- running

    def execute(self, run_id: str, should_cancel: Callable[[], bool] | None = None,
                allow_guidance: bool = True) -> SessionOutcome:
        """Run an admitted session from the start."""
        return self._drive(run_id, {"run_id": run_id}, should_cancel)

    def resume(self, run_id: str, should_cancel: Callable[[], bool] | None = None, answer: Any = None) -> SessionOutcome:
        """Continue a session from its last completed step.

        Works after a crash, after a cancellation, and after the session stopped to ask something
        (pass the reply as `answer`). The step that was in flight runs again from its beginning.
        """
        run = self.store.get_run(run_id)
        if run is None:
            raise ValueError(f"unknown run '{run_id}'")
        if not self.can_resume(run_id):
            raise ValueError(f"run '{run_id}' has nothing left to resume")
        other = self.store.running_run()
        if other is not None and other["id"] != run_id:
            raise ValueError(f"session '{other['id']}' is running; one session runs at a time")
        revision_id = self._revision_to_continue_under(run)
        now = iso(self.base.clock())
        self.store.update("session_run", {"id": run_id}, {
            "status": RunStatus.RUNNING, "finished_at": None, "error": None, "stop_reason": None,
            "cancel_requested": 0, "heartbeat_at": now, "revision_id": revision_id})
        self.store.add_event(run_id, run["watch_id"], "session_resumed", {"next": list(self.snapshot(run_id).next)})
        asked = self.pending_question(run_id) is not None
        return self._drive(run_id, Command(resume=answer) if asked else None, should_cancel)

    def _revision_to_continue_under(self, run: dict[str, Any]) -> str:
        """A session may only continue under what the user currently approves.

        Pausing and resuming a watch creates a new approved revision. If its objective and scope are
        the ones the session started with, the session carries on under it. If they changed, the
        half-finished plan was made for a different brief, so continuing is refused.
        """
        watch = self.store.get_watch(run["watch_id"])
        if watch is None or watch["status"] != "ACTIVE" or not watch["active_revision_id"]:
            raise ValueError(f"watch '{run['watch_id']}' is not active; a session cannot continue")
        if watch["active_revision_id"] == run["revision_id"]:
            return run["revision_id"]
        started, current = self.store.get_revision(run["revision_id"]), self.store.get_revision(watch["active_revision_id"])
        if started and current and (started["objective"], started["spec"]) == (current["objective"], current["spec"]):
            return current["id"]
        raise ValueError("the approved objective or scope changed since this session stopped; start a new session instead")

    def _drive(self, run_id: str, graph_input: Any, should_cancel: Callable[[], bool] | None) -> SessionOutcome:
        context = EngineContext(self.base, should_cancel or (lambda: False), self.ask_before_follow_up)
        config = self._config(run_id)
        try:
            result = self.graph.invoke(graph_input, config, context=context, durability=self.durability)
        except SessionCancelled as exc:
            return self._stop(run_id, RunStatus.CANCELLED, "cancelled", str(exc) or None)
        except LLMUnavailable as exc:
            return self._stop(run_id, RunStatus.FAILED, "llm_unavailable", str(exc))
        except Exception as exc:  # a session must always leave a truthful run record behind
            error = f"{type(exc).__name__}: {exc}"
            run = self.store.get_run(run_id) or {}
            self.store.add_event(run_id, run.get("watch_id"), "session_error",
                                 {"error": error, "traceback": traceback.format_exc()[-3000:]})
            return self._stop(run_id, RunStatus.FAILED, "error", error)
        if isinstance(result, dict) and result.get("__interrupt__"):
            payload = result["__interrupt__"][0].value
            run = self.store.get_run(run_id) or {}
            self.store.add_event(run_id, run.get("watch_id"), "awaiting_input", {"question": payload})
            return self._stop(run_id, RunStatus.CANCELLED, "awaiting_input", None)
        if not self.keep_checkpoints:
            self.checkpointer.delete_thread(run_id)
        return self._outcome(run_id)

    def _stop(self, run_id: str, status: RunStatus, stop_reason: str, error: str | None) -> SessionOutcome:
        """Close the run record for a session that did not reach its report step. Its checkpoint is kept."""
        values = dict(self.snapshot(run_id).values or {"run_id": run_id})
        try:
            session = StepSession(self.base, values)
        except ValueError as exc:  # the run record itself is unusable; still leave a truthful status behind
            self.store.update("session_run", {"id": run_id}, {
                "status": status, "finished_at": iso(self.base.clock()), "stop_reason": stop_reason,
                "error": error or str(exc)})
            return self._outcome(run_id)
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


def use_langgraph_engine(app, **options: Any) -> LangGraphEngine:
    """Swap an `App`'s engine for the LangGraph one in place. Everything else in the app is untouched."""
    engine = LangGraphEngine(app.engine, **options)
    app.engine = engine
    app.runner.engine = engine
    return engine
