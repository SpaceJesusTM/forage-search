"""Checkpoint policy interface (Boundary C) — the experimental variable.

A policy is a pure function of stored state: after each completed session it
returns CONTINUE or REQUEST_REVIEW. It never runs research, never changes a watch
and never sees anything the other policies do not see. Swapping the policy leaves
the engine, model, search provider and budgets untouched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from forage.engine.session import SessionOutcome
from forage.storage.store import Store
from forage.types import QuestionStatus, RunStatus, WatchSpec

CONTINUE = "CONTINUE"
REQUEST_REVIEW = "REQUEST_REVIEW"


@dataclass
class Choice:
    id: str
    label: str
    effect: str  # continue | amend | pause | close
    spec_patch: dict[str, Any] | None = None  # applied only for effect == "amend", and only after confirmation

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CheckpointDecision:
    action: str
    reason: str
    signal: str | None = None
    signals: dict[str, Any] = field(default_factory=dict)
    what_happened: str = ""
    why_human: str = ""
    question: str = ""
    choices: list[Choice] = field(default_factory=list)
    safe_default: str = "continue"
    urgency: str = "normal"
    evidence: list[dict[str, Any]] = field(default_factory=list)

    @property
    def request_review(self) -> bool:
        return self.action == REQUEST_REVIEW

    @classmethod
    def carry_on(cls, reason: str, signals: dict[str, Any] | None = None) -> "CheckpointDecision":
        return cls(CONTINUE, reason, signals=signals or {})


@dataclass
class CheckpointContext:
    """Everything a policy may look at. Built from the store, so decisions can be replayed."""

    watch_id: str
    objective: str
    spec: WatchSpec
    config: dict[str, Any]
    completed_sessions: int
    sessions_since_review: int
    run_metrics: dict[str, Any]
    recent_runs: list[dict[str, Any]]  # completed runs, oldest first, including this one
    open_questions: list[dict[str, Any]]
    guidance: dict[str, Any] | None  # the model's own suggestion that a user decision is needed
    out_of_scope: list[dict[str, Any]]
    reviews_answered: int
    last_review_signal: str | None
    hard_limits: dict[str, Any] = field(default_factory=dict)  # user-set lifetime limits for the watch

    @classmethod
    def load(cls, store: Store, watch: dict[str, Any], outcome: SessionOutcome) -> "CheckpointContext":
        watch_id = watch["id"]
        revision = store.get_revision(watch["active_revision_id"]) or {"objective": "", "spec": {}}
        completed = [r for r in store.list_runs(watch_id) if r["status"] == RunStatus.COMPLETED]
        last_answer = store.one(
            "SELECT r.signal, COALESCE(s.seq, 0) AS seq FROM checkpoint_request r LEFT JOIN session_run s ON s.id = r.run_id "
            "WHERE r.watch_id = ? AND r.status = 'ANSWERED' ORDER BY r.rowid DESC LIMIT 1", (watch_id,))
        # Sessions that started after the one which raised the last answered review.
        since = [r for r in completed if not last_answer or r["seq"] > last_answer["seq"]]
        metrics = outcome.metrics or {}
        return cls(
            watch_id=watch_id,
            objective=revision["objective"],
            spec=WatchSpec.model_validate(revision["spec"]),
            config=watch.get("policy_config") or {},
            completed_sessions=len(completed),
            sessions_since_review=len(since),
            run_metrics=metrics,
            recent_runs=completed[-8:],
            open_questions=store.list_questions(watch_id, QuestionStatus.OPEN),
            guidance=metrics.get("guidance_proposed"),
            out_of_scope=metrics.get("out_of_scope_examples", []),
            reviews_answered=int(store.scalar(
                "SELECT COUNT(*) FROM checkpoint_request WHERE watch_id = ? AND status = 'ANSWERED'", (watch_id,)) or 0),
            last_review_signal=last_answer["signal"] if last_answer else None,
            hard_limits={k: v for k, v in (watch.get("budget") or {}).items() if k.startswith("max_total_")},
        )


class CheckpointPolicy(Protocol):
    name: str

    def evaluate(self, context: CheckpointContext) -> CheckpointDecision: ...


class ContinuousPolicy:
    """Baseline: never asks for a strategic review. User pause/close and hard limits still apply."""

    name = "continuous"

    def evaluate(self, context: CheckpointContext) -> CheckpointDecision:
        return CheckpointDecision.carry_on("continuous policy never requests review")


def standard_choices(extra: list[Choice] | None = None) -> list[Choice]:
    """Choices every review offers, after any policy-specific ones."""
    return [
        Choice("continue", "Continue with the current objective and scope", "continue"),
        *(extra or []),
        Choice("pause", "Pause this watch", "pause"),
        Choice("close", "Close this watch (history is kept)", "close"),
    ]


def hard_stop(context: CheckpointContext) -> CheckpointDecision | None:
    """Gate shared by every policy: a user-set lifetime limit was reached.

    It is deliberately outside the policies. A stop that all three conditions apply in the
    same way must not be counted as a benefit of the adaptive policy; its signal name
    starts with `common_` so analyses can exclude it.
    """
    limit = context.hard_limits.get("max_total_sessions")
    if not limit or context.completed_sessions < int(limit):
        return None
    if context.sessions_since_review < 1:
        return None
    return CheckpointDecision(
        action=REQUEST_REVIEW,
        reason=f"lifetime limit of {limit} sessions reached",
        signal="common_hard_limit",
        signals={"completed_sessions": context.completed_sessions, "max_total_sessions": int(limit)},
        what_happened=f"This watch has now run {context.completed_sessions} session(s), the limit you set for it.",
        why_human="Spending beyond a limit you set needs your say-so.",
        question="The session limit for this watch is reached. Continue anyway, pause or close it?",
        choices=standard_choices(),
        safe_default="pause",
        urgency="normal",
    )


def evaluate(policy: CheckpointPolicy, context: CheckpointContext) -> CheckpointDecision:
    """The decision for one completed session: the shared hard-stop gate first, then the policy."""
    return hard_stop(context) or policy.evaluate(context)


def get_policy(name: str) -> CheckpointPolicy:
    if name == "continuous":
        return ContinuousPolicy()
    if name == "periodic":
        from forage.checkpoint.periodic import PeriodicPolicy

        return PeriodicPolicy()
    if name == "adaptive":
        from forage.checkpoint.adaptive import AdaptivePolicy

        return AdaptivePolicy()
    raise ValueError(f"unknown checkpoint policy '{name}'")
