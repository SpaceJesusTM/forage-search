"""Watch controller (Boundary B): the only code that changes a watch's status or approved objective.

Research code can read an approved revision and can *propose* things. Every change
to what a watch is allowed to research passes through `approve` or a confirmed
review response here, and is recorded with who did it.
"""

from __future__ import annotations

from typing import Any

from forage.engine import prompts
from forage.engine.schemas import WatchDraft
from forage.llm.base import LLMClient, LLMError
from forage.storage.store import Store
from forage.types import RevisionState, RunStatus, WatchSpec, WatchStatus
from forage.util import random_id, slugify

S = WatchStatus

ALLOWED_TRANSITIONS: frozenset[tuple[WatchStatus, WatchStatus]] = frozenset({
    (S.DRAFT, S.PENDING_CONFIRMATION),
    (S.PENDING_CONFIRMATION, S.ACTIVE),
    (S.ACTIVE, S.PAUSED),
    (S.ACTIVE, S.AWAITING_REVIEW),
    (S.ACTIVE, S.CLOSED),
    (S.PAUSED, S.CLOSED),
    (S.AWAITING_REVIEW, S.CLOSED),
    (S.PAUSED, S.REVIEW_PROPOSED),
    (S.CLOSED, S.REVIEW_PROPOSED),
    (S.REVIEW_PROPOSED, S.ACTIVE),
    (S.AWAITING_REVIEW, S.ACTIVE),
    # Not in the minimal lifecycle, added so no state is a dead end:
    (S.DRAFT, S.CLOSED),  # abandon before proposing
    (S.PENDING_CONFIRMATION, S.CLOSED),  # abandon before approving
    (S.REVIEW_PROPOSED, S.PAUSED),  # decline a resume plan
    (S.REVIEW_PROPOSED, S.CLOSED),  # decline a reopen plan
    (S.AWAITING_REVIEW, S.PAUSED),  # user answers a review with "pause"
})

# Only these statuses may be given research time.
RESEARCHABLE = frozenset({S.ACTIVE})

CHECKPOINT_POLICIES = ("continuous",)  # periodic and adaptive arrive with the checkpoint milestone


class WatchError(Exception):
    pass


class InvalidTransition(WatchError):
    pass


class ConfirmationRequired(WatchError):
    """A response would change the approved scope and was not explicitly confirmed."""

    def __init__(self, message: str, preview: dict[str, Any]):
        super().__init__(message)
        self.preview = preview


SPEC_LIST_FIELDS = ("key_questions", "include", "exclude", "priorities", "source_hints")


def apply_spec_patch(spec: WatchSpec, patch: dict[str, Any]) -> WatchSpec:
    """Apply `{field}_add` / `{field}_remove` / scalar overrides to a copy of `spec`."""
    data = spec.model_dump()
    for name in SPEC_LIST_FIELDS:
        current = list(data[name])
        removals = {str(v).casefold() for v in patch.get(f"{name}_remove", [])}
        current = [v for v in current if v.casefold() not in removals]
        current += [v for v in patch.get(f"{name}_add", []) if str(v).casefold() not in {c.casefold() for c in current}]
        if name in patch:
            current = list(patch[name])
        data[name] = current
    for name in ("geography", "freshness_days", "notes"):
        if name in patch:
            data[name] = patch[name]
    return WatchSpec.model_validate(data)


def spec_diff(old: WatchSpec, new: WatchSpec) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    a, b = old.model_dump(), new.model_dump()
    for name in SPEC_LIST_FIELDS:
        added = [v for v in b[name] if v not in a[name]]
        removed = [v for v in a[name] if v not in b[name]]
        if added or removed:
            diff[name] = {"added": added, "removed": removed}
    for name in ("geography", "freshness_days", "notes"):
        if a[name] != b[name]:
            diff[name] = {"from": a[name], "to": b[name]}
    return diff


class WatchController:
    def __init__(self, store: Store):
        self.store = store

    # -- lookup -----------------------------------------------------------

    def get(self, ref: str) -> dict[str, Any]:
        watch = self.store.find_watch(ref)
        if watch is None:
            raise WatchError(f"no watch matches '{ref}'")
        return watch

    def active_spec(self, watch: dict[str, Any]) -> tuple[str, WatchSpec] | None:
        revision = self.store.get_revision(watch["active_revision_id"])
        if revision is None:
            return None
        return revision["objective"], WatchSpec.model_validate(revision["spec"])

    def is_researchable(self, watch: dict[str, Any]) -> bool:
        return WatchStatus(watch["status"]) in RESEARCHABLE and bool(watch["active_revision_id"])

    # -- internals --------------------------------------------------------

    def _transition(self, watch: dict[str, Any], to: WatchStatus, actor: str, kind: str,
                    detail: dict[str, Any] | None = None, **extra: Any) -> None:
        current = WatchStatus(watch["status"])
        if (current, to) not in ALLOWED_TRANSITIONS:
            raise InvalidTransition(f"watch '{watch['id']}' cannot go from {current} to {to}")
        self.store.update("watch", {"id": watch["id"]}, {"status": to, "updated_at": self.store.now(), **extra})
        self.store.add_watch_event(watch["id"], kind, actor, current, to, detail)

    def _add_revision(self, watch_id: str, objective: str, spec: WatchSpec, origin: str, note: str | None) -> str:
        number = int(self.store.scalar(
            "SELECT COALESCE(MAX(revision_no), 0) + 1 FROM watch_revision WHERE watch_id = ?", (watch_id,)))
        revision_id = f"{watch_id}.r{number}"
        self.store.insert("watch_revision", {
            "id": revision_id, "watch_id": watch_id, "revision_no": number, "objective": objective.strip(),
            "spec": spec.model_dump(), "origin": origin, "state": RevisionState.PROPOSED, "note": note,
            "created_at": self.store.now(),
        })
        return revision_id

    def _retire_pending(self, watch: dict[str, Any], state: RevisionState, actor: str) -> None:
        if watch["pending_revision_id"]:
            self.store.update("watch_revision", {"id": watch["pending_revision_id"]},
                              {"state": state, "decided_at": self.store.now(), "decided_by": actor})

    def _cancel_running(self, watch_id: str) -> None:
        self.store.execute(
            "UPDATE session_run SET cancel_requested = 1 WHERE watch_id = ? AND status = ?", (watch_id, RunStatus.RUNNING))

    # -- creation and approval --------------------------------------------

    def draft_spec(self, objective: str, llm: LLMClient) -> tuple[str, WatchSpec]:
        """Ask the local model for a structured brief. The result is only ever a proposal."""
        draft, _ = llm.generate(purpose="structure_watch", **_prompt(prompts.structure_watch(objective)),
                                schema=WatchDraft, max_tokens=420)
        spec = WatchSpec(key_questions=draft.key_questions[:5], include=draft.include, exclude=draft.exclude,
                         priorities=draft.priorities, freshness_days=draft.freshness_days)
        return draft.title, spec

    def create(self, objective: str, *, title: str | None = None, spec: WatchSpec | None = None,
               llm: LLMClient | None = None, policy: str = "continuous", policy_config: dict[str, Any] | None = None,
               budget: dict[str, Any] | None = None, actor: str = "user") -> dict[str, Any]:
        objective = " ".join(objective.split())
        if len(objective) < 8:
            raise WatchError("the objective is too short to research")
        if policy not in CHECKPOINT_POLICIES:
            raise WatchError(f"unknown checkpoint policy '{policy}'")
        origin, note = "user", None
        if spec is None:
            spec = WatchSpec()
            if llm is not None:
                try:
                    drafted_title, spec = self.draft_spec(objective, llm)
                    title = title or drafted_title
                    origin = "model_draft"
                    note = f"structured by {llm.model}; needs user confirmation"
                except LLMError as exc:
                    note = f"model could not structure the objective ({exc}); using the objective as written"
        title = (title or " ".join(objective.split()[:8])).strip()[:90]
        watch_id = f"{slugify(title, 28)}-{random_id('x', 4)[2:]}"
        now = self.store.now()
        with self.store.tx():
            self.store.insert("watch", {
                "id": watch_id, "title": title, "status": S.DRAFT, "checkpoint_policy": policy,
                "policy_config": policy_config or {}, "budget": budget or {},
                "queue_position": self.store.next_queue_position(), "created_at": now, "updated_at": now,
            })
            self.store.add_watch_event(watch_id, "created", actor, None, S.DRAFT, {"objective": objective})
            revision_id = self._add_revision(watch_id, objective, spec, origin, note)
            self._transition(self.get(watch_id), S.PENDING_CONFIRMATION, actor, "revision_proposed",
                             {"revision_id": revision_id, "origin": origin}, pending_revision_id=revision_id)
        return self.get(watch_id)

    def propose_revision(self, ref: str, *, patch: dict[str, Any] | None = None, objective: str | None = None,
                         title: str | None = None, origin: str = "user", note: str | None = None,
                         actor: str = "user") -> dict[str, Any]:
        """Stage an amended objective. It takes effect only when approved."""
        watch = self.get(ref)
        status = WatchStatus(watch["status"])
        if status in (S.CLOSED, S.PAUSED, S.AWAITING_REVIEW):
            raise WatchError(f"watch is {status}; use resume/reopen or answer its review to change it")
        base = self.store.get_revision(watch["pending_revision_id"]) or self.store.get_revision(watch["active_revision_id"])
        if base is None:
            raise WatchError("watch has no revision to amend")
        spec = apply_spec_patch(WatchSpec.model_validate(base["spec"]), patch or {})
        with self.store.tx():
            self._retire_pending(watch, RevisionState.SUPERSEDED, actor)
            revision_id = self._add_revision(watch["id"], objective or base["objective"], spec, origin, note)
            values: dict[str, Any] = {"pending_revision_id": revision_id, "updated_at": self.store.now()}
            if title:
                values["title"] = title.strip()[:90]
            self.store.update("watch", {"id": watch["id"]}, values)
            self.store.add_watch_event(watch["id"], "revision_proposed", actor, status, status,
                                       {"revision_id": revision_id, "origin": origin})
        return self.get(watch["id"])

    def approve(self, ref: str, actor: str = "user") -> dict[str, Any]:
        """Confirm the pending revision. This is the only path by which an objective becomes researchable."""
        watch = self.get(ref)
        status = WatchStatus(watch["status"])
        pending = self.store.get_revision(watch["pending_revision_id"])
        if pending is None or pending["state"] != RevisionState.PROPOSED:
            raise WatchError("nothing is waiting for confirmation on this watch")
        if status not in (S.PENDING_CONFIRMATION, S.REVIEW_PROPOSED, S.ACTIVE):
            raise InvalidTransition(f"watch is {status}; it cannot be approved from this state")
        now = self.store.now()
        with self.store.tx():
            if watch["active_revision_id"]:
                self.store.update("watch_revision", {"id": watch["active_revision_id"]}, {"state": RevisionState.SUPERSEDED})
            self.store.update("watch_revision", {"id": pending["id"]},
                              {"state": RevisionState.APPROVED, "decided_at": now, "decided_by": actor})
            values = {"active_revision_id": pending["id"], "pending_revision_id": None, "previous_status": None}
            detail = {"revision_id": pending["id"]}
            if status is S.ACTIVE:
                self.store.update("watch", {"id": watch["id"]}, {**values, "updated_at": now})
                self.store.add_watch_event(watch["id"], "revision_approved", actor, status, status, detail)
            else:
                self._transition(watch, S.ACTIVE, actor, "approved", detail, **values)
        return self.get(watch["id"])

    # -- user controls ----------------------------------------------------

    def pause(self, ref: str, actor: str = "user") -> dict[str, Any]:
        """Immediate, no confirmation. A running session for the watch is asked to stop."""
        watch = self.get(ref)
        with self.store.tx():
            self._transition(watch, S.PAUSED, actor, "paused")
            self._cancel_running(watch["id"])
        return self.get(watch["id"])

    def close(self, ref: str, actor: str = "user") -> dict[str, Any]:
        """Archive. Nothing is deleted."""
        watch = self.get(ref)
        with self.store.tx():
            self._retire_pending(watch, RevisionState.REJECTED, actor)
            self._transition(watch, S.CLOSED, actor, "closed", pending_revision_id=None, previous_status=None)
            self._cancel_running(watch["id"])
            self.store.execute(
                "UPDATE checkpoint_request SET status = 'WITHDRAWN', answered_at = ? WHERE watch_id = ? AND status = 'PENDING'",
                (self.store.now(), watch["id"]))
        return self.get(watch["id"])

    def request_resume(self, ref: str, *, patch: dict[str, Any] | None = None, note: str | None = None,
                       actor: str = "user") -> dict[str, Any]:
        """PAUSED or CLOSED -> REVIEW_PROPOSED with a refreshed plan the user must confirm."""
        watch = self.get(ref)
        status = WatchStatus(watch["status"])
        base = self.store.get_revision(watch["active_revision_id"])
        if base is None:
            raise WatchError("watch was never approved; approve or recreate it instead")
        spec = apply_spec_patch(WatchSpec.model_validate(base["spec"]), patch or {})
        with self.store.tx():
            revision_id = self._add_revision(watch["id"], base["objective"], spec, "resume_plan",
                                             note or f"resume plan after {status}")
            self._transition(watch, S.REVIEW_PROPOSED, actor, "resume_proposed", {"revision_id": revision_id},
                             pending_revision_id=revision_id, previous_status=status)
        return self.get(watch["id"])

    def decline(self, ref: str, actor: str = "user") -> dict[str, Any]:
        """Reject whatever is pending. A resume plan returns the watch to where it was."""
        watch = self.get(ref)
        status = WatchStatus(watch["status"])
        with self.store.tx():
            self._retire_pending(watch, RevisionState.REJECTED, actor)
            if status is S.REVIEW_PROPOSED:
                back = WatchStatus(watch["previous_status"] or S.PAUSED)
                self._transition(watch, back, actor, "resume_declined", pending_revision_id=None, previous_status=None)
            elif status is S.PENDING_CONFIRMATION:
                self._transition(watch, S.CLOSED, actor, "declined", pending_revision_id=None)
            elif status is S.ACTIVE and watch["pending_revision_id"]:
                self.store.update("watch", {"id": watch["id"]}, {"pending_revision_id": None, "updated_at": self.store.now()})
                self.store.add_watch_event(watch["id"], "revision_declined", actor, status, status)
            else:
                raise WatchError("nothing is waiting for confirmation on this watch")
        return self.get(watch["id"])

    def set_policy(self, ref: str, policy: str, config: dict[str, Any] | None = None, actor: str = "user") -> dict[str, Any]:
        if policy not in CHECKPOINT_POLICIES:
            raise WatchError(f"unknown checkpoint policy '{policy}'")
        watch = self.get(ref)
        self.store.update("watch", {"id": watch["id"]},
                          {"checkpoint_policy": policy, "policy_config": config or {}, "updated_at": self.store.now()})
        self.store.add_watch_event(watch["id"], "policy_changed", actor, watch["status"], watch["status"],
                                   {"policy": policy, "config": config or {}})
        return self.get(watch["id"])

    def resume_plan(self, ref: str) -> dict[str, Any]:
        """Deterministic digest of a watch's history, shown before the user confirms a resume."""
        watch = self.get(ref)
        watch_id = watch["id"]
        runs = self.store.list_runs(watch_id)
        last = runs[-1] if runs else None
        pending = self.store.get_revision(watch["pending_revision_id"])
        active = self.store.get_revision(watch["active_revision_id"])
        diff = {}
        if pending and active:
            diff = spec_diff(WatchSpec.model_validate(active["spec"]), WatchSpec.model_validate(pending["spec"]))
        return {
            "watch_id": watch_id,
            "status": watch["status"],
            "sessions_run": len(runs),
            "last_session_at": last["started_at"] if last else None,
            "findings": int(self.store.scalar("SELECT COUNT(*) FROM finding WHERE watch_id = ?", (watch_id,)) or 0),
            "open_questions": [q["text"] for q in self.store.list_questions(watch_id, "OPEN")],
            "saved_directions": [lead["text"] for lead in self.store.list_leads(watch_id, "OPEN")],
            "unread_relevant_sources": int(self.store.scalar(
                "SELECT COUNT(*) FROM source WHERE watch_id = ? AND status = 'DISCOVERED' AND relevance >= 2", (watch_id,)) or 0),
            "scope_changes_in_plan": diff,
        }


def _prompt(parts: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    system, user, payload = parts
    return {"system": system, "user": user, "payload": payload}
