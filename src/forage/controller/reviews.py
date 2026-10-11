"""Review requests and responses: how a watch enters and leaves AWAITING_REVIEW.

A request moves exactly one watch out of the research queue. A response that
would change the approved scope is refused unless it is explicitly confirmed,
and the confirmed change becomes a new approved revision with its provenance.
"""

from __future__ import annotations

from typing import Any

from forage.checkpoint.base import CheckpointDecision
from forage.controller.watches import ConfirmationRequired, WatchController, WatchError, apply_spec_patch, spec_diff
from forage.storage.db import loads
from forage.storage.store import Store
from forage.types import RevisionState, WatchSpec, WatchStatus
from forage.util import random_id


class ReviewController:
    def __init__(self, store: Store):
        self.store = store
        self.watches = WatchController(store)

    # -- logging ----------------------------------------------------------

    def record_decision(self, watch_id: str, run_id: str | None, policy: str, decision: CheckpointDecision) -> None:
        self.store.insert("checkpoint_decision", {
            "run_id": run_id, "watch_id": watch_id, "policy": policy, "decision": decision.action,
            "signal": decision.signal, "reason": decision.reason, "signals": decision.signals, "at": self.store.now(),
        })

    # -- requests ---------------------------------------------------------

    def open_request(self, watch_id: str, run_id: str | None, policy: str, decision: CheckpointDecision) -> dict[str, Any]:
        """Create the request and move the watch to AWAITING_REVIEW in one transaction."""
        if not decision.request_review:
            raise WatchError("decision does not request a review")
        if not decision.question or len(decision.choices) < 2:
            raise WatchError("a review request needs a question and at least two choices")
        if decision.safe_default not in {c.id for c in decision.choices}:
            raise WatchError("the safe default must be one of the choices")
        request_id = random_id("rev", 10)
        with self.store.tx():
            watch = self.watches.get(watch_id)
            self.store.insert("checkpoint_request", {
                "id": request_id, "watch_id": watch_id, "run_id": run_id, "policy": policy,
                "signal": decision.signal or "unspecified", "status": "PENDING", "what_happened": decision.what_happened,
                "why_human": decision.why_human, "question": decision.question,
                "choices": [c.as_dict() for c in decision.choices], "safe_default": decision.safe_default,
                "urgency": decision.urgency, "evidence": decision.evidence, "created_at": self.store.now(),
            })
            self.watches._transition(watch, WatchStatus.AWAITING_REVIEW, f"policy:{policy}", "review_requested",
                                     {"request_id": request_id, "signal": decision.signal})
            if run_id:
                self.store.insert("outbox", {
                    "run_id": run_id, "watch_id": watch_id, "kind": "REVIEW_REQUEST",
                    "payload": {"request_id": request_id, "question": decision.question},
                    "created_at": self.store.now(),
                }, or_ignore=True)
        return self.get(request_id)

    def get(self, request_id: str) -> dict[str, Any]:
        row = self.store.one("SELECT * FROM checkpoint_request WHERE id = ? OR id LIKE ?", (request_id, request_id + "%"))
        if row is None:
            raise WatchError(f"no review request matches '{request_id}'")
        row["choices"] = loads(row["choices"], [])
        row["evidence"] = loads(row["evidence"], [])
        return row

    def list(self, watch_id: str | None = None, status: str | None = "PENDING") -> list[dict[str, Any]]:
        clauses, params = [], []
        if watch_id:
            clauses.append("watch_id = ?")
            params.append(watch_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self.store.all(f"SELECT * FROM checkpoint_request{where} ORDER BY created_at", tuple(params))
        for row in rows:
            row["choices"] = loads(row["choices"], [])
            row["evidence"] = loads(row["evidence"], [])
        return rows

    def pending_for(self, watch_id: str) -> dict[str, Any] | None:
        rows = self.list(watch_id, "PENDING")
        return rows[0] if rows else None

    # -- responses --------------------------------------------------------

    def respond(self, request_id: str, choice_id: str, *, note: str | None = None, patch: dict[str, Any] | None = None,
                confirm_scope_change: bool = False, actor: str = "user") -> dict[str, Any]:
        """Answer a pending review.

        `patch` lets the user state their own scope change. Any scope change, whether
        from the chosen option or from `patch`, raises ConfirmationRequired until
        `confirm_scope_change` is true.
        """
        request = self.get(request_id)
        if request["status"] != "PENDING":
            raise WatchError(f"review '{request['id']}' is already {request['status']}")
        choice = next((c for c in request["choices"] if c["id"] == choice_id), None)
        if choice is None:
            valid = ", ".join(c["id"] for c in request["choices"])
            raise WatchError(f"'{choice_id}' is not one of the choices ({valid})")
        watch = self.watches.get(request["watch_id"])
        if watch["status"] != WatchStatus.AWAITING_REVIEW:
            raise WatchError(f"watch is {watch['status']}, not awaiting review")
        active = self.store.get_revision(watch["active_revision_id"])
        assert active is not None
        old_spec = WatchSpec.model_validate(active["spec"])

        combined: dict[str, Any] = {}
        if choice["effect"] == "amend":
            combined.update(choice.get("spec_patch") or {})
        combined.update(patch or {})
        new_spec = apply_spec_patch(old_spec, combined) if combined else old_spec
        diff = spec_diff(old_spec, new_spec)
        if diff and not confirm_scope_change:
            raise ConfirmationRequired(
                "this answer changes the approved scope; repeat it with explicit confirmation to apply", diff)

        effect = choice["effect"]
        target = {"pause": WatchStatus.PAUSED, "close": WatchStatus.CLOSED}.get(effect, WatchStatus.ACTIVE)
        now = self.store.now()
        revision_id = None
        with self.store.tx():
            if diff and target is WatchStatus.ACTIVE:
                revision_id = self.watches._add_revision(
                    watch["id"], active["objective"], new_spec, "checkpoint_response",
                    f"review {request['id']}: {choice['label']}" + (f" — {note}" if note else ""))
                self.store.update("watch_revision", {"id": active["id"]}, {"state": RevisionState.SUPERSEDED})
                self.store.update("watch_revision", {"id": revision_id},
                                  {"state": RevisionState.APPROVED, "decided_at": now, "decided_by": actor})
            self.store.update("checkpoint_request", {"id": request["id"]}, {"status": "ANSWERED", "answered_at": now})
            extra = {"active_revision_id": revision_id} if revision_id else {}
            self.watches._transition(watch, target, actor, "review_answered",
                                     {"request_id": request["id"], "choice": choice_id, "scope_changed": bool(revision_id)},
                                     **extra)
            response_id = random_id("resp", 10)
            self.store.insert("checkpoint_response", {
                "id": response_id, "request_id": request["id"], "watch_id": watch["id"], "choice_id": choice_id,
                "note": note, "actor": actor, "scope_change_confirmed": bool(revision_id),
                "resulting_revision_id": revision_id, "resulting_status": target,
                "detail": {"scope_diff": diff if revision_id else {}}, "at": now,
            })
        return {"request_id": request["id"], "response_id": response_id, "watch_status": str(target),
                "revision_id": revision_id, "scope_diff": diff if revision_id else {}}
