"""Periodic baseline: ask for a review every N completed sessions, whatever happened in them."""

from __future__ import annotations

from forage.checkpoint.base import REQUEST_REVIEW, CheckpointContext, CheckpointDecision, standard_choices

DEFAULT_EVERY = 3


class PeriodicPolicy:
    name = "periodic"

    def evaluate(self, context: CheckpointContext) -> CheckpointDecision:
        every = max(1, int(context.config.get("every", DEFAULT_EVERY)))
        signals = {"sessions_since_review": context.sessions_since_review, "every": every}
        if context.sessions_since_review < every:
            return CheckpointDecision.carry_on(
                f"{context.sessions_since_review} of {every} sessions since the last review", signals)
        recent = context.recent_runs[-context.sessions_since_review:]
        new = sum((r["metrics"] or {}).get("findings_new", 0) for r in recent)
        updated = sum((r["metrics"] or {}).get("findings_updated", 0) for r in recent)
        answered = sum((r["metrics"] or {}).get("questions_resolved", 0) for r in recent)
        return CheckpointDecision(
            action=REQUEST_REVIEW,
            reason=f"scheduled review after {context.sessions_since_review} sessions",
            signal="periodic_interval",
            signals=signals,
            what_happened=(f"{len(recent)} session(s) have run since the last review: {new} new finding(s), "
                           f"{updated} updated, {answered} question(s) answered."),
            why_human=f"This watch is set to pause for your review every {every} session(s).",
            question="Should this watch continue with its current objective and scope?",
            choices=standard_choices(),
            safe_default="continue",
            urgency="low",
            evidence=[{"session": r["seq"], "run_id": r["id"], "new": (r["metrics"] or {}).get("findings_new", 0),
                       "updated": (r["metrics"] or {}).get("findings_updated", 0),
                       "meaningful_update": bool(r["has_meaningful_update"])} for r in recent],
        )
