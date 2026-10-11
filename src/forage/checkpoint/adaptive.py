"""Adaptive policy (the candidate contribution): ask only when a decision plausibly belongs to the user.

Three signals, all computed from stored state so every decision can be replayed:

- scope_conflict: sessions keep surfacing relevant-looking results that the approved
  scope excludes. Continuing productively may need a scope decision.
- user_decision: the local model proposed a preference/scope/strategy question, and that
  proposal passed deterministic gates (it recurs or is corroborated, it reads like a
  decision rather than a missing fact, and it is not too soon after the last review).
- stalled: several consecutive sessions searched successfully and verified nothing new.

What deliberately does NOT trigger a review: missing facts, model uncertainty, provider
failures, or a single quiet session. More research inside the approved scope can address
those, and interrupting for them would be a premature checkpoint.

This is a hypothesis under evaluation, not an established improvement.
"""

from __future__ import annotations

import re
from typing import Any

from forage.checkpoint.base import (
    REQUEST_REVIEW, CheckpointContext, CheckpointDecision, Choice, standard_choices,
)

DEFAULTS = {
    "min_sessions_before_review": 2,  # never interrupt before the research has had a real attempt
    "min_sessions_between_reviews": 2,  # do not nag right after an answer
    "stall_sessions": 3,  # consecutive searched-but-empty sessions that count as stalled
    "scope_conflict_min_examples": 3,  # out-of-scope candidates needed across recent sessions
    "scope_conflict_min_sessions": 2,  # ...seen in at least this many sessions
    "guidance_recurrence": 2,  # model proposals within the lookback that count as recurring
    "lookback_sessions": 3,
}

# A question for the user is about what they want, not about what is true.
_DECISION_CUES = re.compile(
    r"\b(should|would you|do you|prefer|want|priorit|focus|include|exclude|broaden|narrow|expand|limit|continue|"
    r"keep|drop|switch|instead|which (?:direction|option|approach))", re.I)


def _metrics(run: dict[str, Any]) -> dict[str, Any]:
    return run.get("metrics") or {}


def _searched_ok(run: dict[str, Any]) -> bool:
    m = _metrics(run)
    return m.get("search_requests", 0) > m.get("search_failures", 0) or m.get("pages_read", 0) > 0


class AdaptivePolicy:
    name = "adaptive"

    def evaluate(self, context: CheckpointContext) -> CheckpointDecision:
        cfg = DEFAULTS | {k: v for k, v in context.config.items() if k in DEFAULTS}
        since = context.recent_runs[-context.sessions_since_review:] if context.sessions_since_review else []
        lookback = since[-int(cfg["lookback_sessions"]):]
        signals: dict[str, Any] = {
            "completed_sessions": context.completed_sessions,
            "sessions_since_review": context.sessions_since_review,
            "reviews_answered": context.reviews_answered,
        }

        # -- measure every signal first, so declined checkpoints are logged with their numbers
        scope_sessions = [r for r in lookback if _metrics(r).get("out_of_scope_candidates", 0) > 0]
        scope_examples = [ex for r in lookback for ex in _metrics(r).get("out_of_scope_examples", [])]
        scope_total = sum(_metrics(r).get("out_of_scope_candidates", 0) for r in lookback)
        signals["scope_conflict"] = {"examples": scope_total, "sessions": len(scope_sessions)}

        dry_streak = 0
        for run in reversed(since):
            if run["has_meaningful_update"] or not _searched_ok(run):
                break
            dry_streak += 1
        signals["stalled"] = {"consecutive_dry_sessions": dry_streak}

        proposal = context.guidance if (context.guidance or {}).get("needed") else None
        proposals = [r for r in lookback if (_metrics(r).get("guidance_proposed") or {}).get("needed")]
        signals["user_decision"] = {"proposed_now": bool(proposal), "proposals_in_lookback": len(proposals)}

        # -- gates common to every adaptive signal
        if context.completed_sessions < cfg["min_sessions_before_review"]:
            return CheckpointDecision.carry_on(
                f"only {context.completed_sessions} session(s) so far; too early to interrupt", signals)
        if context.reviews_answered and context.sessions_since_review < cfg["min_sessions_between_reviews"]:
            return CheckpointDecision.carry_on(
                f"{context.sessions_since_review} session(s) since the last answered review; not asking again yet", signals)

        # -- signal 1: scope conflict
        if scope_total >= cfg["scope_conflict_min_examples"] and len(scope_sessions) >= cfg["scope_conflict_min_sessions"]:
            shown = scope_examples[:5]
            return CheckpointDecision(
                action=REQUEST_REVIEW,
                reason=f"{scope_total} relevant-looking results were out of scope across {len(scope_sessions)} sessions",
                signal="scope_conflict", signals=signals,
                what_happened=(f"Over the last {len(lookback)} session(s), {scope_total} search results looked relevant to "
                               "the objective but were set aside because the approved scope excludes them."),
                why_human=("Whether to widen the scope is your decision. The agent will not broaden an approved "
                           "objective on its own, and more searching inside the current scope cannot settle this."),
                question="Results outside the approved scope keep appearing. Keep the scope as it is, or change it?",
                choices=standard_choices([Choice(
                    "amend", "Change the scope (state the change with --include / --exclude / --remove-exclude)", "amend")]),
                safe_default="continue", urgency="normal",
                evidence=[{"title": ex.get("title"), "url": ex.get("url"), "why_excluded": ex.get("reason")} for ex in shown],
            )

        # -- signal 2: a decision the model says only the user can make
        if proposal:
            verdict = self._gate_proposal(proposal, len(proposals), scope_total, dry_streak, cfg)
            signals["user_decision"]["gate"] = verdict
            if verdict == "passed":
                field = "include_add" if proposal.get("kind") == "scope" else "priorities_add"
                options = [Choice(f"option{i + 1}", str(option), "amend", {field: [str(option)]})
                           for i, option in enumerate(proposal.get("options", [])[:3])]
                return CheckpointDecision(
                    action=REQUEST_REVIEW,
                    reason=f"model-proposed {proposal.get('kind', 'other')} decision passed the gate",
                    signal="user_decision", signals=signals,
                    what_happened=("At the end of this session the research model reported that continuing well depends "
                                   "on a choice it cannot make from evidence."),
                    why_human=(str(proposal.get("reason") or "The choice depends on your preferences.")
                               + " (This explanation was written by the local model.)"),
                    question=str(proposal["question"]),
                    choices=standard_choices(options),
                    safe_default="continue", urgency="normal",
                    evidence=[{"proposed_in_sessions": [r["seq"] for r in proposals], "kind": proposal.get("kind")}],
                )

        # -- signal 3: sustained lack of progress
        if dry_streak >= cfg["stall_sessions"]:
            stalled = since[-dry_streak:]
            return CheckpointDecision(
                action=REQUEST_REVIEW,
                reason=f"{dry_streak} consecutive sessions searched successfully and verified nothing new",
                signal="stalled", signals=signals,
                what_happened=(f"The last {dry_streak} session(s) ran their searches and read pages but verified nothing "
                               "new, updated or newly answered."),
                why_human=("Further sessions on the same plan are unlikely to pay off. Whether to redirect the watch, "
                           "let it idle, or stop is a judgement about what you still want from it."),
                question="This watch has stopped producing new findings. Continue as is, change direction, pause or close?",
                choices=standard_choices([Choice(
                    "amend", "Change direction (state the change with --question / --include / --notes)", "amend")]),
                safe_default="continue", urgency="low",
                evidence=[{"session": r["seq"], "searches": _metrics(r).get("search_requests", 0),
                           "pages_read": _metrics(r).get("pages_read", 0),
                           "known_urls_skipped": _metrics(r).get("duplicate_urls_skipped", 0)} for r in stalled],
            )

        return CheckpointDecision.carry_on("no signal met its threshold", signals)

    @staticmethod
    def _gate_proposal(proposal: dict[str, Any], recurring: int, scope_total: int, dry_streak: int,
                       cfg: dict[str, Any]) -> str:
        """Why a model-proposed checkpoint is accepted or declined. A small model over-asks; this keeps it honest."""
        question = str(proposal.get("question") or "")
        if len(question) < 15 or len(proposal.get("options") or []) < 2:
            return "declined: no precise question with at least two options"
        if not _DECISION_CUES.search(question + " " + " ".join(map(str, proposal.get("options", [])))):
            return "declined: reads like a missing fact, which more research can address"
        if recurring >= cfg["guidance_recurrence"]:
            return "passed"
        if scope_total > 0 or dry_streak > 0:
            return "passed"  # a one-off proposal backed by an independent signal
        return "declined: proposed once with no corroborating signal"
