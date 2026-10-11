"""Checkpoint policies and review handling. SIMULATED web and model throughout.

These tests show that the mechanism behaves as specified. They are not evidence that
adaptive checkpoints improve research; that needs the evaluation described in
docs/CSC2524_EVALUATION.md.
"""

import json
import re

import pytest

from forage.controller.watches import ConfirmationRequired, WatchError
from forage.scheduler.queue import Scheduler
from forage.scheduler.runner import RunRefused
from forage.simulated import SimulatedLLM
from forage.types import RunStatus, WatchStatus

from conftest import OBJECTIVE
from test_scheduler import night_harness, served


def decisions(harness, watch_id):
    return harness.app.store.all(
        "SELECT policy, decision, signal, reason, signals FROM checkpoint_decision WHERE watch_id = ? ORDER BY id", (watch_id,))


def status(harness, watch_id) -> str:
    return harness.app.store.get_watch(watch_id)["status"]


def guidance_llm(sessions: set[int], question="Should the watch focus on capacity figures or on cost figures?",
                 options=("Focus on installed capacity", "Focus on cost per megawatt hour"), kind="priority"):
    """A simulated model whose end-of-session review asks for user guidance in the given session numbers."""
    counter = {"n": 0}

    def reflect(payload):
        counter["n"] += 1
        base = SimulatedLLM._reflect(payload)
        if counter["n"] in sessions:
            base["guidance"] = {"needed": True, "kind": kind, "question": question, "options": list(options),
                                "reason": "Both directions are large and the objective does not rank them."}
        return base

    return SimulatedLLM(overrides={"reflect": reflect})


# ------------------------------------------------------------ continuous


def test_continuous_policy_never_requests_review_but_logs_each_decision(harness):
    watch_id = harness.watch(policy="continuous")
    for _ in range(5):
        assert harness.run(watch_id).status is RunStatus.COMPLETED
    assert status(harness, watch_id) == WatchStatus.ACTIVE
    log = decisions(harness, watch_id)
    assert len(log) == 5 and {d["decision"] for d in log} == {"CONTINUE"} and {d["policy"] for d in log} == {"continuous"}
    assert harness.app.reviews.list(watch_id, status=None) == []


# -------------------------------------------------------------- periodic


def test_periodic_policy_requests_review_at_fixed_intervals(harness):
    watch_id = harness.watch(policy="periodic", policy_config={"every": 2})
    first = harness.run(watch_id)
    assert status(harness, watch_id) == WatchStatus.ACTIVE
    second = harness.run(watch_id)
    assert status(harness, watch_id) == WatchStatus.AWAITING_REVIEW
    assert [d["decision"] for d in decisions(harness, watch_id)] == ["CONTINUE", "REQUEST_REVIEW"]

    request = harness.app.reviews.pending_for(watch_id)
    assert request["policy"] == "periodic" and request["signal"] == "periodic_interval" and request["run_id"] == second.run_id
    assert "2 session(s) have run since the last review" in request["what_happened"]
    assert [c["id"] for c in request["choices"]] == ["continue", "amend", "pause", "close"]
    assert request["safe_default"] == "continue"
    assert [e["run_id"] for e in request["evidence"]] == [first.run_id, second.run_id]

    with pytest.raises(RunRefused, match="AWAITING_REVIEW"):
        harness.run(watch_id)  # no research while the question is open

    report = harness.report(second.run_id)
    assert report["pending_review"]["request_id"] == request["id"]
    assert "## Decision needed from you" in harness.app.store.get_report(second.run_id)["markdown"]

    result = harness.app.reviews.respond(request["id"], "continue", note="carry on")
    assert result["watch_status"] == "ACTIVE" and result["revision_id"] is None
    harness.run(watch_id)
    assert status(harness, watch_id) == WatchStatus.ACTIVE  # the interval restarts after an answer
    harness.run(watch_id)
    assert status(harness, watch_id) == WatchStatus.AWAITING_REVIEW


# -------------------------------------------------------------- adaptive


def test_adaptive_requests_review_after_sustained_lack_of_progress(harness):
    watch_id = harness.watch(policy="adaptive")
    outcomes = [harness.run(watch_id) for _ in range(3)]
    assert [o.has_meaningful_update for o in outcomes] == [True, False, False]
    assert status(harness, watch_id) == WatchStatus.ACTIVE  # two quiet sessions are not yet a stall
    harness.run(watch_id)
    assert status(harness, watch_id) == WatchStatus.AWAITING_REVIEW
    request = harness.app.reviews.pending_for(watch_id)
    assert request["signal"] == "stalled" and "verified nothing new" in request["what_happened"]
    assert [e["session"] for e in request["evidence"]] == [2, 3, 4]
    assert {c["id"] for c in request["choices"]} == {"continue", "amend", "pause", "close"}
    last = decisions(harness, watch_id)[-1]
    assert json.loads(last["signals"])["stalled"] == {"consecutive_dry_sessions": 3}


def test_adaptive_does_not_blame_the_watch_for_provider_outages(harness):
    watch_id = harness.watch(policy="adaptive")
    harness.run(watch_id)
    harness.web.rate_limit_next = 10_000  # every search is throttled from now on
    harness.app.store.execute("UPDATE source SET status = 'IRRELEVANT' WHERE status = 'DISCOVERED'")  # no saved leads to read
    for _ in range(4):
        outcome = harness.run(watch_id)
        assert outcome.has_meaningful_update is False and outcome.metrics["search_failures"] > 0
    assert status(harness, watch_id) == WatchStatus.ACTIVE
    assert {d["decision"] for d in decisions(harness, watch_id)} == {"CONTINUE"}


def test_adaptive_requests_review_when_scope_keeps_excluding_relevant_results(harness):
    watch_id = harness.watch(policy="adaptive", exclude=["installed capacity"])
    first = harness.run(watch_id)
    assert first.metrics["out_of_scope_candidates"] >= 3
    assert status(harness, watch_id) == WatchStatus.ACTIVE  # one session is not a pattern
    assert not any("installed tidal turbine capacity" in c["text"] for c in harness.app.store.list_claims(watch_id=watch_id))

    harness.web.epoch = 1  # more pages appear, again about the excluded subject
    second = harness.run(watch_id)
    assert second.metrics["out_of_scope_candidates"] >= 1
    assert status(harness, watch_id) == WatchStatus.AWAITING_REVIEW
    request = harness.app.reviews.pending_for(watch_id)
    assert request["signal"] == "scope_conflict"
    assert "will not broaden an approved objective on its own" in request["why_human"]
    assert request["evidence"] and all(e["url"].startswith("https://") and e["why_excluded"] for e in request["evidence"])
    assert request["safe_default"] == "continue"
    # The agent did not widen the scope by itself.
    _, spec = harness.app.watches.active_spec(harness.app.store.get_watch(watch_id))
    assert spec.exclude == ["installed capacity"] and len(harness.app.store.list_revisions(watch_id)) == 1


def test_scope_change_needs_explicit_confirmation_and_then_reopens_set_aside_results(harness):
    watch_id = harness.watch(policy="adaptive", exclude=["installed capacity"])
    harness.run(watch_id)
    harness.web.epoch = 1
    harness.run(watch_id)
    request = harness.app.reviews.pending_for(watch_id)
    before = harness.app.store.get_watch(watch_id)["active_revision_id"]

    with pytest.raises(ConfirmationRequired) as excinfo:
        harness.app.reviews.respond(request["id"], "amend", patch={"exclude_remove": ["installed capacity"]})
    assert excinfo.value.preview == {"exclude": {"added": [], "removed": ["installed capacity"]}}
    assert status(harness, watch_id) == WatchStatus.AWAITING_REVIEW  # nothing changed
    assert harness.app.store.get_watch(watch_id)["active_revision_id"] == before
    assert harness.app.reviews.pending_for(watch_id)["status"] == "PENDING"

    result = harness.app.reviews.respond(request["id"], "amend", patch={"exclude_remove": ["installed capacity"]},
                                         note="capacity is in scope after all", confirm_scope_change=True)
    assert result["watch_status"] == "ACTIVE" and result["revision_id"] and result["scope_diff"]
    watch = harness.app.store.get_watch(watch_id)
    assert watch["status"] == WatchStatus.ACTIVE and watch["active_revision_id"] == result["revision_id"]
    revisions = harness.app.store.list_revisions(watch_id)
    assert [(r["revision_no"], r["state"], r["origin"]) for r in revisions] == [
        (1, "SUPERSEDED", "user"), (2, "APPROVED", "checkpoint_response")]
    assert revisions[1]["decided_by"] == "user" and request["id"] in revisions[1]["note"]
    assert revisions[1]["objective"] == OBJECTIVE  # history kept; the objective text itself is unchanged
    response = harness.app.store.one("SELECT * FROM checkpoint_response WHERE request_id = ?", (request["id"],))
    assert response["choice_id"] == "amend" and response["scope_change_confirmed"] == 1
    assert response["resulting_revision_id"] == result["revision_id"]

    reopened = harness.app.store.list_sources(watch_id, ["DISCOVERED"])
    assert len(reopened) >= 3  # results set aside under the old scope are leads again
    third = harness.run(watch_id)
    assert third.metrics["backlog_sources_read"] >= 1 and third.metrics["findings_new"] >= 1
    assert any("installed tidal turbine capacity" in c["text"] for c in harness.app.store.list_claims(watch_id=watch_id))


def test_answering_continue_leaves_scope_untouched_and_is_not_asked_again_immediately(harness):
    watch_id = harness.watch(policy="adaptive", exclude=["installed capacity"])
    harness.run(watch_id)
    harness.web.epoch = 1
    harness.run(watch_id)
    request = harness.app.reviews.pending_for(watch_id)
    result = harness.app.reviews.respond(request["id"], "continue")
    assert result == {"request_id": request["id"], "response_id": result["response_id"], "watch_status": "ACTIVE",
                      "revision_id": None, "scope_diff": {}}
    assert len(harness.app.store.list_revisions(watch_id)) == 1
    harness.run(watch_id)
    assert status(harness, watch_id) == WatchStatus.ACTIVE
    assert "not asking again yet" in decisions(harness, watch_id)[-1]["reason"]


def test_review_can_be_answered_with_pause_or_close(harness):
    for choice, expected in (("pause", WatchStatus.PAUSED), ("close", WatchStatus.CLOSED)):
        watch_id = harness.watch(f"{OBJECTIVE} ({choice})", policy="periodic", policy_config={"every": 1})
        harness.run(watch_id)
        request = harness.app.reviews.pending_for(watch_id)
        assert harness.app.reviews.respond(request["id"], choice)["watch_status"] == expected
        assert status(harness, watch_id) == expected
        with pytest.raises(WatchError, match="already ANSWERED"):
            harness.app.reviews.respond(request["id"], "continue")


def test_closing_a_watch_withdraws_its_pending_review(harness):
    watch_id = harness.watch(policy="periodic", policy_config={"every": 1})
    harness.run(watch_id)
    request = harness.app.reviews.pending_for(watch_id)
    harness.app.watches.close(watch_id)
    assert harness.app.reviews.get(request["id"])["status"] == "WITHDRAWN"
    assert harness.app.reviews.pending_for(watch_id) is None


def test_invalid_review_answers_are_refused(harness):
    watch_id = harness.watch(policy="periodic", policy_config={"every": 1})
    harness.run(watch_id)
    request = harness.app.reviews.pending_for(watch_id)
    with pytest.raises(WatchError, match="not one of the choices"):
        harness.app.reviews.respond(request["id"], "broaden-everything")
    with pytest.raises(WatchError, match="the change must be stated"):
        harness.app.reviews.respond(request["id"], "amend", confirm_scope_change=True)
    assert status(harness, watch_id) == WatchStatus.AWAITING_REVIEW


# ------------------------------------- model-proposed guidance and its gate


def test_single_uncorroborated_model_proposal_is_declined(make_harness):
    harness = make_harness(guidance_llm({2}))
    watch_id = harness.watch(policy="adaptive")
    harness.run(watch_id)
    harness.web.epoch = 1  # session 2 finds new material, so nothing else is wrong
    second = harness.run(watch_id)
    assert second.guidance is not None and second.has_meaningful_update
    assert status(harness, watch_id) == WatchStatus.ACTIVE
    last = decisions(harness, watch_id)[-1]
    assert last["decision"] == "CONTINUE"
    assert json.loads(last["signals"])["user_decision"]["gate"] == "declined: proposed once with no corroborating signal"


def test_recurring_model_proposal_becomes_a_review_with_actionable_choices(make_harness):
    harness = make_harness(guidance_llm({2, 3}))
    watch_id = harness.watch(policy="adaptive")
    harness.run(watch_id)
    harness.web.epoch = 1
    harness.run(watch_id)
    assert status(harness, watch_id) == WatchStatus.ACTIVE
    harness.run(watch_id)
    assert status(harness, watch_id) == WatchStatus.AWAITING_REVIEW
    request = harness.app.reviews.pending_for(watch_id)
    assert request["signal"] == "user_decision"
    assert request["question"] == "Should the watch focus on capacity figures or on cost figures?"
    assert "written by the local model" in request["why_human"]
    options = [c for c in request["choices"] if c["id"].startswith("option")]
    assert [c["label"] for c in options] == ["Focus on installed capacity", "Focus on cost per megawatt hour"]
    assert options[0]["effect"] == "amend" and options[0]["spec_patch"] == {"priorities_add": ["Focus on installed capacity"]}

    with pytest.raises(ConfirmationRequired):
        harness.app.reviews.respond(request["id"], "option2")  # picking an option changes priorities: confirm first
    harness.app.reviews.respond(request["id"], "option2", confirm_scope_change=True)
    _, spec = harness.app.watches.active_spec(harness.app.store.get_watch(watch_id))
    assert spec.priorities == ["Focus on cost per megawatt hour"]


def test_model_proposal_in_the_first_session_is_too_early(make_harness):
    harness = make_harness(guidance_llm({1, 2}))
    watch_id = harness.watch(policy="adaptive")
    harness.run(watch_id)
    assert status(harness, watch_id) == WatchStatus.ACTIVE
    assert "too early to interrupt" in decisions(harness, watch_id)[-1]["reason"]


def test_factual_questions_from_the_model_do_not_interrupt_the_user(make_harness):
    harness = make_harness(guidance_llm({2, 3}, question="What is the installed tidal capacity in Wales in 2027?",
                                        options=("30 MW", "60 MW")))
    watch_id = harness.watch(policy="adaptive")
    harness.run(watch_id)
    harness.web.epoch = 1
    harness.run(watch_id)
    harness.run(watch_id)
    assert status(harness, watch_id) == WatchStatus.ACTIVE
    gate = json.loads(decisions(harness, watch_id)[-1]["signals"])["user_decision"]["gate"]
    assert gate == "declined: reads like a missing fact, which more research can address"


def test_policies_other_than_adaptive_ignore_model_proposals(make_harness):
    for policy in ("continuous", "periodic"):
        harness = make_harness(guidance_llm({1, 2, 3, 4}))
        watch_id = harness.watch(policy=policy, policy_config={"every": 99})
        for _ in range(4):
            outcome = harness.run(watch_id)
            assert outcome.guidance is not None  # the engine reports the proposal identically under every policy
        assert status(harness, watch_id) == WatchStatus.ACTIVE


# ------------------------------------------------------ shared behaviour


def test_hard_limit_gate_applies_to_every_policy_and_is_labelled_common(make_harness):
    for policy in ("continuous", "periodic", "adaptive"):
        harness = make_harness()
        watch_id = harness.watch(policy=policy, policy_config={"every": 99}, budget={"max_total_sessions": 2})
        harness.run(watch_id)
        assert status(harness, watch_id) == WatchStatus.ACTIVE
        harness.run(watch_id)
        assert status(harness, watch_id) == WatchStatus.AWAITING_REVIEW, policy
        request = harness.app.reviews.pending_for(watch_id)
        assert request["signal"] == "common_hard_limit" and request["policy"] == policy
        assert request["safe_default"] == "pause"


def test_the_engine_behaves_identically_under_every_policy(make_harness):
    """The policy is the only experimental variable: same searches, pages, model calls and findings."""
    traces = {}
    for policy in ("continuous", "periodic", "adaptive"):
        harness = make_harness()
        watch_id = harness.watch(policy=policy, policy_config={"every": 99})
        for _ in range(2):
            harness.run(watch_id)
        traces[policy] = {
            "searches": list(harness.web.search_log),
            "fetches": list(harness.web.fetch_log),
            # Identifiers are derived from the (random) watch id, so they are masked before comparing.
            "model_calls": [(purpose, re.sub(r"\b[qfsc]-[0-9a-f]{8,}\b", "<id>", call["user"]))
                            for purpose, call in harness.llm.calls],
            "findings": sorted(f["title"] for f in harness.app.store.list_findings(watch_id)),
            "claims": sorted(c["text"] for c in harness.app.store.list_claims(watch_id=watch_id)),
            "budgets": [r["budget"] for r in harness.app.store.list_runs(watch_id)],
        }
    assert traces["continuous"] == traces["periodic"] == traces["adaptive"]
    assert traces["continuous"]["searches"] and traces["continuous"]["findings"]


# ----------------------------- one watch waits for review, another continues


def test_awaiting_review_watch_yields_its_turns_to_other_watches(make_harness):
    harness = night_harness(make_harness)
    blocked = harness.watch("Objective alpha: installed tidal turbine capacity and tidal energy cost",
                            policy="adaptive", exclude=["installed capacity"])
    other = harness.watch("Objective beta: installed tidal turbine capacity and tidal energy cost", policy="adaptive")
    scheduler: Scheduler = harness.scheduler

    scheduler.run_batch(max_sessions=2)
    harness.web.epoch = 1  # new pages appear, some again inside the excluded subject
    scheduler.run_batch(max_sessions=2)
    assert served(harness) == [blocked, other, blocked, other]
    assert status(harness, blocked) == WatchStatus.AWAITING_REVIEW and status(harness, other) == WatchStatus.ACTIVE
    request = harness.app.reviews.pending_for(blocked)
    assert request["signal"] == "scope_conflict"
    assert harness.app.store.scalar("SELECT COUNT(*) FROM outbox WHERE kind = 'REVIEW_REQUEST' AND watch_id = ?", (blocked,)) == 1

    # While the question is open, every further turn goes to the other watch.
    scheduler.run_batch(max_sessions=3)
    assert served(harness)[4:] == [other, other, other]  # all three turns, none for the waiting watch
    passed_over = harness.app.store.all(
        "SELECT reason FROM queue_admission WHERE decision = 'SKIPPED' AND watch_id = ?", (blocked,))
    assert passed_over and {row["reason"] for row in passed_over} == {"status AWAITING_REVIEW"}
    assert len(harness.app.store.list_runs(blocked)) == 2  # no compute spent on the blocked watch

    # The user's confirmed answer changes the approved scope and returns the watch to the queue.
    harness.app.reviews.respond(request["id"], "amend", patch={"exclude_remove": ["installed capacity"]},
                                confirm_scope_change=True)
    assert status(harness, blocked) == WatchStatus.ACTIVE
    eligible, reason, _ = scheduler.eligibility(harness.app.store.get_watch(blocked), harness.clock())
    assert eligible and reason == "eligible"
    before = len(harness.app.store.list_runs(blocked))
    harness.app.runner.run_watch(blocked)
    assert len(harness.app.store.list_runs(blocked)) == before + 1
    assert harness.app.store.list_runs(blocked)[-1]["revision_id"].endswith(".r2")  # researched under the new brief
