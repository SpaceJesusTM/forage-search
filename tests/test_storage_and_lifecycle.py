"""Persistence and watch-control tests. SIMULATED adapters only."""

import sqlite3
from datetime import timedelta

import pytest

from forage.controller.watches import InvalidTransition, WatchError, apply_spec_patch, spec_diff
from forage.scheduler.runner import RunRefused
from forage.storage.store import Store
from forage.types import ChangeType, RunStatus, WatchSpec, WatchStatus

# --------------------------------------------------------------- storage


def test_migrations_apply_once_and_are_recorded(tmp_path):
    from forage.storage.db import _migration_files

    latest = _migration_files()[-1][0]
    path = tmp_path / "f.sqlite3"
    store = Store(path)
    assert store.schema_version() == latest >= 3
    assert store.migrate() == []  # nothing left to apply
    store.close()
    again = Store(path)
    assert again.schema_version() == latest
    names = {row["name"] for row in again.all("SELECT name FROM sqlite_master WHERE type = 'table'")}
    for table in ("watch", "watch_revision", "session_run", "source", "source_snapshot", "claim", "finding", "report",
                  "search_attempt", "open_question", "checkpoint_request", "checkpoint_response", "queue_state", "outbox"):
        assert table in names
    again.close()


def test_transaction_rolls_back_on_error(tmp_path):
    store = Store(tmp_path / "f.sqlite3")
    with pytest.raises(RuntimeError):
        with store.tx():
            store.add_event(None, None, "inside", {})
            raise RuntimeError("boom")
    assert store.scalar("SELECT COUNT(*) FROM run_event") == 0


def test_only_one_session_can_be_running(harness):
    first = harness.watch()
    second = harness.watch("What is the levelised cost of tidal energy over time?")
    harness.app.runner.admit(first, "manual")
    with pytest.raises(RunRefused, match="one session runs at a time"):
        harness.app.runner.admit(second, "manual")
    # The database enforces it too, independently of the manager's own check.
    with pytest.raises(sqlite3.IntegrityError):
        harness.app.store.insert("session_run", {
            "id": "rogue", "watch_id": second, "revision_id": "x", "seq": 1, "status": RunStatus.RUNNING,
            "trigger": "manual", "policy": "continuous", "budget": {}, "started_at": "t", "heartbeat_at": "t"})


def test_run_ids_are_single_use(harness):
    watch_id = harness.watch()
    harness.app.runner.run_watch(watch_id, run_id="caller-issued-1")
    with pytest.raises(RunRefused, match="already exists"):
        harness.app.runner.run_watch(watch_id, run_id="caller-issued-1")


def test_stale_running_session_is_recovered_with_a_report(harness):
    watch_id = harness.watch()
    run = harness.app.runner.admit(watch_id, "manual")  # process "dies" before the engine finishes
    harness.clock.advance(minutes=5)
    assert harness.app.runner.recover() == []  # heartbeat still fresh
    harness.clock.advance(hours=1)
    assert harness.app.runner.recover() == [run["id"]]
    recovered = harness.app.store.get_run(run["id"])
    assert recovered["status"] == RunStatus.INTERRUPTED and "no heartbeat" in recovered["error"]
    assert harness.report(run["id"])["status"] == "interrupted"
    # The watch is researchable again.
    assert harness.run(watch_id).status is RunStatus.COMPLETED


def test_change_log_keeps_the_strongest_change_and_merges_details(tmp_path):
    store = Store(tmp_path / "f.sqlite3")
    store.record_change("run1", "w", "f1", ChangeType.UNCHANGED, {"seen_again_in": ["s1"]})
    store.record_change("run1", "w", "f1", ChangeType.UPDATED, {"new_claim_ids": ["c1"]})
    store.record_change("run1", "w", "f1", ChangeType.CORROBORATED, {"new_claim_ids": ["c2"]})
    (change,) = store.list_changes("run1")
    assert change["change_type"] == ChangeType.UPDATED
    assert change["detail"] == {"seen_again_in": ["s1"], "new_claim_ids": ["c1", "c2"]}


# ------------------------------------------------------------- lifecycle


def test_new_watch_is_not_researchable_until_confirmed(harness):
    watch_id = harness.watch(approve=False)
    watch = harness.app.store.get_watch(watch_id)
    assert watch["status"] == WatchStatus.PENDING_CONFIRMATION
    assert watch["active_revision_id"] is None and watch["pending_revision_id"]
    with pytest.raises(RunRefused, match="PENDING_CONFIRMATION"):
        harness.run(watch_id)
    assert harness.web.search_log == [] and harness.web.fetch_log == []

    approved = harness.app.watches.approve(watch_id)
    assert approved["status"] == WatchStatus.ACTIVE and approved["pending_revision_id"] is None
    revision = harness.app.store.get_revision(approved["active_revision_id"])
    assert revision["state"] == "APPROVED" and revision["decided_by"] == "user"


def test_history_records_every_transition_with_actor(harness):
    watch_id = harness.watch()
    harness.app.watches.pause(watch_id)
    kinds = [(e["kind"], e["from_status"], e["to_status"], e["actor"]) for e in harness.app.store.list_watch_events(watch_id)]
    assert kinds == [
        ("created", None, "DRAFT", "user"),
        ("revision_proposed", "DRAFT", "PENDING_CONFIRMATION", "user"),
        ("approved", "PENDING_CONFIRMATION", "ACTIVE", "user"),
        ("paused", "ACTIVE", "PAUSED", "user"),
    ]


def test_pause_is_immediate_and_blocks_research(harness):
    watch_id = harness.watch()
    harness.app.watches.pause(watch_id)
    with pytest.raises(RunRefused, match="PAUSED"):
        harness.run(watch_id)


def test_resume_requires_a_confirmed_plan(harness):
    watch_id = harness.watch()
    harness.run(watch_id)
    harness.app.watches.pause(watch_id)
    proposed = harness.app.watches.request_resume(watch_id, patch={"exclude_add": ["wave energy"]})
    assert proposed["status"] == WatchStatus.REVIEW_PROPOSED
    with pytest.raises(RunRefused):
        harness.run(watch_id)  # a proposed plan is not permission
    plan = harness.app.watches.resume_plan(watch_id)
    assert plan["sessions_run"] == 1 and plan["findings"] >= 1
    assert plan["scope_changes_in_plan"] == {"exclude": {"added": ["wave energy"], "removed": []}}

    resumed = harness.app.watches.approve(watch_id)
    assert resumed["status"] == WatchStatus.ACTIVE
    _, spec = harness.app.watches.active_spec(resumed)
    assert spec.exclude == ["wave energy"]
    states = [(r["revision_no"], r["state"], r["origin"]) for r in harness.app.store.list_revisions(watch_id)]
    assert states == [(1, "SUPERSEDED", "user"), (2, "APPROVED", "resume_plan")]


def test_declining_a_resume_plan_returns_to_the_previous_state(harness):
    watch_id = harness.watch()
    harness.app.watches.pause(watch_id)
    harness.app.watches.request_resume(watch_id)
    declined = harness.app.watches.decline(watch_id)
    assert declined["status"] == WatchStatus.PAUSED and declined["pending_revision_id"] is None
    assert harness.app.store.list_revisions(watch_id)[-1]["state"] == "REJECTED"


def test_close_archives_without_deleting_and_reopen_needs_confirmation(harness):
    watch_id = harness.watch()
    harness.run(watch_id)
    findings_before = len(harness.app.store.list_findings(watch_id))
    assert findings_before > 0
    closed = harness.app.watches.close(watch_id)
    assert closed["status"] == WatchStatus.CLOSED
    assert len(harness.app.store.list_findings(watch_id)) == findings_before  # history preserved
    with pytest.raises(RunRefused):
        harness.run(watch_id)
    harness.app.watches.request_resume(watch_id)
    assert harness.app.store.get_watch(watch_id)["status"] == WatchStatus.REVIEW_PROPOSED
    harness.app.watches.approve(watch_id)
    assert harness.run(watch_id).status is RunStatus.COMPLETED


@pytest.mark.parametrize("action", ["pause", "request_resume"])
def test_invalid_transitions_are_refused(harness, action):
    watch_id = harness.watch(approve=False)  # PENDING_CONFIRMATION
    with pytest.raises((InvalidTransition, WatchError)):
        getattr(harness.app.watches, action)(watch_id)
    assert harness.app.store.get_watch(watch_id)["status"] == WatchStatus.PENDING_CONFIRMATION


def test_amending_an_active_watch_takes_effect_only_after_approval(harness):
    watch_id = harness.watch()
    harness.app.watches.propose_revision(watch_id, patch={"include_add": ["tidal lagoons"]})
    watch = harness.app.store.get_watch(watch_id)
    assert watch["status"] == WatchStatus.ACTIVE
    assert harness.app.watches.active_spec(watch)[1].include == []  # still the old brief
    harness.app.watches.approve(watch_id)
    assert harness.app.watches.active_spec(harness.app.store.get_watch(watch_id))[1].include == ["tidal lagoons"]


def test_model_drafted_brief_is_marked_as_a_proposal(make_harness):
    harness = make_harness()
    watch = harness.app.watches.create("Find tidal energy internships in Toronto for 2027", llm=harness.llm)
    assert watch["status"] == WatchStatus.PENDING_CONFIRMATION
    revision = harness.app.store.get_revision(watch["pending_revision_id"])
    assert revision["origin"] == "model_draft" and revision["state"] == "PROPOSED"
    assert "needs user confirmation" in revision["note"]
    assert revision["objective"] == "Find tidal energy internships in Toronto for 2027"  # the user's words are kept verbatim


def test_spec_patch_and_diff():
    spec = WatchSpec(include=["a", "b"], exclude=["x"])
    patched = apply_spec_patch(spec, {"include_add": ["c", "A"], "include_remove": ["b"], "exclude_add": ["y"], "geography": "Toronto"})
    assert patched.include == ["a", "c"] and patched.exclude == ["x", "y"] and patched.geography == "Toronto"
    assert spec.include == ["a", "b"]  # original untouched
    assert spec_diff(spec, patched) == {
        "include": {"added": ["c"], "removed": ["b"]}, "exclude": {"added": ["y"], "removed": []},
        "geography": {"from": None, "to": "Toronto"}}
    assert spec_diff(spec, spec) == {}


def test_state_survives_process_restart(harness):
    watch_id = harness.watch()
    first = harness.run(watch_id)
    before = {
        "findings": [f["id"] for f in harness.app.store.list_findings(watch_id)],
        "claims": len(harness.app.store.list_claims(watch_id=watch_id)),
        "queries": [a["query"] for a in harness.app.store.list_search_attempts(watch_id)],
        "report": harness.app.store.get_report(first.run_id)["json"],
    }
    app = harness.reopen()  # new Store object on the same file, fresh engine and adapters
    assert [f["id"] for f in app.store.list_findings(watch_id)] == before["findings"]
    assert len(app.store.list_claims(watch_id=watch_id)) == before["claims"]
    assert app.store.get_report(first.run_id)["json"] == before["report"]
    assert app.store.get_watch(watch_id)["status"] == WatchStatus.ACTIVE

    harness.web.epoch = 1
    harness.clock.advance(days=1)
    second = harness.run(watch_id)
    report = harness.report(second.run_id)
    assert report["session_number"] == 2
    assert report["comparison"]["prior_sessions"] == 1
    assert report["comparison"]["known_findings_before"] == len(before["findings"])
    assert report["comparison"]["queries_already_tried_before"] == len(before["queries"])
    assert not set(a["query"] for a in app.store.list_search_attempts(watch_id, second.run_id)) & set(before["queries"])
