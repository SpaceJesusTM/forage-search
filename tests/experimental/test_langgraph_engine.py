"""EXPERIMENTAL: the step-driven session prototypes. SIMULATED web and model throughout.

Every test that is not specific to LangGraph runs twice: once with the LangGraph driver and
once with the plain-loop driver over the same steps. Skipped when LangGraph is not installed.
These tests back the comparison in docs/LANGGRAPH_ARCHITECTURE_DECISION.md.
"""

import json
import sqlite3
import sys
from pathlib import Path

import importlib.util

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from conftest import Harness  # noqa: E402
from experiments.langgraph.faults import Faults, SimulatedCrash  # noqa: E402
from experiments.langgraph import trace  # noqa: E402
from experiments.langgraph.trace import collect, first_difference  # noqa: E402
from forage.experimental.langgraph_engine import NODE_NAMES, StepSession, use_engine  # noqa: E402
from forage.experimental.langgraph_engine.nodes import STEPS as _STEPS, EngineContext  # noqa: E402
from forage.simulated import SimulatedLLM  # noqa: E402
from forage.types import RunStatus, WatchStatus  # noqa: E402


DRIVER = "langgraph"


HAVE_LANGGRAPH = all(importlib.util.find_spec(name) for name in ("langgraph", "aiosqlite")) and \
    importlib.util.find_spec("langgraph.checkpoint.sqlite") is not None


@pytest.fixture(params=["langgraph", "plain"], autouse=True)
def driver(request):
    global DRIVER
    if request.param == "langgraph" and not HAVE_LANGGRAPH:
        pytest.skip("LangGraph is not installed (pip install 'forage-search[langgraph]')")
    DRIVER = request.param
    return request.param


def langgraph_only() -> None:
    if DRIVER != "langgraph":
        pytest.skip("exercises a LangGraph facility the plain loop does not have")


def graph_harness(tmp_path, name="graph", llm=None, engine_options=None, **settings) -> Harness:
    harness = Harness(tmp_path / name, llm, **settings)
    harness.engine_kind, harness.engine_options = DRIVER, engine_options or {}
    harness.graph_engine = use_engine(harness.app, DRIVER, **harness.engine_options)
    return harness


def restart(harness: Harness):
    """A new process: every in-memory object is rebuilt from the two SQLite files."""
    harness.graph_engine.close()
    harness.app.close()
    harness.open()
    harness.graph_engine = use_engine(harness.app, harness.engine_kind, **harness.engine_options)
    return harness.graph_engine


def research_state(harness: Harness, watch_id: str) -> dict:
    return trace.research_state(harness.app.store, watch_id)


# ---------------------------------------------- 1. same research, graph-level steps


def scenario(harness: Harness) -> str:
    watch_id = harness.watch(freshness_days=7)
    for epoch in range(4):
        harness.web.epoch = epoch
        harness.run(watch_id)
        harness.clock.advance(days=8)
    return watch_id


def test_graph_engine_does_exactly_the_same_research_as_the_linear_engine(tmp_path):
    linear = Harness(tmp_path / "linear")
    graph = graph_harness(tmp_path)
    traces = [collect(h.app, h.web, h.llm, [scenario(h)]) for h in (linear, graph)]
    assert first_difference(*traces) is None  # searches, fetches, prompts, claims, events, metrics, reports
    assert len(traces[0]["searches"]) > 8 and len(traces[0]["model_calls"]) > 30
    linear.close(), graph.close()


def test_session_runs_as_separate_checkpointed_graph_steps(tmp_path):
    harness = graph_harness(tmp_path)
    watch_id = harness.watch()
    outcome = harness.run(watch_id)
    visited = [nodes[0] for _, nodes in harness.graph_engine.history(outcome.run_id) if nodes and nodes[0] != "__start__"]
    assert visited[:7] == ["recall", "dispatch", "plan", "dispatch", "search", "triage", "dispatch"]
    assert visited[-2:] == ["reflect", "report"]
    for step in ("fetch", "extract", "integrate"):
        assert visited.count(step) >= 3
    assert set(visited) <= set(NODE_NAMES)
    assert harness.graph_engine.history(outcome.run_id)[-1][1] == ()  # the session ran to its end
    assert not harness.graph_engine.can_resume(outcome.run_id)
    harness.close()


def test_checkpoints_hold_references_not_documents(tmp_path):
    langgraph_only()
    harness = graph_harness(tmp_path)
    watch_id = harness.watch()
    outcome = harness.run(watch_id)
    values = harness.graph_engine.snapshot(outcome.run_id).values
    assert set(values) <= {
        "run_id", "recalled", "phase", "route", "recheck_ids", "lead_ids", "leads_read", "plan", "query_index", "quota",
        "candidates", "picks", "page", "extraction", "followup_round", "recall", "memory_view", "focus_questions",
        "counters", "rejections", "out_of_scope", "domain_count", "notes", "budget_tally", "budget_flags",
        "seconds_used", "work_limits", "guidance"}
    harness.graph_engine.close()
    checkpoints = sqlite3.connect(harness.settings.data_dir / "langgraph_checkpoints.sqlite3")
    largest = checkpoints.execute("SELECT MAX(LENGTH(checkpoint)) FROM checkpoints").fetchone()[0]
    blobs = b"".join(row[0] for row in checkpoints.execute("SELECT checkpoint FROM checkpoints UNION ALL SELECT value FROM writes"))
    checkpoints.close()
    page_text = harness.app.store.one("SELECT text FROM source_snapshot ORDER BY LENGTH(text) DESC LIMIT 1")["text"]
    assert page_text[:120].encode() not in blobs and page_text[-120:].encode() not in blobs  # no page bodies in checkpoints
    assert largest < 20_000
    harness.app.close()


# ------------------------------------------------------- 3. crash and resume


def reference(tmp_path, **settings) -> tuple[dict, list[str]]:
    harness = graph_harness(tmp_path, "reference", **settings)
    faults = Faults(harness)
    watch_id = harness.watch()
    harness.run(watch_id)
    state = research_state(harness, watch_id)
    harness.close()
    return state, faults.sequence


@pytest.mark.parametrize("kind, when", [
    ("llm:plan", "after"), ("search", "after"), ("llm:triage", "before"), ("fetch", "after"),
    ("llm:extract", "after"), ("llm:support", "before"), ("llm:reflect", "after"),
])
def test_session_resumes_after_a_crash_and_reaches_the_same_result(tmp_path, kind, when):
    expected, sequence = reference(tmp_path)
    position = [i for i, k in enumerate(sequence) if k == kind][1 if sequence.count(kind) > 1 else 0]

    harness = graph_harness(tmp_path, "crashed")
    faults = Faults(harness)
    watch_id = harness.watch()
    faults.arm(position, when)
    with pytest.raises(SimulatedCrash):
        harness.run(watch_id)
    run = harness.app.store.list_runs(watch_id)[-1]
    assert run["status"] == RunStatus.RUNNING  # nothing got to tidy up

    engine = restart(harness)
    assert engine.can_resume(run["id"])
    outcome = engine.resume(run["id"])
    assert outcome.status is RunStatus.COMPLETED and outcome.run_id == run["id"]
    assert len(harness.app.store.list_runs(watch_id)) == 1  # the same session finished; no second one was needed
    assert research_state(harness, watch_id) == expected
    harness.close()


def test_crash_after_page_was_stored_does_not_lose_the_page(tmp_path):
    """The dangerous window: content hash recorded, extraction not yet done."""
    expected, _ = reference(tmp_path)
    harness = graph_harness(tmp_path, "crashed")
    watch_id = harness.watch()
    store = harness.app.store
    real, calls = store.add_event, {"n": 0}

    def die_after_first_fetch_event(run_id, watch, kind, data=None):
        real(run_id, watch, kind, data)
        if kind == "fetch" and (data or {}).get("outcome") == "OK":
            calls["n"] += 1
            if calls["n"] == 2:
                store.add_event = real
                raise SimulatedCrash("died after the fetch was recorded, before the snapshot")

    store.add_event = die_after_first_fetch_event
    with pytest.raises(SimulatedCrash):
        harness.run(watch_id)
    run = harness.app.store.list_runs(watch_id)[-1]
    assert research_state(harness_resumed(harness, run["id"]), watch_id) == expected
    harness.close()


def harness_resumed(harness: Harness, run_id: str) -> Harness:
    restart(harness).resume(run_id)
    return harness


def test_crash_between_storing_and_extracting_is_replayed_from_the_snapshot(tmp_path):
    expected, _ = reference(tmp_path)
    harness = graph_harness(tmp_path, "crashed")
    watch_id = harness.watch()
    store = harness.app.store
    real_update, seen = store.update_source, {"n": 0}

    def die_after_recording_fetch(source_id, **values):
        real_update(source_id, **values)
        if "content_hash" in values:
            seen["n"] += 1
            if seen["n"] == 2:
                store.update_source = real_update
                raise SimulatedCrash("died right after the page was stored and marked FETCHED")

    store.update_source = die_after_recording_fetch
    with pytest.raises(SimulatedCrash):
        harness.run(watch_id)
    run = harness.app.store.list_runs(watch_id)[-1]
    fetches_before = len(harness.web.fetch_log)
    restart(harness).resume(run["id"])
    replayed = [e["data"] for e in harness.app.store.list_events(run["id"]) if e["kind"] == "step_replayed"]
    assert {"step": "fetch"}.items() <= replayed[0].items()
    assert research_state(harness, watch_id) == expected
    # The stored page was read from the snapshot; no page was downloaded a second time.
    assert len(harness.web.fetch_log) == len(set(harness.web.fetch_log)) > fetches_before
    harness.close()


def test_crash_while_filing_claims_leaves_no_half_recorded_finding(tmp_path):
    expected, _ = reference(tmp_path)
    harness = graph_harness(tmp_path, "crashed")
    watch_id = harness.watch()
    store = harness.app.store
    real, seen = store.record_change, {"n": 0}

    def die_before_change_record(*args, **kwargs):
        seen["n"] += 1
        if seen["n"] == 2:
            store.record_change = real
            raise SimulatedCrash("died after the claims were inserted, before the change record")
        return real(*args, **kwargs)

    store.record_change = die_before_change_record
    with pytest.raises(SimulatedCrash):
        harness.run(watch_id)
    run = harness.app.store.list_runs(watch_id)[-1]
    # The item's claims and its change record are one transaction, so the claims did not survive alone.
    orphans = harness.app.store.scalar(
        "SELECT COUNT(*) FROM claim c WHERE c.status = 'ACCEPTED' AND c.run_id = ? AND NOT EXISTS "
        "(SELECT 1 FROM finding_change fc WHERE fc.run_id = c.run_id AND fc.finding_id = c.finding_id)", (run["id"],))
    assert orphans == 0
    restart(harness).resume(run["id"])
    assert research_state(harness, watch_id) == expected
    harness.close()


# ------------------------------------------------- 4. retried nodes are idempotent


class Transient(Exception):
    pass


def test_automatic_node_retry_does_not_repeat_the_search(tmp_path):
    langgraph_only()
    from langgraph.types import RetryPolicy

    harness = graph_harness(tmp_path, engine_options={
        "retry": {"search": RetryPolicy(max_attempts=3, initial_interval=0.0, jitter=False, retry_on=Transient)}})
    watch_id = harness.watch()
    store = harness.app.store
    real, state = store.execute, {"armed": True}

    def fail_once_after_search_was_recorded(sql, params=()):
        result = real(sql, params)
        if state["armed"] and sql.startswith("UPDATE search_attempt SET new_url_count"):
            state["armed"] = False
            raise Transient("failed after the provider answered and the attempt was stored")
        return result

    store.execute = fail_once_after_search_was_recorded
    outcome = harness.run(watch_id)
    assert outcome.status is RunStatus.COMPLETED
    queries = [a["query"] for a in store.list_search_attempts(watch_id, outcome.run_id)]
    assert len(queries) == len(set(queries)) == len(harness.web.search_log)  # the provider saw each query once
    assert outcome.metrics["search_requests"] == len(queries)
    assert any(e["kind"] == "step_replayed" and e["data"]["step"] == "search" for e in store.list_events(outcome.run_id))
    harness.close()


def test_resume_never_exceeds_the_budget_or_repeats_recorded_side_effects(tmp_path):
    budget = dict(max_search_requests=2, max_fetched_pages=3)
    expected, sequence = reference(tmp_path, **budget)
    for position in range(len(sequence)):
        harness = graph_harness(tmp_path, f"crash{position}", **budget)
        faults = Faults(harness)
        watch_id = harness.watch()
        faults.arm(position, "after")
        with pytest.raises(SimulatedCrash):
            harness.run(watch_id)
        run_id = harness.app.store.list_runs(watch_id)[-1]["id"]
        outcome = restart(harness).resume(run_id)
        store = harness.app.store
        attempts = store.list_search_attempts(watch_id, run_id)
        assert len(attempts) <= 2 and outcome.metrics["search_requests"] <= 2, position
        assert outcome.metrics["fetched_pages"] <= 3, position
        assert len({a["query"] for a in attempts}) == len(attempts), position  # no search recorded twice
        assert store.scalar("SELECT COUNT(*) FROM source_snapshot") == store.scalar(
            "SELECT COUNT(DISTINCT source_id || content_hash) FROM source_snapshot"), position
        assert research_state(harness, watch_id) == expected, f"diverged after crash at call {position} ({sequence[position]})"
        harness.close()


# ----------------------------------------- 5. cancellation, then safe continuation


def test_paused_session_is_cancelled_like_before_and_can_later_continue(tmp_path):
    expected, _ = reference(tmp_path)
    harness = graph_harness(tmp_path, "paused")
    watch_id = harness.watch()
    original_fetch = harness.web.fetch

    def fetch_then_pause(url):
        result = original_fetch(url)
        if len(harness.web.fetch_log) == 2:
            harness.app.watches.pause(watch_id)
        return result

    harness.web.fetch = fetch_then_pause
    outcome = harness.run(watch_id)
    assert outcome.status is RunStatus.CANCELLED  # same contract as the linear engine
    assert harness.report(outcome.run_id)["status"] == "cancelled"
    assert harness.app.store.running_run() is None
    engine = harness.graph_engine
    assert engine.can_resume(outcome.run_id) and engine.history(outcome.run_id)[-1][1] == ("fetch",)

    harness.app.watches.request_resume(watch_id)
    harness.app.watches.approve(watch_id)
    resumed = engine.resume(outcome.run_id)
    assert resumed.status is RunStatus.COMPLETED and resumed.run_id == outcome.run_id
    assert len(harness.app.store.list_runs(watch_id)) == 1
    assert research_state(harness, watch_id) == expected
    assert harness.web.fetch_log.count(harness.web.fetch_log[1]) == 2  # the interrupted page was fetched again: at-least-once
    harness.close()


def test_session_does_not_continue_under_a_changed_scope(tmp_path):
    harness = graph_harness(tmp_path, "rescoped")
    watch_id = harness.watch()
    original_fetch = harness.web.fetch

    def fetch_then_pause(url):
        result = original_fetch(url)
        if len(harness.web.fetch_log) == 1:
            harness.app.watches.pause(watch_id)
        return result

    harness.web.fetch = fetch_then_pause
    outcome = harness.run(watch_id)
    with pytest.raises(ValueError, match="not active"):
        harness.graph_engine.resume(outcome.run_id)  # still paused: no research without the user's say-so
    harness.app.watches.request_resume(watch_id, patch={"exclude_add": ["installed capacity"]})
    harness.app.watches.approve(watch_id)
    with pytest.raises(ValueError, match="scope changed"):
        harness.graph_engine.resume(outcome.run_id)
    assert harness.app.store.running_run() is None
    assert harness.run(watch_id).status is RunStatus.COMPLETED  # a fresh session under the new brief is the way on
    harness.close()


def test_resume_is_refused_when_nothing_is_left_or_another_session_runs(tmp_path):
    harness = graph_harness(tmp_path)
    first, second = harness.watch(), harness.watch("What does tidal energy cost per megawatt hour?")
    done = harness.run(first)
    with pytest.raises(ValueError, match="nothing left to resume"):
        harness.graph_engine.resume(done.run_id)
    faults = Faults(harness)
    faults.arm(1, "before")
    with pytest.raises(SimulatedCrash):
        harness.run(second)
    stuck = harness.app.store.list_runs(second)[-1]["id"]
    harness.app.store.update("session_run", {"id": stuck}, {"status": "INTERRUPTED"})
    harness.app.runner.admit(first, "manual")
    with pytest.raises(ValueError, match="one session runs at a time"):
        harness.graph_engine.resume(stuck)
    harness.close()


# -------------------------------------- 7. controller and policies work unchanged


def test_run_manager_scheduler_and_policies_work_with_the_graph_engine(tmp_path):
    from forage.scheduler.queue import Scheduler

    harness = graph_harness(tmp_path, min_minutes_between_sessions=0)
    blocked = harness.watch("Objective alpha: installed tidal turbine capacity and tidal energy cost",
                            policy="adaptive", exclude=["installed capacity"])
    other = harness.watch("Objective beta: installed tidal turbine capacity and tidal energy cost", policy="periodic",
                          policy_config={"every": 99})
    scheduler = Scheduler(harness.app.store, harness.settings, harness.app.runner, clock=harness.clock, owner="t")
    scheduler.run_batch(max_sessions=2, respect_window=False)
    harness.web.epoch = 1
    scheduler.run_batch(max_sessions=3, respect_window=False)
    store = harness.app.store
    assert store.get_watch(blocked)["status"] == WatchStatus.AWAITING_REVIEW
    assert harness.app.reviews.pending_for(blocked)["signal"] == "scope_conflict"
    assert [r["status"] for r in store.list_runs(other)] == ["COMPLETED"] * 3
    assert store.scalar("SELECT COUNT(*) FROM checkpoint_decision") == 5
    harness.close()


# ------------------------------------------- 9. conditional follow-up branch, HITL


def directions_llm():
    """A simulated model whose review always proposes one further search direction."""
    def reflect(payload):
        base = SimulatedLLM._reflect(payload)
        base["next_directions"] = [{"query": "tidal turbine array announced Wales construction", "rationale": "new array"}]
        return base
    return SimulatedLLM(overrides={"reflect": reflect})


def test_follow_up_branch_matches_the_linear_engine(tmp_path):
    settings = dict(followup_rounds=1, max_search_requests=4)
    linear = Harness(tmp_path / "linear", directions_llm(), **settings)
    graph = graph_harness(tmp_path, llm=directions_llm(), **settings)
    traces = []
    for h in (linear, graph):
        h.web.epoch = 1
        watch_id = h.watch()
        outcome = h.run(watch_id, max_search_requests=4)
        assert [purpose for purpose, _ in h.llm.calls].count("reflect") == 2  # reviewed, followed up, reviewed again
        assert any(q.startswith("tidal turbine array announced") for q in h.web.search_log)  # the proposed direction
        traces.append(collect(h.app, h.web, h.llm, [watch_id]))
    assert first_difference(*traces) is None
    visited = [n[0] for _, n in graph.graph_engine.history(outcome.run_id) if n]
    assert visited.count("reflect") == 2 and "follow_up" in visited
    linear.close(), graph.close()


@pytest.mark.parametrize("approve", [True, False])
def test_session_can_stop_to_ask_and_continue_with_the_answer(tmp_path, approve):
    harness = graph_harness(tmp_path, llm=directions_llm(), engine_options={"ask_before_follow_up": True},
                            followup_rounds=1, max_search_requests=4)
    harness.web.epoch = 1
    watch_id = harness.watch()
    parked = harness.run(watch_id)
    engine = harness.graph_engine
    assert parked.status is RunStatus.CANCELLED and harness.app.store.get_run(parked.run_id)["stop_reason"] == "awaiting_input"
    assert harness.app.store.running_run() is None  # a waiting session does not hold the single session slot
    question = engine.pending_question(parked.run_id)
    assert "follow-up round" in question["question"] and question["directions"]
    searches_before = len(harness.web.search_log)

    engine = restart(harness)  # the answer may arrive days later, in another process
    assert engine.pending_question(parked.run_id) == question
    outcome = engine.resume(parked.run_id, answer={"approve": approve})
    assert outcome.status is RunStatus.COMPLETED and outcome.run_id == parked.run_id
    followed = len(harness.web.search_log) > searches_before
    assert followed is approve
    assert [p for p, _ in harness.llm.calls].count("reflect") == (2 if approve else 1)
    notes = harness.app.store.get_run(parked.run_id)["metrics"]["notes"]
    assert ("A follow-up round was proposed and declined by the user." in notes) is (not approve)
    harness.close()


# ---------------------------------------- 10. testing and debugging single steps


def test_a_single_step_can_be_tested_from_a_handwritten_state(tmp_path):
    """Steps are plain functions of (session, state, context): no driver is involved here at all."""
    harness = graph_harness(tmp_path)
    watch_id = harness.watch()
    run = harness.app.runner.admit(watch_id, "manual")
    context = EngineContext(harness.app.engine.base, lambda: False)
    recalled = StepSession(context.engine, {"run_id": run["id"]})
    state = {"run_id": run["id"]} | recalled.export() | _STEPS["recall"](recalled, {"run_id": run["id"]}, context)

    def route(**changes):
        probe = state | changes
        session = StepSession(context.engine, probe)
        return _STEPS["dispatch"](session, probe, context)

    assert route()["route"] == "plan"  # nothing stale, no saved leads: go and plan
    assert route(phase="search", plan=[{"query": "tidal energy cost"}], query_index=0)["route"] == "search"
    assert route(phase="search", plan=[{"query": "tidal energy cost"}], query_index=1)["route"] == "reflect"
    assert route(phase="search", picks=["s-1", "s-2"])["page"] == {"source_id": "s-1", "recheck": False}
    spent = dict(state["budget_tally"], fetched_pages=99)
    assert route(phase="search", picks=["s-1"], budget_tally=spent, plan=[], query_index=0)["route"] == "reflect"
    harness.close()


def test_a_late_step_can_be_rerun_from_its_checkpoint_without_repeating_the_session(tmp_path):
    """Time travel: re-run only the review step, for example to try a different prompt or model."""
    langgraph_only()
    harness = graph_harness(tmp_path)
    watch_id = harness.watch()
    outcome = harness.run(watch_id)
    engine = harness.graph_engine
    before_reflect = next(s for s in engine.graph.get_state_history(engine._config(outcome.run_id)) if s.next == ("reflect",))
    calls_before, fetches_before = len(harness.llm.calls), len(harness.web.fetch_log)
    harness.app.store.update("session_run", {"id": outcome.run_id}, {"status": "RUNNING"})
    engine.graph.invoke(None, before_reflect.config, context=EngineContext(engine.base, lambda: False), durability="sync")
    assert [p for p, _ in harness.llm.calls[calls_before:]] == ["reflect"]  # one model call, no searching or fetching
    assert len(harness.web.fetch_log) == fetches_before
    assert harness.app.store.get_run(outcome.run_id)["status"] == RunStatus.COMPLETED
    harness.close()
