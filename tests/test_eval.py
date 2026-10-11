"""Metric export and the policy-comparison harness. Everything here is SIMULATED."""

import json

from forage.eval.experiment import run_experiment, run_policy
from forage.eval.export import export_all, watch_summary
from forage.eval.scenarios import load_scenario


def test_export_writes_every_log_table_and_a_summary(harness, tmp_path):
    watch_id = harness.watch(policy="periodic", policy_config={"every": 2})
    harness.run(watch_id)
    harness.web.epoch = 1
    harness.run(watch_id)
    out = tmp_path / "export"
    summary = export_all(harness.app.store, out)
    for name in ("runs", "search_attempts", "llm_calls", "sources", "claims", "findings", "finding_changes",
                 "checkpoint_decisions", "checkpoint_requests", "queue_admissions", "run_events", "watch_events"):
        assert (out / f"{name}.jsonl").is_file(), name
    assert not (out / "source_snapshots.jsonl").exists()  # page text is opt-in
    runs = [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines()]
    assert len(runs) == 2 and isinstance(runs[0]["metrics"], dict) and runs[0]["model"] == "simulated"
    calls = [json.loads(line) for line in (out / "llm_calls.jsonl").read_text().splitlines()]
    assert {"purpose", "prompt_tokens", "output_tokens", "duration_ms", "eval_ms"} <= set(calls[0])
    decisions = [json.loads(line) for line in (out / "checkpoint_decisions.jsonl").read_text().splitlines()]
    assert [d["decision"] for d in decisions] == ["CONTINUE", "REQUEST_REVIEW"]

    (watch,) = summary["watches"]
    assert watch == watch_summary(harness.app.store, watch_id)
    assert watch["sessions"] == 2 and watch["reviews_requested"] == 1 and watch["reviews_by_signal"] == {"periodic_interval": 1}
    assert watch["search_requests"] > 0 and watch["claims_verified"] > 0
    assert watch["new_findings_per_100_searches"] == round(100 * watch["findings_new_reported"] / watch["search_requests"], 3)

    again = export_all(harness.app.store, tmp_path / "export2")
    assert again == summary  # an export is a pure function of the database


def test_export_can_include_page_text_when_asked(harness, tmp_path):
    harness.run(harness.watch())
    export_all(harness.app.store, tmp_path / "full", include_text=True)
    snapshots = (tmp_path / "full" / "source_snapshots.jsonl").read_text()
    assert "Northbay project" in snapshots


def test_policy_comparison_is_reproducible_and_keeps_scope_safe(tmp_path):
    scenario = load_scenario("three-watches")
    first = run_experiment(scenario, tmp_path / "a")
    second = run_experiment(load_scenario("builtin:three-watches"), tmp_path / "b")
    assert first["results"] == second["results"]  # identical inputs, identical outcome
    assert first["simulated"] is True and first["what_this_cannot_show"]

    by_policy = {r["policy"]: r for r in first["results"]}
    assert set(by_policy) == {"continuous", "periodic", "adaptive"}
    for result in by_policy.values():
        totals = result["totals"]
        # No policy ever verifies pages that the approved scope excluded at the time, or off-topic pages.
        assert totals["pages_verified_outside_approved_scope"] == 0 and totals["off_topic_pages_verified"] == 0
        assert totals["sessions_run"] <= totals["session_slots_available"]
        assert totals["relevant_pages_verified"] <= totals["relevant_pages_available"]

    continuous, periodic, adaptive = (by_policy[p] for p in ("continuous", "periodic", "adaptive"))
    assert continuous["totals"]["reviews_requested"] == 0 and continuous["user_answers"] == []
    assert continuous["per_watch"]["internships"]["scope_amended_by_user"] is False
    # The scope decision reaches the user only under policies that ask.
    assert periodic["per_watch"]["internships"]["scope_amended_by_user"] is True
    assert adaptive["per_watch"]["internships"]["scope_amended_by_user"] is True
    assert adaptive["per_watch"]["internships"]["reviews_by_signal"].get("scope_conflict") == 1
    # The productive watch is never interrupted by the adaptive policy, and is by the periodic one.
    assert adaptive["per_watch"]["events"]["reviews_requested"] == 0
    assert periodic["per_watch"]["events"]["reviews_requested"] > 0
    assert adaptive["totals"]["reviews_requested"] < periodic["totals"]["reviews_requested"]

    assert (tmp_path / "a" / "comparison.md").read_text().startswith("# Checkpoint policy comparison")
    assert "SIMULATED" in (tmp_path / "a" / "comparison.md").read_text()
    assert (tmp_path / "a" / "adaptive" / "export" / "checkpoint_decisions.jsonl").is_file()


def test_pausing_a_dry_watch_has_a_visible_cost_in_the_scenario(tmp_path):
    """The harness must be able to show a policy's downside, not only its upside."""
    scenario = load_scenario("three-watches")
    continuous = run_policy(scenario, "continuous", tmp_path / "c")
    adaptive = run_policy(scenario, "adaptive", tmp_path / "d")
    assert continuous["per_watch"]["pilot"]["relevant_pages_verified"] == 4  # kept looking, found the late page
    assert adaptive["per_watch"]["pilot"]["final_status"] == "PAUSED"
    assert adaptive["per_watch"]["pilot"]["relevant_pages_verified"] == 3  # paused on the user's say-so, missed it


def test_experiment_cli(tmp_path, capsys):
    from forage.cli import main

    out = tmp_path / "exp"
    assert main(["experiment", "--scenario", "three-watches", "--out", str(out), "--policies", "continuous,adaptive"]) == 0
    printed = capsys.readouterr().out
    assert "| | continuous | adaptive |" in printed and "SIMULATED" in printed and "What this cannot show" in printed
    assert json.loads((out / "comparison.json").read_text())["scenario"] == "three-watches"
