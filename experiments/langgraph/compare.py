"""Measured comparison of three ways to drive a Forage research session.

    linear     the production engine: one Python call runs the whole session
    langgraph  the prototype: the same steps as a LangGraph StateGraph with a SQLite checkpointer
    plain      the same steps driven by a small loop with its own checkpoint table

Everything here uses the SIMULATED web and model, so differences are due to orchestration
alone. Run from the repository root:

    .venv/bin/python -m experiments.langgraph.compare all --out experiments/langgraph/results

Subcommands: equivalence, crash, overhead, scale, kill, all.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

from conftest import Harness  # noqa: E402

from experiments.langgraph.faults import Faults, SimulatedCrash  # noqa: E402
from experiments.langgraph.trace import collect, first_difference, research_state  # noqa: E402

DRIVERS = ("linear", "langgraph", "plain")


def make(tmp: Path, name: str, driver: str, llm=None, engine_options: dict | None = None, **settings) -> Harness:
    from forage.experimental.langgraph_engine import use_engine

    harness = Harness(tmp / name, llm, **settings)
    harness.engine_kind, harness.engine_options = driver, engine_options or {}
    harness.graph_engine = None if driver == "linear" else use_engine(harness.app, driver, **harness.engine_options)
    return harness


def restart(harness: Harness):
    from forage.experimental.langgraph_engine import use_engine

    harness.close()
    harness.open()
    if harness.engine_kind != "linear":
        harness.graph_engine = use_engine(harness.app, harness.engine_kind, **harness.engine_options)
    return harness.graph_engine


# ------------------------------------------------------------ equivalence


def tidal(harness: Harness) -> list[str]:
    watch_id = harness.watch(freshness_days=7)
    for epoch in range(5):
        harness.web.epoch = min(epoch, 3)
        harness.run(watch_id)
        harness.clock.advance(days=8)
    return [watch_id]


def scoped(harness: Harness) -> list[str]:
    watch_id = harness.watch(policy="adaptive", exclude=["installed capacity"])
    harness.run(watch_id, max_fetched_pages=2)
    harness.web.epoch = 1
    harness.clock.advance(hours=3)
    harness.run(watch_id, max_search_requests=1)
    return [watch_id]


def equivalence(tmp: Path) -> dict[str, Any]:
    """Do the three drivers do the same research? Full traces on fixed scenarios, and a 90-session experiment."""
    from forage.eval.experiment import POLICIES, run_policy
    from forage.eval.scenarios import load_scenario
    from forage.experimental.langgraph_engine import use_engine

    out: dict[str, Any] = {"traces": {}, "three_watch_experiment": {}}
    for name, scenario in (("tidal_5_sessions", tidal), ("scoped_2_sessions", scoped)):
        traces = {}
        for driver in DRIVERS:
            harness = make(tmp, f"eq-{name}-{driver}", driver)
            traces[driver] = collect(harness.app, harness.web, harness.llm, scenario(harness))
            harness.close()
        base = traces["linear"]
        out["traces"][name] = {
            "sessions": sum(len(w["runs"]) for w in base["watches"]), "searches": len(base["searches"]),
            "fetches": len(base["fetches"]), "model_calls": len(base["model_calls"]),
            "langgraph_vs_linear": first_difference(base, traces["langgraph"]) or "identical",
            "plain_vs_linear": first_difference(base, traces["plain"]) or "identical",
        }
    scenario = load_scenario("three-watches")
    for policy in POLICIES:
        results = {}
        for driver in DRIVERS:
            configure = None if driver == "linear" else (lambda app, kind=driver: use_engine(app, kind))
            results[driver] = run_policy(scenario, policy, tmp / f"exp-{policy}-{driver}", configure)
        base = results["linear"]
        out["three_watch_experiment"][policy] = {
            "sessions": base["totals"]["sessions_run"],
            "langgraph_vs_linear": first_difference(base, results["langgraph"]) or "identical",
            "plain_vs_linear": first_difference(base, results["plain"]) or "identical",
        }
    return out


# ------------------------------------------------------------ crash sweep


def _setup_fresh(harness: Harness) -> str:
    return harness.watch()


def _setup_second_session(harness: Harness) -> str:
    watch_id = harness.watch()
    harness.run(watch_id, max_fetched_pages=3)  # leaves relevant pages unread for the next session
    harness.web.epoch = 1
    harness.clock.advance(hours=3)
    return watch_id


SETUPS = {"first_session": _setup_fresh, "second_session_with_saved_leads": _setup_second_session}


def crash_sweep(tmp: Path) -> dict[str, Any]:
    """Kill the session before and after every adapter call, recover the way each driver can, compare outcomes."""
    out: dict[str, Any] = {}
    for setup_name, setup in SETUPS.items():
        # Reference: the same session with no crash (identical across drivers, see `equivalence`).
        ref = make(tmp, f"ref-{setup_name}", "linear")
        watch_id = setup(ref)
        faults = Faults(ref)
        ref.run(watch_id)
        expected = research_state(ref.app.store, watch_id)
        sequence = list(faults.sequence)
        ref_claims = {c[0] for c in expected["claims"] if c[1] == "ACCEPTED"}
        ref_reported = {title for title, _, _ in expected["report"]}
        ref.close()

        per_driver: dict[str, Any] = {}
        for driver in DRIVERS:
            rows = []
            for position in range(len(sequence)):
                for when in ("before", "after"):
                    harness = make(tmp, f"crash-{setup_name}-{driver}-{position}-{when}", driver)
                    watch_id = setup(harness)
                    faults = Faults(harness)
                    faults.arm(position, when)
                    try:
                        harness.run(watch_id)
                        crashed = False
                    except SimulatedCrash:
                        crashed = True
                    assert crashed
                    run_id = harness.app.store.list_runs(watch_id)[-1]["id"]
                    engine = restart(harness)  # the simulated web and model objects, and their call counter, carry over
                    if driver == "linear":
                        harness.app.runner.recover(force=True)  # the run is closed as INTERRUPTED, with a partial report
                        harness.run(watch_id)  # the only way forward is another session
                    else:
                        engine.resume(run_id)
                    store = harness.app.store
                    state = research_state(store, watch_id)
                    accepted = {c[0] for c in state["claims"] if c[1] == "ACCEPTED"}
                    reported = set()
                    for run in store.list_runs(watch_id)[-(2 if driver == "linear" else 1):]:
                        row = store.get_report(run["id"])
                        reported |= {f["title"] for f in json.loads(row["json"])["findings"]} if row else set()
                    stranded = store.scalar(
                        "SELECT COUNT(*) FROM source s WHERE s.watch_id = ? AND s.status = 'FETCHED' "
                        "AND NOT EXISTS (SELECT 1 FROM claim c WHERE c.source_id = s.id) "
                        "AND NOT EXISTS (SELECT 1 FROM run_event e WHERE e.kind = 'source_result' "
                        "AND json_extract(e.data, '$.source_id') = s.id)", (watch_id,))
                    made = list(faults.sequence)
                    calls = len(made)
                    rows.append({
                        "position": position, "call": sequence[position], "when": when,
                        "same_as_uninterrupted": state == expected,
                        "verified_claims_missing": len(ref_claims - accepted),
                        "findings_never_reported": len(ref_reported - reported),
                        "pages_stored_but_never_read": int(stranded or 0),
                        "sessions_used": len(store.list_runs(watch_id)) - (1 if setup is _setup_second_session else 0),
                        "adapter_calls": calls, "extra_adapter_calls": calls - len(sequence),
                        "extra_model_calls": sum(1 for k in made if k.startswith("llm")) - sum(1 for k in sequence if k.startswith("llm")),
                        "extra_searches": made.count("search") - sequence.count("search"),
                        "extra_fetches": made.count("fetch") - sequence.count("fetch"),
                    })
                    harness.close()
                    shutil.rmtree(harness.tmp_path, ignore_errors=True)
            n = len(rows)
            per_driver[driver] = {
                "crash_points": n,
                "same_as_uninterrupted": sum(r["same_as_uninterrupted"] for r in rows),
                "crash_points_losing_verified_claims": sum(1 for r in rows if r["verified_claims_missing"]),
                "crash_points_with_findings_never_reported": sum(1 for r in rows if r["findings_never_reported"]),
                "crash_points_leaving_a_page_stored_but_unread": sum(1 for r in rows if r["pages_stored_but_never_read"]),
                "sessions_used_mean": round(statistics.mean(r["sessions_used"] for r in rows), 2),
                "extra_model_calls_mean": round(statistics.mean(r["extra_model_calls"] for r in rows), 2),
                "extra_model_calls_max": max(r["extra_model_calls"] for r in rows),
                "extra_searches_mean": round(statistics.mean(r["extra_searches"] for r in rows), 2),
                "extra_searches_max": max(r["extra_searches"] for r in rows),
                "extra_fetches_mean": round(statistics.mean(r["extra_fetches"] for r in rows), 2),
                "extra_fetches_max": max(r["extra_fetches"] for r in rows),
                "worst_points": sorted(
                    ({k: r[k] for k in ("call", "when", "verified_claims_missing", "findings_never_reported",
                                        "pages_stored_but_never_read")} for r in rows if not r["same_as_uninterrupted"]),
                    key=lambda r: -(r["verified_claims_missing"] + r["findings_never_reported"]))[:6],
            }
        out[setup_name] = {
            "adapter_calls_in_uninterrupted_session": len(sequence),
            "of_which_model_calls": sum(1 for k in sequence if k.startswith("llm")),
            "drivers": per_driver,
        }
    return out


# ------------------------------------------------------------- overhead


def _session_count(harness: Harness) -> int:
    return int(harness.app.store.scalar("SELECT COUNT(*) FROM session_run") or 0)


def _file_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.parent.glob(path.name + "*"))


def overhead(tmp: Path, repeats: int = 7) -> dict[str, Any]:
    """Time and storage per session with instant adapters, so only orchestration and SQLite cost is measured."""
    variants: list[tuple[str, str, dict[str, Any]]] = [
        ("linear", "linear", {}),
        ("plain", "plain", {}),
        ("langgraph durability=sync", "langgraph", {"durability": "sync"}),
        ("langgraph durability=async", "langgraph", {"durability": "async"}),
        ("langgraph durability=exit", "langgraph", {"durability": "exit"}),
        ("langgraph sync, checkpoints deleted after each session", "langgraph", {"durability": "sync", "keep_checkpoints": False}),
    ]
    out: dict[str, Any] = {"repeats": repeats, "variants": {}}
    for label, driver, options in variants:
        times, steps, sizes = [], 0, {}
        for i in range(repeats):
            harness = make(tmp, f"ovh-{driver}-{len(out['variants'])}-{i}", driver, engine_options=options)
            started = time.perf_counter()
            watch_ids = tidal(harness)
            elapsed = time.perf_counter() - started
            sessions = _session_count(harness)
            times.append(elapsed / sessions)
            if i == 0:
                data = harness.settings.data_dir
                rows = payload = 0
                if driver != "linear":
                    steps = sum(len(harness.graph_engine.history(r["id"])) for w in watch_ids for r in harness.app.store.list_runs(w))
                    conn = harness.graph_engine._conn
                    if driver == "langgraph":
                        rows, payload = conn.execute(
                            "SELECT (SELECT COUNT(*) FROM checkpoints) + (SELECT COUNT(*) FROM writes), "
                            "COALESCE((SELECT SUM(LENGTH(checkpoint) + LENGTH(metadata)) FROM checkpoints), 0) + "
                            "COALESCE((SELECT SUM(LENGTH(value)) FROM writes), 0)").fetchone()
                    else:
                        rows, payload = conn.execute("SELECT COUNT(*), COALESCE(SUM(LENGTH(state)), 0) FROM step_checkpoint").fetchone()
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                harness.app.store.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                sizes = {
                    "sessions": sessions,
                    "research_db_bytes": _file_size(data / "forage.sqlite3"),
                    "checkpoint_rows_kept": rows, "checkpoint_payload_bytes_kept": payload,
                    "checkpoint_db_bytes": _file_size(data / "langgraph_checkpoints.sqlite3") if driver == "langgraph"
                    else _file_size(data / "step_checkpoints.sqlite3") if driver == "plain" else 0,
                }
            harness.close()
            shutil.rmtree(harness.tmp_path, ignore_errors=True)
        out["variants"][label] = {
            "ms_per_session_median": round(1000 * statistics.median(times), 2),
            "ms_per_session_min": round(1000 * min(times), 2),
            "checkpointed_steps_total": steps,
            **sizes,
            "checkpoint_payload_bytes_per_session": round(sizes["checkpoint_payload_bytes_kept"] / sizes["sessions"]) if sizes.get("sessions") else 0,
        }
    base = out["variants"]["linear"]["ms_per_session_median"]
    for label, row in out["variants"].items():
        row["overhead_ms_per_session_vs_linear"] = round(row["ms_per_session_median"] - base, 2)
        if row["checkpointed_steps_total"]:
            row["steps_per_session"] = round(row["checkpointed_steps_total"] / row["sessions"], 1)
            row["overhead_ms_per_step"] = round(row["overhead_ms_per_session_vs_linear"] / row["steps_per_session"], 3)
    return out


def _grow(harness: Harness, watch_id: str, findings: int, claims_each: int) -> None:
    """Give a watch a long history directly in the store (synthetic rows, clearly not research)."""
    store = harness.app.store
    run = harness.app.runner.admit(watch_id, "manual")
    with store.tx():
        for i in range(findings):
            url = f"https://archive{i % 40}.example.org/item/{i}"
            source, _ = store.upsert_discovered_source(watch_id, url, url, f"archive{i % 40}.example.org", f"Archive item {i}",
                                                       "synthetic", "synthetic history", run["id"])
            store.update_source(source["id"], status="FETCHED", content_hash=f"h{i}", last_fetched_at=store.now(), fetch_count=1)
            snapshot_id, _ = store.add_snapshot(source["id"], run["id"], f"h{i}", f"Synthetic archive text {i}. " * 20)
            finding_id = store.create_finding(watch_id, run["id"], f"Archived topic {i} about harbour dredging", f"harbour dredging {i}",
                                              2, None, f"seed{i}")
            for j in range(claims_each):
                store.add_claim(watch_id=watch_id, finding_id=finding_id, source_id=source["id"], snapshot_id=snapshot_id,
                                run_id=run["id"], text=f"Dredging contract {i}-{j} moved {1000 + i * 7 + j} cubic metres of silt in phase {j}.",
                                quote=f"Synthetic archive text {i}.", quote_offset=0, verification="QUOTE_EXACT",
                                support="SUPPORTED", status="ACCEPTED")
    store.update("session_run", {"id": run["id"]}, {"status": "COMPLETED", "finished_at": store.now(),
                                                    "metrics": {}, "has_meaningful_update": 0})


def scale(tmp: Path, repeats: int = 3) -> dict[str, Any]:
    """How does per-session cost grow with the amount of research already stored for the watch?"""
    out: dict[str, Any] = {}
    for findings, claims_each in ((0, 0), (200, 4), (1000, 5)):
        row: dict[str, Any] = {"stored_findings": findings, "stored_claims": findings * claims_each}
        for driver in DRIVERS:
            times = []
            for i in range(repeats):
                harness = make(tmp, f"scale-{findings}-{driver}-{i}", driver)
                watch_id = harness.watch()
                if findings:
                    _grow(harness, watch_id, findings, claims_each)
                started = time.perf_counter()
                outcome = harness.run(watch_id)
                times.append(time.perf_counter() - started)
                assert outcome.status == "COMPLETED", outcome.error
                harness.close()
                shutil.rmtree(harness.tmp_path, ignore_errors=True)
            row[f"{driver}_ms_per_session"] = round(1000 * statistics.median(times), 1)
        out[f"{findings}_findings"] = row
    return out


def memory_and_import() -> dict[str, Any]:
    """Peak resident memory and import time, each driver in its own fresh interpreter."""
    out: dict[str, Any] = {}
    for driver in DRIVERS:
        proc = subprocess.run([sys.executable, "-m", "experiments.langgraph.compare", "_child", driver],
                              capture_output=True, text=True, cwd=ROOT, check=True)
        out[driver] = json.loads(proc.stdout.strip().splitlines()[-1])
    return out


def _child(driver: str) -> None:
    started = time.perf_counter()
    import forage.engine.session  # noqa: F401
    if driver == "langgraph":
        import forage.experimental.langgraph_engine.engine  # noqa: F401
    elif driver == "plain":
        import forage.experimental.langgraph_engine.plain_runner  # noqa: F401
    import_seconds = time.perf_counter() - started
    with tempfile.TemporaryDirectory() as tmp:
        harness = make(Path(tmp), "mem", driver)
        tracemalloc.start()
        tidal(harness)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        harness.close()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
    print(json.dumps({"import_ms": round(1000 * import_seconds), "modules_loaded": len(sys.modules),
                      "python_heap_peak_mb_during_5_sessions": round(peak / 1e6, 2), "process_peak_rss_mb": round(rss_mb, 1)}))


# ----------------------------------------------------- real process death


def kill_test(tmp: Path, repeats: int = 6) -> dict[str, Any]:
    """Kill a real process (os._exit, no cleanup) in the middle of a session, then resume in a new process."""
    out: dict[str, Any] = {}
    ref = make(tmp, "kill-ref", "linear")
    watch_id = ref.watch()
    ref.run(watch_id)
    expected = research_state(ref.app.store, watch_id)
    ref.close()
    cases = [("plain", {}), ("langgraph", {"durability": "sync"}), ("langgraph", {"durability": "async"}),
             ("langgraph", {"durability": "exit"})]
    for driver, options in cases:
        label = driver + (f" durability={options['durability']}" if options else "")
        rows = []
        for repeat in range(repeats):
            for kill_at in (3, 9, 14, 20):
                data = tmp / f"kill-{label.replace(' ', '-').replace('=', '-')}-{kill_at}-{repeat}"
                args = [sys.executable, "-m", "experiments.langgraph.compare", "_kill_child", driver, json.dumps(options), str(data)]
                first = subprocess.run(args + ["run", str(kill_at)], capture_output=True, text=True, cwd=ROOT)
                second = subprocess.run(args + ["resume", "-1"], capture_output=True, text=True, cwd=ROOT)
                try:
                    result = json.loads(second.stdout.strip().splitlines()[-1])
                except (IndexError, ValueError):
                    result = {"resumed": False, "error": (second.stderr or second.stdout)[-300:]}
                result["first_process_exit_code"] = first.returncode
                if result.get("state") is not None:
                    result["same_as_uninterrupted"] = json.loads(json.dumps(result.pop("state"))) == json.loads(json.dumps(expected))
                rows.append({"killed_at_adapter_call": kill_at, **result})
                shutil.rmtree(data, ignore_errors=True)
        out[label] = {
            "kills": len(rows),
            "resumed_in_same_session": sum(1 for r in rows if r.get("resumed")),
            "same_result_as_uninterrupted": sum(1 for r in rows if r.get("same_as_uninterrupted")),
            "not_resumable": sum(1 for r in rows if not r.get("resumed")),
            "not_resumable_detail": [{k: r.get(k) for k in ("killed_at_adapter_call", "checkpoints_found", "latest_next",
                                                           "latest_step", "error")} for r in rows if not r.get("resumed")][:6],
        }
    return out


def _kill_child(driver: str, options_json: str, data: str, mode: str, kill_at: str) -> None:
    harness = make(Path(data), "h", driver, engine_options=json.loads(options_json))
    store = harness.app.store
    if mode == "run":
        faults = Faults(harness)
        faults.arm(int(kill_at), "after", on_crash=lambda: os._exit(9))  # the process vanishes: no finally, no close
        watch_id = harness.watch()
        harness.run(watch_id)
        return
    watch = store.list_watches()[0]
    run = store.list_runs(watch["id"])[-1]
    history = harness.graph_engine.history(run["id"])
    result: dict[str, Any] = {"run_status_found": run["status"], "can_resume": harness.graph_engine.can_resume(run["id"]),
                              "checkpoints_found": len(history), "latest_step": history[-1][0] if history else None,
                              "latest_next": list(history[-1][1]) if history else None}
    if result["can_resume"]:
        outcome = harness.graph_engine.resume(run["id"])
        result.update(resumed=True, final_status=str(outcome.status), sessions=len(store.list_runs(watch["id"])),
                      state=research_state(store, watch["id"]))
    else:
        result.update(resumed=False, state=None)
    print(json.dumps(result, default=str))
    harness.close()


# --------------------------------------------------------------------- cli


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["equivalence", "crash", "overhead", "scale", "memory", "kill", "all", "_child", "_kill_child"])
    parser.add_argument("rest", nargs="*")
    parser.add_argument("--out", default=str(ROOT / "experiments" / "langgraph" / "results"))
    args = parser.parse_args()
    if args.what == "_child":
        return _child(args.rest[0])
    if args.what == "_kill_child":
        return _kill_child(*args.rest)
    import logging

    logging.disable(logging.WARNING)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = {"equivalence": equivalence, "crash": crash_sweep, "overhead": overhead, "scale": scale, "kill": kill_test}
    selected = list(jobs) + ["memory"] if args.what == "all" else [args.what]
    for name in selected:
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix=f"forage-compare-{name}-") as tmp:
            result = memory_and_import() if name == "memory" else jobs[name](Path(tmp))
        (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"[{name}] done in {time.perf_counter() - started:.1f}s -> {out_dir / (name + '.json')}", flush=True)


if __name__ == "__main__":
    main()
