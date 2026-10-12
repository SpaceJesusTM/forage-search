"""LIVE check of the LangGraph driver: real Ollama model, real web search and fetch.

Uses its own data directory (never the project's `data/`). Two modes:

    start  <data_dir>   create and approve a watch, run one bounded session through the graph
    resume <data_dir>   continue the last session from its checkpoint (after a kill or a stop)

    .venv/bin/python -m experiments.langgraph.live_session start /tmp/forage-lg-live
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

from forage.app import build_app
from forage.config import load_settings
from forage.experimental.langgraph_engine import use_engine
from forage.types import WatchSpec

OBJECTIVE = ("What does LangGraph's SQLite checkpointer guarantee about durability and concurrency, "
             "and what do its maintainers recommend for production use?")
QUESTIONS = ["What durability modes does LangGraph offer and what does each guarantee?",
             "What do the LangGraph maintainers say about using the SQLite checkpointer in production?"]


def main() -> None:
    mode, data_dir = sys.argv[1], Path(sys.argv[2])
    settings = load_settings(env={})
    settings.data_dir = data_dir
    app = build_app(settings, progress=lambda m: print(f"  . {m}", flush=True))
    engine = use_engine(app, "langgraph", durability="sync")
    started = time.perf_counter()
    if mode == "start":
        watch = app.watches.create(OBJECTIVE, title="LangGraph SQLite checkpointer", spec=WatchSpec(key_questions=QUESTIONS))
        app.watches.approve(watch["id"])
        print(f"model {app.llm.model}, search {app.search.name}, watch {watch['id']}", flush=True)
        outcome = app.runner.run_watch(watch["id"], overrides={
            "max_duration_seconds": 420, "max_search_requests": 2, "max_fetched_pages": 3, "max_llm_output_tokens": 6000})
    else:
        watch = app.store.list_watches()[0]
        run = app.store.list_runs(watch["id"])[-1]
        print(f"found run {run['id']} with status {run['status']}; checkpoints: {len(engine.history(run['id']))}; "
              f"resumable: {engine.can_resume(run['id'])}", flush=True)
        outcome = engine.resume(run["id"])
    wall = time.perf_counter() - started
    run = app.store.get_run(outcome.run_id)
    calls = app.store.list_llm_calls(outcome.run_id)
    history = engine.history(outcome.run_id)
    fetch_ms = sum(e["data"].get("ms", 0) for e in app.store.list_events(outcome.run_id) if e["kind"] == "fetch")
    search_ms = sum(a["duration_ms"] or 0 for a in app.store.list_search_attempts(watch["id"], outcome.run_id))
    llm_ms = sum(c["duration_ms"] for c in calls)
    checkpoints = sqlite3.connect(str(data_dir / "langgraph_checkpoints.sqlite3"))
    rows, payload = checkpoints.execute(
        "SELECT (SELECT COUNT(*) FROM checkpoints) + (SELECT COUNT(*) FROM writes), "
        "(SELECT SUM(LENGTH(checkpoint) + LENGTH(metadata)) FROM checkpoints) + (SELECT SUM(LENGTH(value)) FROM writes)").fetchone()
    report = json.loads(app.store.get_report(outcome.run_id)["json"])
    print(json.dumps({
        "mode": mode, "status": str(outcome.status), "headline": report["headline"],
        "wall_seconds_this_process": round(wall, 1), "session_seconds_recorded": run["metrics"].get("wall_seconds"),
        "graph_steps": len(history), "model_calls_recorded": len(calls), "model_seconds": round(llm_ms / 1000, 1),
        "search_seconds": round(search_ms / 1000, 1), "fetch_seconds": round(fetch_ms / 1000, 1),
        "searches": run["metrics"].get("search_requests"), "pages": run["metrics"].get("fetched_pages"),
        "claims_kept": run["metrics"].get("claims_accepted"), "claims_rejected": run["metrics"].get("claims_rejected"),
        "findings_new": run["metrics"].get("findings_new"),
        "steps_replayed_after_resume": [e["data"] for e in app.store.list_events(outcome.run_id) if e["kind"] == "step_replayed"],
        "checkpoint_rows": rows, "checkpoint_payload_bytes": payload,
        "report": app.store.get_report(outcome.run_id)["markdown_path"],
    }, indent=1))
    engine.close()
    app.close()


if __name__ == "__main__":
    main()
