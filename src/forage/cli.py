"""`forage` command line. Presentation only: every command delegates to the controller, run manager or store."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from forage import __version__
from forage.app import App, build_app
from forage.config import Settings, load_settings
from forage.controller.watches import CHECKPOINT_POLICIES, ConfirmationRequired, WatchError
from forage.llm.base import LLMError
from forage.report.build import write_report
from forage.scheduler.runner import RunRefused
from forage.search.base import SearchError
from forage.storage.store import Store
from forage.types import ClaimStatus, WatchSpec


def _out(text: str = "") -> None:
    print(text)


def _json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _store(settings: Settings) -> Store:
    return Store(settings.db_path)


def _app(settings: Settings, progress: bool = False) -> App:
    return build_app(settings, progress=(lambda m: print(f"  · {m}", flush=True)) if progress else None)


def _patch_from_args(args: argparse.Namespace) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    for flag, field in (("question", "key_questions"), ("include", "include"), ("exclude", "exclude"),
                        ("priority", "priorities"), ("hint", "source_hints")):
        if getattr(args, flag, None):
            patch[f"{field}_add"] = getattr(args, flag)
        if getattr(args, f"remove_{flag}", None):
            patch[f"{field}_remove"] = getattr(args, f"remove_{flag}")
    for field in ("geography", "freshness_days", "notes"):
        if getattr(args, field, None) is not None:
            patch[field] = getattr(args, field)
    return patch


def _add_spec_flags(parser: argparse.ArgumentParser, removal: bool = False) -> None:
    parser.add_argument("--question", action="append", metavar="TEXT", help="key question to answer (repeatable)")
    parser.add_argument("--include", action="append", metavar="TEXT", help="in-scope constraint (repeatable)")
    parser.add_argument("--exclude", action="append", metavar="TEXT", help="out-of-scope constraint (repeatable)")
    parser.add_argument("--priority", action="append", metavar="TEXT", help="what matters most (repeatable)")
    parser.add_argument("--hint", action="append", metavar="URL", help="site or URL to consult (repeatable)")
    parser.add_argument("--geography", help="geographic limit")
    parser.add_argument("--freshness-days", type=int, dest="freshness_days", help="re-check sources older than N days")
    parser.add_argument("--notes", help="free-text guidance for the researcher")
    if removal:
        for flag in ("question", "include", "exclude", "priority", "hint"):
            parser.add_argument(f"--remove-{flag}", action="append", dest=f"remove_{flag}", metavar="TEXT",
                                help=f"remove an existing {flag} (exact text)")


def _show_spec(objective: str, spec: WatchSpec, indent: str = "  ") -> None:
    _out(f"{indent}objective: {objective}")
    for label, values in (("key questions", spec.key_questions), ("in scope", spec.include),
                          ("out of scope", spec.exclude), ("priorities", spec.priorities),
                          ("source hints", spec.source_hints)):
        if values:
            _out(f"{indent}{label}:")
            for value in values:
                _out(f"{indent}  - {value}")
    if spec.geography:
        _out(f"{indent}geography: {spec.geography}")
    if spec.freshness_days:
        _out(f"{indent}re-check sources after: {spec.freshness_days} days")
    if spec.notes:
        _out(f"{indent}notes: {spec.notes}")


# ---------------------------------------------------------------- commands


def cmd_doctor(args: argparse.Namespace, settings: Settings) -> int:
    from forage.llm.ollama import OllamaClient, select_model

    _out(f"forage {__version__}")
    _out(f"data dir        {settings.data_dir.resolve()}")
    store = _store(settings)
    _out(f"database        {settings.db_path} (schema v{store.schema_version()}, {len(store.list_watches())} watches)")
    running = store.running_run()
    if running:
        _out(f"running session {running['id']} (heartbeat {running['heartbeat_at']})")
    ok = True
    if settings.model == "simulated":
        _out("model           simulated (no real inference)")
    else:
        client = OllamaClient(settings.ollama_url, settings.model, num_ctx=settings.num_ctx)
        try:
            version = client.version()
            models = client.list_models()
            _out(f"ollama          {settings.ollama_url} (version {version})")
            for model in models:
                _out(f"  installed     {model.name}  {model.parameter_size}  {model.size_gb:.1f} GB")
            chosen = select_model(models, settings.model, settings.max_model_gb)
            _out(f"model selected  {chosen}" + ("" if settings.model else "  (largest installed under max_model_gb)"))
            if args.live:
                from pydantic import BaseModel

                class Ping(BaseModel):
                    answer: str

                client.model = chosen
                out, usage = client.generate(purpose="doctor", system="Reply with JSON only.",
                                             user='Return {"answer": "ready"}.', schema=Ping, max_tokens=20)
                _out(f"inference       ok: {out.answer!r} in {usage.duration_ms} ms ({usage.output_tokens} tokens)")
        except LLMError as exc:
            ok = False
            _out(f"ollama          PROBLEM: {exc}")
    _out(f"search provider {settings.search_provider}")
    if args.live and settings.search_provider != "fixture":
        from forage.app import build_fetcher, build_search

        try:
            results = build_search(settings).search("open source local language model research agent", 3)
            _out(f"search          ok: {len(results)} results" + (f", first: {results[0].url}" if results else ""))
            page = build_fetcher(settings).fetch("https://example.com/")
            _out(f"fetch           {page.outcome}: {len(page.text)} chars from example.com")
        except SearchError as exc:
            ok = False
            _out(f"search          PROBLEM: {exc}")
    _out(f"night window    {settings.night_start}-{settings.night_end} {settings.night_timezone}; "
         f"scheduler authority: {settings.scheduler_authority}")
    return 0 if ok else 1


def cmd_watch_create(args: argparse.Namespace, settings: Settings) -> int:
    patch = _patch_from_args(args)
    explicit = bool(patch) or args.no_llm
    spec = None
    llm = None
    if explicit:
        from forage.controller.watches import apply_spec_patch

        spec = apply_spec_patch(WatchSpec(), patch)
    store = _store(settings)
    if not explicit:
        from forage.app import build_llm

        try:
            llm = build_llm(settings)
            _out(f"Structuring the objective with {llm.model} ...")
        except LLMError as exc:
            _out(f"Local model unavailable ({exc}); creating the watch from your words only.")
    from forage.controller.watches import WatchController

    watch = WatchController(store).create(args.objective, title=args.title, spec=spec, llm=llm, policy=args.policy)
    _out(f"Created watch {watch['id']} — status {watch['status']}")
    _print_watch(store, watch)
    _out("")
    _out("Nothing is researched until you confirm this brief:")
    _out(f"  forage watch approve {watch['id']}")
    _out(f"  forage watch edit {watch['id']} --question ... --exclude ...   (to adjust first)")
    return 0


def _print_watch(store: Store, watch: dict[str, Any]) -> None:
    _out(f"{watch['id']}  [{watch['status']}]  {watch['title']}")
    _out(f"  policy: {watch['checkpoint_policy']}   queue position: {watch['queue_position']}")
    active = store.get_revision(watch["active_revision_id"])
    pending = store.get_revision(watch["pending_revision_id"])
    if active:
        _out(f"  approved brief (revision {active['revision_no']}, approved {active['decided_at']} by {active['decided_by']}):")
        _show_spec(active["objective"], WatchSpec.model_validate(active["spec"]), "    ")
    if pending:
        _out(f"  PROPOSED brief awaiting your confirmation (revision {pending['revision_no']}, origin {pending['origin']}):")
        _show_spec(pending["objective"], WatchSpec.model_validate(pending["spec"]), "    ")
        if pending["note"]:
            _out(f"    note: {pending['note']}")


def cmd_watch_list(args: argparse.Namespace, settings: Settings) -> int:
    store = _store(settings)
    watches = store.list_watches()
    if not watches:
        _out("No watches yet. Create one with: forage watch create \"<what to research>\"")
        return 0
    for watch in watches:
        runs = store.list_runs(watch["id"])
        findings = store.scalar("SELECT COUNT(*) FROM finding WHERE watch_id = ?", (watch["id"],))
        last = runs[-1]["started_at"] if runs else "never"
        _out(f"{watch['id']:<36} {watch['status']:<21} {watch['checkpoint_policy']:<10} "
             f"sessions={len(runs):<3} findings={findings:<3} last={last}  {watch['title']}")
    return 0


def cmd_watch_show(args: argparse.Namespace, settings: Settings) -> int:
    from forage.controller.reviews import ReviewController
    from forage.controller.watches import WatchController

    store = _store(settings)
    watch = WatchController(store).get(args.watch)
    _print_watch(store, watch)
    watch_id = watch["id"]
    runs = store.list_runs(watch_id)
    counts = {
        "findings": store.scalar("SELECT COUNT(*) FROM finding WHERE watch_id = ?", (watch_id,)),
        "verified claims": store.scalar("SELECT COUNT(*) FROM claim WHERE watch_id = ? AND status = 'ACCEPTED'", (watch_id,)),
        "sources read": store.scalar("SELECT COUNT(*) FROM source WHERE watch_id = ? AND status = 'FETCHED'", (watch_id,)),
        "open questions": len(store.list_questions(watch_id, "OPEN")),
        "sessions": len(runs),
    }
    _out("  " + "   ".join(f"{k}: {v}" for k, v in counts.items()))
    pending = ReviewController(store).pending_for(watch_id)
    if pending:
        _out(f"  PENDING DECISION ({pending['id']}): {pending['question']}")
        _out(f"    answer with: forage review show {pending['id']}")
    if args.history:
        _out("  history:")
        for event in store.list_watch_events(watch_id):
            move = f"{event['from_status'] or '-'} -> {event['to_status'] or '-'}"
            _out(f"    {event['at']}  {event['kind']:<18} {move:<40} by {event['actor']}")
    return 0


def cmd_watch_edit(args: argparse.Namespace, settings: Settings) -> int:
    from forage.controller.watches import WatchController

    store = _store(settings)
    watch = WatchController(store).propose_revision(args.watch, patch=_patch_from_args(args), objective=args.objective,
                                                    title=args.title)
    _print_watch(store, watch)
    _out(f"\nConfirm with: forage watch approve {watch['id']}")
    return 0


def cmd_watch_simple(args: argparse.Namespace, settings: Settings) -> int:
    from forage.controller.watches import WatchController

    store = _store(settings)
    controller = WatchController(store)
    action = args.action
    if action == "approve":
        watch = controller.approve(args.watch)
        _out(f"Approved. Watch {watch['id']} is {watch['status']} and may now be researched: forage run {watch['id']}")
    elif action == "pause":
        watch = controller.pause(args.watch)
        _out(f"Paused {watch['id']}. It will receive no research turns until you resume it.")
    elif action == "close":
        watch = controller.close(args.watch)
        _out(f"Closed {watch['id']}. History is kept; reopen with: forage watch reopen {watch['id']}")
    elif action in ("resume", "reopen"):
        watch = controller.request_resume(args.watch, patch=_patch_from_args(args))
        _out(f"{watch['id']} is {watch['status']}. Review the plan, then confirm or decline:")
        _json(controller.resume_plan(watch["id"]))
        _print_watch(store, watch)
        _out(f"\n  forage watch approve {watch['id']}    (reactivate)\n  forage watch decline {watch['id']}    (leave as it was)")
    elif action == "decline":
        watch = controller.decline(args.watch)
        _out(f"Declined. Watch {watch['id']} is {watch['status']}.")
    elif action == "policy":
        watch = controller.set_policy(args.watch, args.policy, json.loads(args.config) if args.config else None)
        _out(f"Watch {watch['id']} now uses the '{watch['checkpoint_policy']}' checkpoint policy.")
    return 0


def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    overrides = {
        "max_duration_seconds": args.max_seconds, "max_search_requests": args.max_searches,
        "max_fetched_pages": args.max_pages, "max_llm_output_tokens": args.max_llm_tokens,
    }
    app = _app(settings, progress=not args.quiet)
    watch = app.watches.get(args.watch)
    budget = app.runner.resolve_budget(watch, overrides)
    _out(f"Session for {watch['id']} — model {app.llm.model}, search {app.search.name}")
    _out(f"Budget: {budget.max_duration_seconds}s, {budget.max_search_requests} searches, "
         f"{budget.max_fetched_pages} pages, {budget.max_llm_output_tokens} model output tokens")
    try:
        outcome = app.runner.run_watch(watch["id"], trigger="manual", overrides=overrides, run_id=args.run_id)
    except KeyboardInterrupt:
        app.store.execute("UPDATE session_run SET cancel_requested = 1 WHERE status = 'RUNNING'")
        recovered = app.runner.recover(force=True)
        _out(f"\nInterrupted. Closed out {len(recovered)} session(s); stored evidence is kept.")
        return 130
    report = app.store.get_report(outcome.run_id)
    _out("")
    _out(f"Run {outcome.run_id}: {outcome.status}" + (f" — {outcome.error}" if outcome.error else ""))
    if report:
        data = json.loads(report["json"])
        _out(data["headline"])
        _out(f"Report: {report['markdown_path']}")
        _out(f"        {report['json_path']}")
        if data.get("pending_review"):
            _out(f"\nDECISION NEEDED: {data['pending_review']['question']}")
            _out(f"  forage review show {data['pending_review']['request_id']}")
    return 0 if outcome.status in ("COMPLETED", "CANCELLED") else 1


def _resolve_run(store: Store, watch_id: str, ref: str | None) -> dict[str, Any] | None:
    runs = store.list_runs(watch_id)
    if not runs:
        return None
    if ref is None:
        return runs[-1]
    if ref.isdigit():
        return next((r for r in runs if r["seq"] == int(ref)), None)
    return next((r for r in runs if r["id"] == ref or r["id"].startswith(ref)), None)


def cmd_report(args: argparse.Namespace, settings: Settings) -> int:
    from forage.controller.watches import WatchController

    store = _store(settings)
    watch = WatchController(store).get(args.watch)
    run = _resolve_run(store, watch["id"], args.run)
    if run is None:
        _out("No matching session.")
        return 1
    if args.rebuild:
        write_report(store, settings, run["id"])
    report = store.get_report(run["id"])
    if report is None:
        _out(f"Session {run['id']} has no report (status {run['status']}).")
        return 1
    _out(report["json"] if args.format == "json" else report["markdown"])
    return 0


def cmd_runs(args: argparse.Namespace, settings: Settings) -> int:
    from forage.controller.watches import WatchController

    store = _store(settings)
    watch = WatchController(store).get(args.watch)
    _out(f"{'#':<3} {'run id':<18} {'status':<12} {'started':<21} {'secs':>6} {'srch':>4} {'pages':>5} "
         f"{'tokens':>6} {'new':>3} {'upd':>3} {'dup':>4}  stop")
    for run in store.list_runs(watch["id"]):
        m = run["metrics"] or {}
        _out(f"{run['seq']:<3} {run['id']:<18} {run['status']:<12} {run['started_at']:<21} "
             f"{m.get('wall_seconds', 0):>6} {m.get('search_requests', 0):>4} {m.get('fetched_pages', 0):>5} "
             f"{m.get('llm_output_tokens', 0):>6} {m.get('findings_new', 0):>3} {m.get('findings_updated', 0):>3} "
             f"{m.get('duplicate_urls_skipped', 0) + m.get('duplicate_claims_suppressed', 0):>4}  {run['stop_reason'] or ''}")
    return 0


def cmd_findings(args: argparse.Namespace, settings: Settings) -> int:
    from forage.controller.watches import WatchController

    store = _store(settings)
    watch = WatchController(store).get(args.watch)
    sources = {s["id"]: s for s in store.list_sources(watch["id"])}
    seqs = {r["id"]: r["seq"] for r in store.list_runs(watch["id"])}
    findings = store.list_findings(watch["id"])
    if not findings:
        _out("No findings yet.")
        return 0
    for finding in findings:
        claims = store.list_claims(finding_id=finding["id"], status=ClaimStatus.ACCEPTED)
        domains = {sources[c["source_id"]]["domain"] for c in claims if c["source_id"] in sources}
        _out(f"{finding['id']}  [{finding['status']}]  {finding['title']}")
        _out(f"  first seen session {seqs.get(finding['first_seen_run_id'], '?')}, last changed session "
             f"{seqs.get(finding['last_changed_run_id'], '?')}, {len(domains)} source site(s)")
        for claim in claims:
            source = sources.get(claim["source_id"], {})
            _out(f"  - {claim['text']}")
            if args.evidence:
                _out(f"      \"{claim['quote']}\"")
                _out(f"      {source.get('final_url') or source.get('url')}  [{claim['verification']}, {claim['support']}, "
                     f"checked {claim['last_confirmed_at']}]")
        if finding["uncertainty"]:
            _out(f"  uncertainty (model note): {finding['uncertainty']}")
    return 0


def cmd_sources(args: argparse.Namespace, settings: Settings) -> int:
    from forage.controller.watches import WatchController

    store = _store(settings)
    watch = WatchController(store).get(args.watch)
    for source in store.list_sources(watch["id"]):
        if args.status and source["status"] != args.status.upper():
            continue
        flags = f" flags={','.join(source['flags'])}" if source["flags"] else ""
        _out(f"{source['status']:<10} rel={source['relevance'] if source['relevance'] is not None else '-'} "
             f"fetched={source['last_fetched_at'] or '-':<20} {source['canonical_url']}{flags}")
    return 0


def cmd_questions(args: argparse.Namespace, settings: Settings) -> int:
    from forage.controller.watches import WatchController

    store = _store(settings)
    watch = WatchController(store).get(args.watch)
    if args.reopen:
        store.update_question(args.reopen, status="OPEN", resolution=None, resolved_by=[], resolved_run_id=None, resolved_at=None)
        _out(f"Reopened {args.reopen}.")
    for question in store.list_questions(watch["id"]):
        _out(f"{question['id']}  [{question['status']}] ({question['origin']}, searched {question['attempts']}x)  {question['text']}")
        if question["status"] == "RESOLVED":
            _out(f"    model's answer: {question['resolution']}  (based on {', '.join(question['resolved_by'] or [])})")
    leads = store.list_leads(watch["id"], "OPEN")
    if leads:
        _out("\nSaved directions for the next session:")
        for lead in leads:
            _out(f"  - {lead['text']}" + (f"  ({lead['rationale']})" if lead["rationale"] else ""))
    return 0


def cmd_compare(args: argparse.Namespace, settings: Settings) -> int:
    from forage.controller.watches import WatchController
    from forage.report.compare import compare_runs, render_comparison

    store = _store(settings)
    watch = WatchController(store).get(args.watch)
    runs = [r for r in store.list_runs(watch["id"]) if store.get_report(r["id"])]
    if len(runs) < 2 and not (args.a and args.b):
        _out("Need at least two sessions with reports to compare.")
        return 1
    run_a = _resolve_run(store, watch["id"], args.a) if args.a else runs[-2]
    run_b = _resolve_run(store, watch["id"], args.b) if args.b else runs[-1]
    if run_a is None or run_b is None:
        _out("No matching sessions.")
        return 1
    comparison = compare_runs(store, run_a["id"], run_b["id"])
    if args.format == "json":
        _json(comparison)
    else:
        _out(render_comparison(comparison))
    return 0


def cmd_budget(args: argparse.Namespace, settings: Settings) -> int:
    from forage.controller.watches import WatchController

    store = _store(settings)
    watch = WatchController(store).get(args.watch)
    run = _resolve_run(store, watch["id"], args.run)
    if run is None:
        _out("No matching session.")
        return 1
    m, b = run["metrics"] or {}, run["budget"] or {}
    _out(f"Session {run['seq']} ({run['id']}) — {run['status']}, stop reason: {run['stop_reason']}")
    _out(f"  wall time        {m.get('wall_seconds', 0)} / {b.get('max_duration_seconds')} s")
    _out(f"  search requests  {m.get('search_requests', 0)} / {b.get('max_search_requests')}")
    _out(f"  pages fetched    {m.get('fetched_pages', 0)} / {b.get('max_fetched_pages')}")
    _out(f"  model out tokens {m.get('llm_output_tokens', 0)} / {b.get('max_llm_output_tokens')}")
    _out("  model calls by purpose:")
    by_purpose: dict[str, list[int]] = {}
    for call in store.list_llm_calls(run["id"]):
        row = by_purpose.setdefault(call["purpose"], [0, 0, 0, 0])
        row[0] += 1
        row[1] += call["prompt_tokens"]
        row[2] += call["output_tokens"]
        row[3] += call["duration_ms"]
    _out(f"    {'purpose':<10} {'calls':>5} {'prompt tok':>10} {'output tok':>10} {'seconds':>8}")
    for purpose, (calls, prompt, output, ms) in sorted(by_purpose.items(), key=lambda kv: -kv[1][3]):
        _out(f"    {purpose:<10} {calls:>5} {prompt:>10} {output:>10} {ms / 1000:>8.1f}")
    return 0


def cmd_log(args: argparse.Namespace, settings: Settings) -> int:
    from forage.controller.watches import WatchController

    store = _store(settings)
    watch = WatchController(store).get(args.watch)
    run = _resolve_run(store, watch["id"], args.run)
    if run is None:
        _out("No matching session.")
        return 1
    for event in store.list_events(run["id"]):
        if args.format == "json":
            print(json.dumps(event, ensure_ascii=False))
        else:
            _out(f"{event['at']}  {event['kind']:<22} {json.dumps(event['data'], ensure_ascii=False)[:220]}")
    return 0


def cmd_review(args: argparse.Namespace, settings: Settings) -> int:
    from forage.controller.reviews import ReviewController

    store = _store(settings)
    reviews = ReviewController(store)
    if args.action == "list":
        rows = reviews.list(status=None if args.all else "PENDING")
        if not rows:
            _out("No pending reviews." if not args.all else "No reviews recorded.")
        for row in rows:
            _out(f"{row['id']}  [{row['status']}]  watch {row['watch_id']}  ({row['policy']}/{row['signal']})  {row['question']}")
        return 0
    request = reviews.get(args.request)
    if args.action == "show":
        _out(f"Review {request['id']} — {request['status']} — watch {request['watch_id']}")
        _out(f"Raised by the '{request['policy']}' policy on signal '{request['signal']}' at {request['created_at']}")
        _out(f"\nWhat happened: {request['what_happened']}")
        _out(f"Why this needs you: {request['why_human']}")
        if request["evidence"]:
            _out("Evidence:")
            for item in request["evidence"]:
                _out(f"  - {json.dumps(item, ensure_ascii=False)[:260]}")
        _out(f"\nQuestion: {request['question']}")
        for choice in request["choices"]:
            default = "  (default)" if choice["id"] == request["safe_default"] else ""
            change = "  [changes approved scope]" if choice.get("spec_patch") else ""
            _out(f"  {choice['id']:<10} {choice['label']}{change}{default}")
        _out(f"\nAnswer with: forage review respond {request['id']} --choice <id> [--note ...]")
        return 0
    try:
        result = reviews.respond(request["id"], args.choice, note=args.note, patch=_patch_from_args(args),
                                 confirm_scope_change=args.confirm_scope_change)
    except ConfirmationRequired as exc:
        _out("This answer would change the approved scope:")
        _json(exc.preview)
        _out("Nothing was changed. To apply it, repeat the command with --confirm-scope-change")
        return 2
    _out(f"Recorded. Watch is now {result['watch_status']}." +
         (f" New approved revision: {result['revision_id']}" if result["revision_id"] else " Scope unchanged."))
    return 0


def cmd_queue(args: argparse.Namespace, settings: Settings) -> int:
    from forage.scheduler.queue import Scheduler

    import signal

    from forage.scheduler.queue import SchedulerRefused

    app = _app(settings, progress=args.action != "status" and not getattr(args, "quiet", False))
    scheduler = Scheduler(app.store, app.settings, app.runner)
    if args.action == "status":
        _json(scheduler.status())
        return 0

    def stop(signum, frame):
        # First signal: finish cleanly at the next checkpoint. A second one falls through to the default handler.
        print("\nstop requested: the current session will end at its next checkpoint", flush=True)
        scheduler.request_stop()
        signal.signal(signum, signal.SIG_DFL)

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if args.action == "run":
            results = scheduler.run_batch(max_sessions=args.max_sessions, respect_window=not args.ignore_window,
                                          trigger="queue")
        else:
            results = scheduler.run_night(max_sessions=args.max_sessions, wait=args.wait)
    except SchedulerRefused as exc:
        print(f"forage: {exc}", file=sys.stderr)
        return 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    for line in results:
        _out(line)
    return 0


# ------------------------------------------------------------------ parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forage", description="Forage: local-first persistent web research.")
    parser.add_argument("--version", action="version", version=f"forage {__version__}")
    parser.add_argument("--config", help="path to a forage.toml (default: ./forage.toml if present)")
    parser.add_argument("--data-dir", help="where the database and reports live (default: ./data)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="check the environment: database, Ollama, model, search")
    p.add_argument("--live", action="store_true", help="also run one tiny inference, one search and one fetch")
    p.set_defaults(func=cmd_doctor)

    watch = sub.add_parser("watch", help="create and control research watches").add_subparsers(dest="watch_command", required=True)
    p = watch.add_parser("create", help="create a watch from a natural-language objective")
    p.add_argument("objective")
    p.add_argument("--title")
    p.add_argument("--policy", choices=CHECKPOINT_POLICIES, default="continuous")
    p.add_argument("--no-llm", action="store_true", help="do not ask the local model to structure the objective")
    _add_spec_flags(p)
    p.set_defaults(func=cmd_watch_create)
    p = watch.add_parser("list", help="list watches")
    p.set_defaults(func=cmd_watch_list)
    p = watch.add_parser("show", help="show a watch, its brief and any pending decision")
    p.add_argument("watch")
    p.add_argument("--history", action="store_true", help="include the status/approval history")
    p.set_defaults(func=cmd_watch_show)
    p = watch.add_parser("edit", help="propose a change to the brief (takes effect when approved)")
    p.add_argument("watch")
    p.add_argument("--objective")
    p.add_argument("--title")
    _add_spec_flags(p, removal=True)
    p.set_defaults(func=cmd_watch_edit)
    for action, text in (("approve", "confirm the proposed brief or resume plan"), ("pause", "stop researching now"),
                         ("close", "archive (history kept)"), ("decline", "reject the proposed brief or resume plan")):
        p = watch.add_parser(action, help=text)
        p.add_argument("watch")
        p.set_defaults(func=cmd_watch_simple, action=action)
    for action, text in (("resume", "propose resuming a paused watch"), ("reopen", "propose reopening a closed watch")):
        p = watch.add_parser(action, help=text)
        p.add_argument("watch")
        _add_spec_flags(p, removal=True)
        p.set_defaults(func=cmd_watch_simple, action=action)
    p = watch.add_parser("policy", help="choose the checkpoint policy for a watch")
    p.add_argument("watch")
    p.add_argument("policy", choices=CHECKPOINT_POLICIES)
    p.add_argument("--config", help='policy settings as JSON, e.g. \'{"every": 3}\'')
    p.set_defaults(func=cmd_watch_simple, action="policy")

    p = sub.add_parser("run", help="run one bounded research session for a watch")
    p.add_argument("watch")
    p.add_argument("--max-seconds", type=int)
    p.add_argument("--max-searches", type=int)
    p.add_argument("--max-pages", type=int)
    p.add_argument("--max-llm-tokens", type=int)
    p.add_argument("--run-id", help="caller-issued unique id (single use)")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_run)

    for name, func, text in (("report", cmd_report, "print a session report"), ("budget", cmd_budget, "show where a session's budget went"),
                             ("log", cmd_log, "print a session's event log")):
        p = sub.add_parser(name, help=text)
        p.add_argument("watch")
        p.add_argument("--run", help="session number or run id (default: latest)")
        if name in ("report", "log"):
            p.add_argument("--format", choices=("md", "json") if name == "report" else ("text", "json"),
                           default="md" if name == "report" else "text")
        if name == "report":
            p.add_argument("--rebuild", action="store_true", help="regenerate the report from stored state")
        p.set_defaults(func=func)

    p = sub.add_parser("runs", help="session history with budget use")
    p.add_argument("watch")
    p.set_defaults(func=cmd_runs)
    p = sub.add_parser("findings", help="everything found so far for a watch")
    p.add_argument("watch")
    p.add_argument("--evidence", action="store_true", help="show quotes and source URLs")
    p.set_defaults(func=cmd_findings)
    p = sub.add_parser("sources", help="sources seen for a watch")
    p.add_argument("watch")
    p.add_argument("--status", help="DISCOVERED, FETCHED, FAILED, BLOCKED or IRRELEVANT")
    p.set_defaults(func=cmd_sources)
    p = sub.add_parser("questions", help="open and answered questions, and saved directions")
    p.add_argument("watch")
    p.add_argument("--reopen", metavar="QUESTION_ID", help="mark an answered question as open again")
    p.set_defaults(func=cmd_questions)
    p = sub.add_parser("compare", help="compare two sessions of a watch (default: the last two)")
    p.add_argument("watch")
    p.add_argument("--a", help="earlier session number or run id")
    p.add_argument("--b", help="later session number or run id")
    p.add_argument("--format", choices=("md", "json"), default="md")
    p.set_defaults(func=cmd_compare)

    review = sub.add_parser("review", help="answer guidance requests").add_subparsers(dest="review_command", required=True)
    p = review.add_parser("list", help="list review requests")
    p.add_argument("--all", action="store_true", help="include answered and withdrawn requests")
    p.set_defaults(func=cmd_review, action="list")
    p = review.add_parser("show", help="show a review request")
    p.add_argument("request")
    p.set_defaults(func=cmd_review, action="show")
    p = review.add_parser("respond", help="answer a review request")
    p.add_argument("request")
    p.add_argument("--choice", required=True)
    p.add_argument("--note")
    p.add_argument("--confirm-scope-change", action="store_true",
                   help="required when the answer changes the approved scope")
    _add_spec_flags(p, removal=True)
    p.set_defaults(func=cmd_review, action="respond")

    queue = sub.add_parser("queue", help="multi-watch round-robin runner").add_subparsers(dest="queue_command", required=True)
    p = queue.add_parser("status", help="show the queue, cursor and night window")
    p.set_defaults(func=cmd_queue, action="status")
    p = queue.add_parser("run", help="run a bounded number of queued sessions now")
    p.add_argument("--max-sessions", type=int, default=1)
    p.add_argument("--ignore-window", action="store_true", help="run outside the night window (manual use)")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_queue, action="run")
    p = queue.add_parser("night", help="foreground runner for the night window (does not install any scheduler)")
    p.add_argument("--max-sessions", type=int, default=12)
    p.add_argument("--wait", action="store_true", help="if the window has not opened yet, wait for it")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_queue, action="night")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    if args.data_dir:
        from pathlib import Path

        settings.data_dir = Path(args.data_dir).expanduser()
    try:
        return int(args.func(args, settings) or 0)
    except (WatchError, RunRefused, LLMError, SearchError) as exc:
        print(f"forage: {exc}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
