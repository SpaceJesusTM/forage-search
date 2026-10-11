"""Replayable comparison of checkpoint policies on a SIMULATED scenario.

Each policy gets a fresh database and the same simulated web, model, budgets,
scheduler rules, clock and scripted user. Only the checkpoint policy differs. The
output is a table of behaviour and cost, with raw counts next to every ratio.

What this can show: how each policy allocates sessions, when it asks, what the
scripted user's answers change, and what is found per unit of budget on this designed
scenario. What it cannot show: that any policy is better on real research, or anything
about how people experience being asked.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from forage.app import build_app
from forage.config import Settings
from forage.controller.watches import apply_spec_patch, spec_diff
from forage.eval.export import export_all
from forage.fetch.urls import canonicalize_url
from forage.scheduler.queue import Scheduler
from forage.simulated import FixtureFetcher, FixtureSearch, SimulatedLLM, SimulatedWeb
from forage.types import ClaimStatus, WatchSpec

POLICIES = ("continuous", "periodic", "adaptive")


class SimClock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


def _night_start(night: int, tz_name: str, start_hour: str) -> datetime:
    hour, minute = (int(part) for part in start_hour.split(":"))
    base = datetime(2026, 10, 12, hour, minute, tzinfo=ZoneInfo(tz_name)) + timedelta(days=night)
    return base.astimezone(timezone.utc)


def _condition(when: str, runs: list[dict[str, Any]]) -> bool:
    """Scripted-user conditions, evaluated on what the user could see in the watch's reports."""
    if when == "always":
        return True
    if when == "scope_pressure":
        return sum((r["metrics"] or {}).get("out_of_scope_candidates", 0) for r in runs) > 0
    if when.startswith("dry_streak>="):
        need = int(when.split(">=")[1])
        streak = 0
        for run in reversed([r for r in runs if r["status"] == "COMPLETED"]):
            if run["has_meaningful_update"]:
                break
            streak += 1
        return streak >= need
    raise ValueError(f"unknown scripted-user condition '{when}'")


def _scripted_answer(rules: dict[str, Any], key: str, runs: list[dict[str, Any]], spec: WatchSpec) -> dict[str, Any]:
    for rule in [*rules.get(key, []), *rules.get("*", [])]:
        if not _condition(rule["when"], runs):
            continue
        if rule.get("patch") and not spec_diff(spec, apply_spec_patch(spec, rule["patch"])):
            continue  # the user already made this change; the rule has nothing left to do
        return rule
    return {"when": "always", "choice": "continue"}


def run_policy(scenario: dict[str, Any], policy: str, data_dir: Path) -> dict[str, Any]:
    if data_dir.exists():
        shutil.rmtree(data_dir)
    budget = scenario.get("budget", {})
    settings = Settings(data_dir=data_dir, search_provider="fixture", model="simulated", search_min_interval_seconds=0.0,
                        min_minutes_between_sessions=0, **budget)
    clock = SimClock(_night_start(0, settings.night_timezone, settings.night_start))
    web = SimulatedWeb(json.loads(json.dumps(scenario["web"]["docs"])))
    app = build_app(settings, llm=SimulatedLLM(), search=FixtureSearch(web), fetcher=FixtureFetcher(web), clock=clock,
                    sleep=lambda seconds: None)
    scheduler = Scheduler(app.store, settings, app.runner, clock=clock, sleep=lambda seconds: None, owner="experiment")
    config = (scenario.get("policy_config") or {}).get(policy, {})
    ids: dict[str, str] = {}
    for spec in scenario["watches"]:
        watch = app.watches.create(spec["objective"], title=spec["key"], spec=WatchSpec(**spec.get("spec", {})),
                                   policy=policy, policy_config=config, actor="scripted-user")
        app.watches.approve(watch["id"], actor="scripted-user")
        ids[spec["key"]] = watch["id"]
    keys = {watch_id: key for key, watch_id in ids.items()}

    user = scenario.get("user", {})
    delay = int(user.get("delay_nights", 1))
    asked_on: dict[str, int] = {}
    answers: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    for night in range(int(scenario["nights"])):
        clock.now = _night_start(night, settings.night_timezone, settings.night_start)
        web.epoch = scenario["epoch_by_night"][night]
        # The scripted user answers what was asked at least `delay` nights ago.
        for request in app.reviews.list(status="PENDING"):
            if night - asked_on.get(request["id"], night) < delay:
                continue
            key = keys[request["watch_id"]]
            _, current_spec = app.watches.active_spec(app.store.get_watch(request["watch_id"]))
            rule = _scripted_answer(user.get("rules", {}), key, app.store.list_runs(request["watch_id"]), current_spec)
            result = app.reviews.respond(request["id"], rule["choice"], patch=rule.get("patch"), confirm_scope_change=True,
                                         actor="scripted-user", note=f"scripted rule: {rule['when']}")
            answers.append({"night": night, "watch": key, "signal": request["signal"], "choice": rule["choice"],
                            "changed_something": rule["choice"] != "continue" or bool(result["revision_id"])})
        scheduler.acquire_lease(3600)
        for _ in range(int(scenario["sessions_per_night"])):
            result = scheduler.run_next(trigger="night")
            if result is None:
                break
            timeline.append({"night": night, "watch": keys[result["watch_id"]], "status": result["status"],
                             "meaningful": result["meaningful"]})
            for request in app.reviews.list(watch_id=result["watch_id"], status="PENDING"):
                asked_on.setdefault(request["id"], night)
            clock.advance(minutes=float(scenario.get("session_minutes", 30)))
        scheduler.release_lease()

    result = _measure(app, scenario, policy, ids, answers, timeline)
    export_all(app.store, data_dir / "export")
    app.close()
    return result


def _measure(app, scenario: dict[str, Any], policy: str, ids: dict[str, str], answers: list[dict[str, Any]],
             timeline: list[dict[str, Any]]) -> dict[str, Any]:
    labels = {canonicalize_url(d["url"]): d.get("labels", {}) for d in scenario["web"]["docs"]}
    slots = int(scenario["nights"]) * int(scenario["sessions_per_night"])
    per_watch: dict[str, Any] = {}
    for key, watch_id in ids.items():
        store = app.store
        runs = store.list_runs(watch_id)
        metrics = [r["metrics"] or {} for r in runs]
        amended = store.scalar(
            "SELECT COUNT(*) FROM watch_revision WHERE watch_id = ? AND origin = 'checkpoint_response' AND state = 'APPROVED'",
            (watch_id,)) > 0
        with_claims = {row["canonical_url"] for row in store.all(
            "SELECT DISTINCT s.canonical_url FROM claim c JOIN source s ON s.id = c.source_id WHERE c.watch_id = ? AND c.status = ?",
            (watch_id, ClaimStatus.ACCEPTED))}
        relevant = [u for u in with_claims if labels.get(u, {}).get("watch") == key and (
            labels[u].get("counts") == "always" or (labels[u].get("counts") == "if_scope_amended" and amended))]
        unapproved = [u for u in with_claims if labels.get(u, {}).get("counts") == "if_scope_amended" and not amended]
        off_topic = [u for u in with_claims if labels.get(u, {}).get("watch") != key]
        available = [u for u, lab in labels.items() if lab.get("watch") == key]
        requests = store.all("SELECT id, signal, status FROM checkpoint_request WHERE watch_id = ? ORDER BY rowid", (watch_id,))
        mine = [a for a in answers if a["watch"] == key]
        searches = sum(m.get("search_requests", 0) for m in metrics)
        tokens = sum(m.get("llm_output_tokens", 0) for m in metrics)
        per_watch[key] = {
            "final_status": store.get_watch(watch_id)["status"],
            "sessions": len(runs),
            "sessions_with_update": sum(1 for r in runs if r["has_meaningful_update"]),
            "sessions_without_update": sum(1 for r in runs if r["status"] == "COMPLETED" and not r["has_meaningful_update"]),
            "search_requests": searches, "pages_fetched": sum(m.get("fetched_pages", 0) for m in metrics),
            "llm_output_tokens": tokens,
            "relevant_pages_available": len(available),
            "relevant_pages_verified": len(relevant),
            "pages_verified_outside_approved_scope": len(unapproved),
            "off_topic_pages_verified": len(off_topic),
            "scope_amended_by_user": bool(amended),
            "reviews_requested": len(requests),
            "reviews_by_signal": {s: sum(1 for r in requests if r["signal"] == s) for s in sorted({r["signal"] for r in requests})},
            "reviews_that_changed_something": sum(1 for a in mine if a["changed_something"]),
            "reviews_answered_continue": sum(1 for a in mine if not a["changed_something"]),
            "reviews_unanswered_at_end": sum(1 for r in requests if r["status"] == "PENDING"),
            "turns_skipped_while_awaiting_review": int(store.scalar(
                "SELECT COUNT(*) FROM queue_admission WHERE watch_id = ? AND decision = 'SKIPPED' AND reason = 'status AWAITING_REVIEW'",
                (watch_id,)) or 0),
            "relevant_pages_per_100_searches": round(100 * len(relevant) / searches, 1) if searches else None,
        }
    totals: dict[str, Any] = {"session_slots_available": slots, "sessions_run": len(timeline)}
    for field in ("search_requests", "pages_fetched", "llm_output_tokens", "relevant_pages_available", "relevant_pages_verified",
                  "pages_verified_outside_approved_scope", "off_topic_pages_verified", "sessions_without_update",
                  "reviews_requested", "reviews_that_changed_something", "reviews_answered_continue",
                  "turns_skipped_while_awaiting_review"):
        totals[field] = sum(w[field] for w in per_watch.values())
    totals["slots_unused"] = slots - len(timeline)
    totals["relevant_pages_per_100_searches"] = (
        round(100 * totals["relevant_pages_verified"] / totals["search_requests"], 1) if totals["search_requests"] else None)
    totals["relevant_pages_per_1k_output_tokens"] = (
        round(1000 * totals["relevant_pages_verified"] / totals["llm_output_tokens"], 2) if totals["llm_output_tokens"] else None)
    totals["user_decisions_requested"] = totals["reviews_requested"]
    return {"policy": policy, "totals": totals, "per_watch": per_watch, "user_answers": answers, "timeline": timeline}


def run_experiment(scenario: dict[str, Any], out_dir: Path, policies: tuple[str, ...] = POLICIES) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [run_policy(scenario, policy, out_dir / policy) for policy in policies]
    summary = {
        "simulated": True,
        "notice": scenario.get("_notice", "SIMULATED scenario."),
        "scenario": scenario.get("name", "custom"),
        "design": {k: scenario[k] for k in ("nights", "sessions_per_night", "budget", "policy_config") if k in scenario},
        "scripted_user": scenario.get("user", {}).get("description"),
        "results": results,
        "what_this_cannot_show": [
            "The scenario was designed to contain one watch that needs a scope decision, one that needs none and one that dries up; "
            "a policy tuned to those cases will look good on it.",
            "The model is a rule-based stand-in, so research quality is not being measured.",
            "The user is scripted. Nothing here speaks to how people experience being asked, or to trust, control or burden.",
            "One deterministic run per policy: no variance, no seeds, no statistics.",
        ],
    }
    (out_dir / "comparison.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "comparison.md").write_text(render_comparison(summary), encoding="utf-8")
    return summary


_ROWS = (
    ("Sessions run (of slots available)", lambda t: f"{t['sessions_run']} / {t['session_slots_available']}"),
    ("Search requests", lambda t: t["search_requests"]),
    ("Model output tokens", lambda t: t["llm_output_tokens"]),
    ("Relevant pages verified (of available)", lambda t: f"{t['relevant_pages_verified']} / {t['relevant_pages_available']}"),
    ("Relevant pages per 100 searches", lambda t: t["relevant_pages_per_100_searches"]),
    ("Relevant pages per 1k output tokens", lambda t: t["relevant_pages_per_1k_output_tokens"]),
    ("Sessions with nothing new", lambda t: t["sessions_without_update"]),
    ("Pages verified outside approved scope", lambda t: t["pages_verified_outside_approved_scope"]),
    ("Reviews requested (user decisions asked)", lambda t: t["reviews_requested"]),
    ("  ...that changed something", lambda t: t["reviews_that_changed_something"]),
    ("  ...answered 'continue'", lambda t: t["reviews_answered_continue"]),
    ("Turns skipped while awaiting review", lambda t: t["turns_skipped_while_awaiting_review"]),
)


def render_comparison(summary: dict[str, Any]) -> str:
    results = summary["results"]
    names = [r["policy"] for r in results]
    lines = [f"# Checkpoint policy comparison — scenario `{summary['scenario']}`", "",
             f"> **SIMULATED.** {summary['notice']} The model is a rule-based stand-in and the user is scripted.", "",
             f"Design: {summary['design'].get('nights')} nights x {summary['design'].get('sessions_per_night')} session slots, "
             f"identical web, model, budgets, scheduler and user rules under every policy.", ""]
    if summary.get("scripted_user"):
        lines += [f"Scripted user: {summary['scripted_user']}", ""]
    lines += ["## Totals", "", "| | " + " | ".join(names) + " |", "| --- | " + " | ".join("---:" for _ in names) + " |"]
    for label, getter in _ROWS:
        lines.append(f"| {label} | " + " | ".join(str(getter(r["totals"])) for r in results) + " |")
    lines += ["", "## Per watch", ""]
    keys = list(results[0]["per_watch"])
    for key in keys:
        lines += [f"### {key}", "", "| | " + " | ".join(names) + " |", "| --- | " + " | ".join("---:" for _ in names) + " |"]
        rows = (
            ("Sessions", "sessions"), ("Sessions with nothing new", "sessions_without_update"),
            ("Search requests", "search_requests"), ("Relevant pages verified", "relevant_pages_verified"),
            ("Relevant pages available", "relevant_pages_available"), ("Reviews requested", "reviews_requested"),
            ("Reviews that changed something", "reviews_that_changed_something"),
            ("Turns skipped awaiting review", "turns_skipped_while_awaiting_review"),
            ("Scope amended by user", "scope_amended_by_user"), ("Final status", "final_status"),
        )
        for label, field in rows:
            lines.append(f"| {label} | " + " | ".join(str(r["per_watch"][key][field]) for r in results) + " |")
        signals = [", ".join(f"{s}×{n}" for s, n in r["per_watch"][key]["reviews_by_signal"].items()) or "-" for r in results]
        lines += ["| Review signals | " + " | ".join(signals) + " |", ""]
    lines += ["## What this cannot show", ""] + [f"- {note}" for note in summary["what_this_cannot_show"]] + [""]
    return "\n".join(lines)
