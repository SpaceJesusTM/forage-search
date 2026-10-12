"""Normalised behaviour traces, used to check that two engines did the same research.

Identifiers derived from the (random) watch id, run ids and timestamps are masked, so
two runs of the same scenario can be compared for equality.
"""

from __future__ import annotations

import json
import re
from typing import Any

_ID = re.compile(r"\b(?:[qfsc]-[0-9a-f]{8,}|run-[0-9a-f]{6,}|rev-[0-9a-f]{6,}|resp-[0-9a-f]{6,})\b")
_STAMP = re.compile(r"\b20\d\d-\d\d-\d\dT\d\d:\d\d:\d\dZ\b")
VOLATILE_METRICS = {"wall_seconds", "llm_seconds"}


def mask(value: Any, watch_ids: list[str]) -> Any:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    for i, watch_id in enumerate(watch_ids):
        text = text.replace(watch_id, f"<watch{i}>")
    text = _ID.sub("<id>", text)
    text = _STAMP.sub("<t>", text)
    return json.loads(text)


def collect(app, web, llm, watch_ids: list[str], prompts: bool = True) -> dict[str, Any]:
    """Everything observable about what the engine did, in a comparable form."""
    store = app.store
    out: dict[str, Any] = {
        "searches": list(web.search_log),
        "fetches": list(web.fetch_log),
        "model_calls": [(purpose, call["user"]) if prompts else purpose for purpose, call in llm.calls],
        "watches": [],
    }
    for watch_id in watch_ids:
        runs = []
        for run in store.list_runs(watch_id):
            metrics = {k: v for k, v in (run["metrics"] or {}).items() if k not in VOLATILE_METRICS}
            report = store.get_report(run["id"])
            body = json.loads(report["json"]) if report else None
            if body:
                for key in ("started_at", "finished_at"):
                    body.pop(key, None)
                body["metrics"] = {k: v for k, v in body["metrics"].items() if k not in VOLATILE_METRICS}
                for source in body.get("sources", []):
                    source.pop("fetched_at", None)
            runs.append({
                "seq": run["seq"], "status": run["status"], "stop_reason": run["stop_reason"], "error": run["error"],
                "meaningful": run["has_meaningful_update"], "metrics": metrics, "report": body,
                "events": [(e["kind"], e["data"]) for e in store.list_events(run["id"])],
                "search_attempts": [(a["query"], a["status"], a["result_count"], a["new_url_count"], a["known_url_count"])
                                    for a in store.list_search_attempts(watch_id, run["id"])],
                "llm_calls": [(c["purpose"], c["prompt_tokens"], c["output_tokens"], c["ok"]) for c in store.list_llm_calls(run["id"])],
                "changes": [(c["change_type"], c["detail"]) for c in store.list_changes(run["id"])],
            })
        out["watches"].append({
            "status": store.get_watch(watch_id)["status"],
            "runs": runs,
            "sources": [(s["canonical_url"], s["status"], s["relevance"], s["fetch_count"], s["flags"]) for s in store.list_sources(watch_id)],
            "claims": [(c["text"], c["status"], c["verification"], c["support"], c["note"]) for c in store.list_claims(watch_id=watch_id, status=None)],
            "findings": [(f["title"], f["status"]) for f in store.list_findings(watch_id)],
            "questions": [(q["text"], q["status"], q["attempts"], q["resolution"]) for q in store.list_questions(watch_id)],
            "leads": [(lead["text"], lead["status"]) for lead in store.list_leads(watch_id, status=None)],
        })
    return mask(out, watch_ids)


def first_difference(a: Any, b: Any, path: str = "") -> str | None:
    """Human-readable location of the first difference between two traces, or None if equal."""
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f"{path}/{key}: present in only one"
            found = first_difference(a[key], b[key], f"{path}/{key}")
            if found:
                return found
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            found = first_difference(x, y, f"{path}[{i}]")
            if found:
                return found
        return None
    return None if a == b else f"{path}: {str(a)[:140]!r} != {str(b)[:140]!r}"


def research_state(store, watch_id: str, report: dict[str, Any] | None = None) -> dict[str, Any]:
    """What the research produced, independent of how many attempts it took to get there."""
    findings = {f["id"]: f["title"] for f in store.list_findings(watch_id)}
    last = store.list_runs(watch_id)[-1]
    if report is None:
        row = store.get_report(last["id"])
        report = json.loads(row["json"]) if row else {"findings": []}
    return {
        "claims": sorted((c["text"], c["status"], c["verification"], c["support"])
                         for c in store.list_claims(watch_id=watch_id, status=None)),
        "findings": sorted((f["title"], f["status"]) for f in store.list_findings(watch_id)),
        "changes": sorted((findings[c["finding_id"]], c["change_type"]) for c in store.list_changes(last["id"])),
        "questions": sorted((q["text"], q["status"]) for q in store.list_questions(watch_id)),
        "sources": sorted((s["canonical_url"], s["status"], s["relevance"]) for s in store.list_sources(watch_id)),
        "run": (last["status"], last["stop_reason"], bool(last["has_meaningful_update"])),
        "report": [(f["title"], f["change_type"], len(f["claims"])) for f in report["findings"]],
    }
