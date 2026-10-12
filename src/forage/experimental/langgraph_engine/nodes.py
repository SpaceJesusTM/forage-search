"""The session's steps and control flow, independent of what drives them.

    recall -> dispatch -> { fetch -> extract -> integrate } -> dispatch ...
                       -> plan -> dispatch
                       -> search -> triage -> dispatch
                       -> reflect -> report
                                  -> follow_up -> dispatch ... -> reflect

`dispatch` is the only place that decides what happens next. It mirrors the order the
linear engine uses: stale pages, a few saved leads, planned searches, remaining leads,
then the review. Every other step does one unit of work by calling a production
`_Session` method, so no research logic is duplicated here.

State holds ids, counters and short strings. Page text stays in the snapshot table.
Nothing in this module imports LangGraph: `graph.py` wires these steps into a StateGraph,
and `plain_runner.py` runs the same steps from a small loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypedDict

from forage.engine.schemas import Extraction, PlannedQuery
from forage.engine.session import ResearchEngine, _Session
from forage.experimental.langgraph_engine.steps import StepSession
from forage.types import RunStatus

NODE_NAMES = ("recall", "dispatch", "plan", "search", "triage", "fetch", "extract", "integrate", "reflect",
              "follow_up", "report")


class SessionState(TypedDict, total=False):
    run_id: str
    recalled: bool
    # where the session is
    phase: str  # recheck | early_leads | plan | search | follow_up_search | late_leads | wrapup
    route: str  # the node `dispatch` (or a branching node) chose to run next
    recheck_ids: list[str]
    lead_ids: list[str]
    leads_read: int
    plan: list[dict[str, Any]]
    query_index: int
    quota: int
    candidates: list[dict[str, Any]] | None  # search results awaiting triage: source id, rank, title, snippet
    picks: list[str]  # source ids chosen by triage, not yet read
    page: dict[str, Any] | None  # the page being read: source id, snapshot id, a few flags
    extraction: dict[str, Any] | None  # model output awaiting verification
    followup_round: int
    # the session-start snapshot later steps depend on
    recall: dict[str, int]
    memory_view: dict[str, Any] | None
    focus_questions: list[str]
    # tallies the linear engine keeps in attributes
    counters: dict[str, int]
    rejections: dict[str, int]
    out_of_scope: list[dict[str, Any]]
    domain_count: dict[str, int]
    notes: list[str]
    budget_tally: dict[str, int]
    budget_flags: list[str]
    seconds_used: float
    work_limits: list[str] | None
    guidance: dict[str, Any] | None


def _no_ask(payload: dict[str, Any]) -> Any:
    raise RuntimeError("this driver cannot pause a session to ask a question")


@dataclass
class EngineContext:
    """Per-invocation objects that must not be checkpointed."""

    engine: ResearchEngine
    should_cancel: Callable[[], bool]
    ask_before_follow_up: bool = False
    # Pause the session with a question and return the reply when it is resumed. Supplied by the driver.
    ask: Callable[[dict[str, Any]], Any] = _no_ask


Step = Callable[[StepSession, SessionState, EngineContext], dict[str, Any] | None]


def _page_ref(source_id: str, recheck: bool = False) -> dict[str, Any]:
    return {"source_id": source_id, "recheck": recheck}


# ------------------------------------------------------------------- steps


def recall(session: StepSession, state: SessionState, context: EngineContext) -> dict[str, Any]:
    engine = context.engine
    session._event("session_started", {"budget": session.budget.budget.model_dump(), "model": engine.llm.model,
                                       "search_provider": engine.search.name})
    session._prepare()
    assert session.memory is not None
    view = session.memory_view()
    view.pop("tried_queries", None)  # only its length is used, and only by the line just printed
    return {
        "recalled": True, "phase": "recheck", "leads_read": 0, "picks": [], "followup_round": 0,
        "recheck_ids": [row["id"] for row in session.memory.recheck[: session.settings.max_rechecks_per_session]],
        "memory_view": view, "focus_questions": session._focus_questions(),
    }


def dispatch(session: StepSession, state: SessionState, context: EngineContext) -> dict[str, Any]:
    """Choose the next unit of work. The checks and their order are those of the linear engine's loops."""
    assert session.memory is not None
    budget = session.budget
    phase = state["phase"]
    leads_read = state.get("leads_read", 0)

    if phase == "recheck":
        pending = list(state.get("recheck_ids") or [])
        if pending and budget.can_fetch():
            source_id = pending.pop(0)
            source = session.store.get_source(source_id)
            session.c["rechecked_sources"] += 1
            session._say(f"recheck: {source['canonical_url'] if source else source_id}")
            return {"recheck_ids": pending, "page": _page_ref(source_id, recheck=True), "route": "fetch"}
        phase, leads_read = "early_leads", 0

    if phase == "early_leads":
        if leads_read < session._early_lead_limit() and session.memory.backlog and budget.can_fetch():
            lead = session._next_lead()
            if lead is not None:
                return {"phase": phase, "leads_read": leads_read + 1, "page": _page_ref(lead["id"]), "route": "fetch"}
        phase = "plan"

    if phase == "plan":
        if budget.budget.max_search_requests > 0:
            return {"phase": "search", "route": "plan"}
        phase, leads_read = "late_leads", 0

    if phase in ("search", "follow_up_search"):
        picks = list(state.get("picks") or [])
        if picks and budget.can_fetch():
            return {"phase": phase, "picks": picks[1:], "page": _page_ref(picks[0]), "route": "fetch"}
        plan, index = state.get("plan") or [], state.get("query_index", 0)
        pages_left = budget.budget.max_fetched_pages - budget.fetched_pages
        if index < len(plan) and pages_left > 0 and budget.can_search():
            return {"phase": phase, "picks": [], "quota": session._page_quota(index, len(plan)), "route": "search"}
        phase, leads_read = ("late_leads", 0) if phase == "search" else ("wrapup", leads_read)

    if phase == "late_leads":
        if leads_read < budget.budget.max_fetched_pages and session.memory.backlog and budget.can_fetch():
            lead = session._next_lead()
            if lead is not None:
                return {"phase": phase, "leads_read": leads_read + 1, "page": _page_ref(lead["id"]), "route": "fetch"}

    session.work_limits = budget.limits_reached()  # measured before the review spends its reserve
    return {"phase": "wrapup", "picks": [], "route": "reflect"}


def plan(session: StepSession, state: SessionState, context: EngineContext) -> dict[str, Any]:
    queries = session._plan()
    return {"plan": [q.model_dump() for q in queries], "query_index": 0, "memory_view": None}


def search(session: StepSession, state: SessionState, context: EngineContext) -> dict[str, Any]:
    index = state["query_index"]
    candidates = session._search_candidates(PlannedQuery(**state["plan"][index]))
    refs = [{"source_id": c["id"], "rank": c["rank"], "title": c["title"], "snippet": c["snippet"]} for c in candidates]
    return {"query_index": index + 1, "candidates": refs or None, "route": "triage" if refs else "dispatch"}


def triage(session: StepSession, state: SessionState, context: EngineContext) -> dict[str, Any]:
    planned = PlannedQuery(**state["plan"][state["query_index"] - 1])
    candidates = []
    for ref in state.get("candidates") or []:
        source = session.store.get_source(ref["source_id"])
        if source is not None:
            candidates.append(source | {"rank": ref["rank"], "title": ref["title"], "snippet": ref["snippet"]})
    picks = session._triage(planned, candidates, state["quota"])
    return {"picks": [p["id"] for p in picks], "candidates": None}


def fetch(session: StepSession, state: SessionState, context: EngineContext) -> dict[str, Any]:
    ref = state["page"] or {}
    source = session.store.get_source(ref["source_id"])
    page = session._fetch_source(source, recheck=bool(ref.get("recheck"))) if source else None
    if page is None:
        return {"page": None, "route": "dispatch"}
    return {"route": "extract", "page": ref | {"snapshot_id": page.snapshot_id, "title": page.title,
                                              "published_at": page.published_at, "patterns": page.patterns}}


def extract(session: StepSession, state: SessionState, context: EngineContext) -> dict[str, Any]:
    extraction = session._extract(session.page_from_ref(state["page"]))
    if extraction is None:
        return {"page": None, "extraction": None, "route": "dispatch"}
    return {"extraction": extraction.model_dump(), "route": "integrate"}


def integrate(session: StepSession, state: SessionState, context: EngineContext) -> dict[str, Any]:
    session._verify_and_integrate(session.page_from_ref(state["page"]), Extraction.model_validate(state["extraction"]))
    return {"page": None, "extraction": None}


def reflect(session: StepSession, state: SessionState, context: EngineContext) -> dict[str, Any]:
    session._reflect()
    more = state.get("followup_round", 0) < session.settings.followup_rounds and session._wants_follow_up()
    return {"route": "follow_up" if more else "report"}


def follow_up(session: StepSession, state: SessionState, context: EngineContext) -> dict[str, Any]:
    """Conditional branch: one more targeted search round on the directions the review just wrote."""
    if context.ask_before_follow_up:
        # Nothing above this line has side effects: the node runs again from the top when resumed.
        leads = [lead["text"] for lead in session.store.list_leads(session.watch_id, "OPEN")]
        answer = context.ask({
            "question": "The review proposed further directions. Spend remaining budget on a follow-up round?",
            "directions": leads[:5],
            "searches_left": session.budget.budget.max_search_requests - session.budget.search_requests,
            "pages_left": session.budget.budget.max_fetched_pages - session.budget.fetched_pages,
        })
        if not (answer is True or (isinstance(answer, dict) and answer.get("approve"))):
            session.notes.append("A follow-up round was proposed and declined by the user.")
            return {"route": "report"}
    queries = session._plan_follow_up()
    return {
        "plan": [q.model_dump() for q in queries], "query_index": 0, "picks": [], "phase": "follow_up_search",
        "followup_round": state.get("followup_round", 0) + 1, "lead_ids": state.get("lead_ids") or [],
        # The follow-up round reads pages with the questions as they stand after the review, as the linear engine does.
        "focus_questions": _Session._focus_questions(session),
        "route": "dispatch",
    }


def report(session: StepSession, state: SessionState, context: EngineContext) -> dict[str, Any]:
    limits = session.work_limits if session.work_limits is not None else session.budget.limits_reached()
    session.work_limits = limits
    session._finalize(RunStatus.COMPLETED, "budget:" + ",".join(limits) if limits else "completed", None)
    return {}


STEPS: dict[str, Step] = {
    "recall": recall, "dispatch": dispatch, "plan": plan, "search": search, "triage": triage, "fetch": fetch,
    "extract": extract, "integrate": integrate, "reflect": reflect, "follow_up": follow_up, "report": report,
}

# The session's control flow, defined once and used by every driver.
ENTRY = "recall"
ALWAYS_NEXT: dict[str, str | None] = {  # None ends the session
    "recall": "dispatch", "plan": "dispatch", "triage": "dispatch", "integrate": "dispatch", "report": None,
}
BRANCHES: dict[str, tuple[str, ...]] = {  # the node writes its choice to state["route"]
    "dispatch": ("fetch", "plan", "search", "reflect"),
    "search": ("triage", "dispatch"),
    "fetch": ("extract", "dispatch"),
    "extract": ("integrate", "dispatch"),
    "reflect": ("follow_up", "report"),
    "follow_up": ("dispatch", "report"),
}


def next_step(name: str, state: SessionState) -> str | None:
    if name in BRANCHES:
        choice = state["route"]
        if choice not in BRANCHES[name]:
            raise ValueError(f"step '{name}' chose '{choice}', which is not one of {BRANCHES[name]}")
        return choice
    return ALWAYS_NEXT[name]


def run_step(name: str, state: SessionState, context: EngineContext) -> dict[str, Any]:
    """Rebuild the session from state, run one step, and return what changed (tallies included)."""
    session = StepSession(context.engine, state, context.should_cancel)
    update = STEPS[name](session, state, context) or {}
    return session.export() | update
