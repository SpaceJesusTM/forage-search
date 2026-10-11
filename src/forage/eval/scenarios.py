"""Built-in SIMULATED scenarios for the policy-comparison harness.

Every organisation, event, figure and URL below is invented. A scenario is a small
web that changes night by night, a set of watches, and a scripted user. It exists to
exercise the checkpoint mechanism reproducibly; it is designed, not sampled, so results
on it illustrate behaviour and are not evidence about real research.
"""

from __future__ import annotations

from typing import Any

NOTICE = "SIMULATED scenario. Every organisation, event, figure and page is invented."


def _doc(url: str, title: str, snippet: str, body: str, keywords: list[str], epoch: int, watch: str,
         counts: str = "always") -> dict[str, Any]:
    return {"url": url, "title": title, "snippet": snippet, "body": f"{title}.\n\n{body}", "keywords": keywords,
            "from_epoch": epoch, "labels": {"watch": watch, "counts": counts}}


def three_watches() -> dict[str, Any]:
    """Three watches with different needs:

    - internships: the approved scope excludes remote roles, but most new postings are remote.
      Continuing productively needs a scope decision that only the user can make.
    - events: new relevant pages arrive every night. No review is needed; asking is a cost.
    - pilot: results dry up after the first night, then one late page appears. Stopping saves
      compute and risks missing the late page.
    """
    docs: list[dict[str, Any]] = []
    intern_kw = ["machine", "learning", "internship", "toronto", "summer", "2027", "posting", "company", "new", "announced"]
    onsite = [
        (0, "northwind-labs", "Northwind Labs", "recommendation models", 34, "has opened a paid machine learning internship posting in Toronto for summer 2027"),
        (0, "harbourfront-ai", "Harbourfront AI", "speech recognition", 31, "is recruiting a paid machine learning intern for its Toronto office for summer 2027"),
        (3, "cedarline-robotics", "Cedarline Robotics", "robot perception", 36, "announced a paid machine learning internship posting in Toronto for summer 2027"),
    ]
    for epoch, slug, company, focus, pay, lead in onsite:
        docs.append(_doc(
            f"https://{slug}.example.com/careers/ml-intern", f"{company} machine learning internship Toronto summer 2027",
            f"Paid machine learning internship posting at {company}, a Toronto company, for summer 2027.",
            f"{company} {lead}. The {company} internship posting covers {focus} and pays {pay} dollars per hour.",
            intern_kw, epoch, "internships"))
    remote = [
        (0, "lakeshore-data", "Lakeshore Data", "fraud detection", 30), (0, "juniper-health-ai", "Juniper Health AI", "medical imaging", 33),
        (1, "portland-street", "Portland Street Analytics", "demand forecasting", 29), (1, "bluewater-semantics", "Bluewater Semantics", "search ranking", 35),
        (2, "trillium-compute", "Trillium Compute", "model compression", 32), (2, "maple-vector", "Maple Vector", "retrieval systems", 37),
    ]
    for epoch, slug, company, focus, pay in remote:
        docs.append(_doc(
            f"https://{slug}.example.com/jobs/remote-ml-intern", f"Remote machine learning internship at {company} Toronto summer 2027",
            f"Remote paid machine learning internship posting from Toronto company {company} for summer 2027.",
            f"{company} lists a remote paid machine learning internship posting for summer 2027 managed from Toronto. "
            f"The {company} remote internship posting covers {focus} and pays {pay} dollars per hour.",
            intern_kw + ["remote"], epoch, "internships", counts="if_scope_amended"))

    event_kw = ["artificial", "intelligence", "meetup", "event", "toronto", "scheduled", "new", "announced"]
    events = [
        (0, "queen-west-ml", "Queen West ML Circle", "14 October", "the Annex Library", "graph neural networks"),
        (0, "distillery-ai", "Distillery AI Society", "21 October", "Brickworks Hall", "evaluation of language models"),
        (1, "roncesvalles-data", "Roncesvalles Data Club", "28 October", "Wabash Centre", "causal inference"),
        (1, "junction-robotics", "Junction Robotics Forum", "4 November", "Keele Street Studio", "robot learning"),
        (2, "danforth-ai", "Danforth AI Collective", "11 November", "Pape Commons", "model interpretability"),
        (2, "leslieville-nlp", "Leslieville NLP Night", "18 November", "Carlaw Loft", "retrieval augmented generation"),
        (3, "parkdale-vision", "Parkdale Vision Group", "25 November", "Sorauren Hub", "video understanding"),
        (3, "corktown-agents", "Corktown Agents Guild", "2 December", "Trinity Street Hall", "agent safety testing"),
        (4, "bloorcourt-ml", "Bloorcourt ML Exchange", "9 December", "Ossington Rooms", "federated learning"),
        (4, "riverside-ai", "Riverside AI Assembly", "16 December", "Broadview Depot", "speech synthesis"),
    ]
    for epoch, slug, group, date, venue, topic in events:
        docs.append(_doc(
            f"https://{slug}.example.org/events/next", f"{group} artificial intelligence meetup Toronto {date}",
            f"Artificial intelligence meetup event in Toronto scheduled by {group} for {date}.",
            f"{group} scheduled an artificial intelligence meetup event in Toronto for {date} 2026 at {venue}. "
            f"The {group} meetup event features a talk about {topic} and registration is free.",
            event_kw, epoch, "events"))

    pilot_kw = ["greenbelt", "battery", "storage", "pilot", "result", "reported", "new", "announced"]
    pilot = [
        (0, "efficiency", "round-trip efficiency", "The Greenbelt battery storage pilot reported a round-trip efficiency of 87 percent in its first results. "
                                                  "The pilot results cover 40 megawatt hours of storage installed across three sites."),
        (0, "outages", "outage response", "The Greenbelt battery storage pilot reported that its batteries covered 11 local outages during the first winter. "
                                          "The pilot results show a median response time of 2 seconds for each outage."),
        (0, "costs", "operating costs", "The Greenbelt battery storage pilot reported operating costs of 19 dollars per megawatt hour delivered. "
                                        "The pilot results attribute most costs to cooling equipment in the storage containers."),
        (4, "final", "final capacity retention", "The Greenbelt battery storage pilot reported final results showing capacity retention of 94 percent after two years. "
                                                 "The final pilot results recommend expanding the storage programme to six additional sites."),
    ]
    for epoch, slug, aspect, body in pilot:
        docs.append(_doc(
            f"https://greenbelt-energy.example.net/pilot/{slug}", f"Greenbelt battery storage pilot results: {aspect}",
            f"Results reported by the Greenbelt battery storage pilot on {aspect}.", body, pilot_kw, epoch, "pilot"))

    return {
        "name": "three-watches",
        "_notice": NOTICE,
        "nights": 5,
        "sessions_per_night": 6,
        "session_minutes": 30,
        "epoch_by_night": [0, 1, 2, 3, 4],
        "budget": {"max_search_requests": 3, "max_fetched_pages": 4, "max_llm_output_tokens": 6000, "max_duration_seconds": 600},
        "policy_config": {"periodic": {"every": 3}, "adaptive": {}},
        "web": {"docs": docs},
        "watches": [
            {"key": "internships", "objective": "Find paid machine learning internship postings at Toronto companies for summer 2027.",
             "spec": {"key_questions": ["Which Toronto companies have machine learning internship postings for summer 2027?"],
                      "exclude": ["remote"]}},
            {"key": "events", "objective": "Track upcoming artificial intelligence meetup events in Toronto.",
             "spec": {"key_questions": ["What artificial intelligence meetup events are scheduled in Toronto?"]}},
            {"key": "pilot", "objective": "What results has the Greenbelt battery storage pilot reported?",
             "spec": {"key_questions": ["What results has the Greenbelt battery storage pilot reported?"]}},
        ],
        "user": {
            "delay_nights": 1,
            "description": ("Answers one night after being asked. Removes the 'remote' exclusion from the internships watch once "
                            "set-aside results show it is costing coverage; pauses a watch that has been dry for two sessions; "
                            "otherwise says continue. The same rules apply under every policy; the user is only asked when "
                            "a policy raises a review."),
            "rules": {
                "internships": [{"when": "scope_pressure", "choice": "amend", "patch": {"exclude_remove": ["remote"]}}],
                "pilot": [{"when": "dry_streak>=2", "choice": "pause"}],
                "*": [{"when": "always", "choice": "continue"}],
            },
        },
    }


BUILTIN = {"three-watches": three_watches}


def load_scenario(ref: str) -> dict[str, Any]:
    """`builtin:<name>`, a bare built-in name, or a path to a JSON scenario file."""
    import json
    from pathlib import Path

    name = ref.removeprefix("builtin:")
    if name in BUILTIN:
        return BUILTIN[name]()
    path = Path(ref)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"unknown scenario '{ref}' (built-in: {', '.join(BUILTIN)})")
