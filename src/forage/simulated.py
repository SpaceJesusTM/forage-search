"""SIMULATED web and SIMULATED model for offline tests and replayable experiments.

Nothing in this module touches the network or a real model. Runs that use it are
recorded with provider `fixture` and model `simulated`, so their reports can never
be mistaken for live research.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from forage.fetch.base import FetchResult
from forage.fetch.extract import extract_html, extract_plain
from forage.fetch.urls import canonicalize_url
from forage.llm.base import LLMUsage, T
from forage.search.base import SearchError, SearchRateLimited, SearchResult
from forage.util import normalize_text, token_set, tokenize


class SimulatedWeb:
    """A small corpus of documents that can appear, change and disappear over numbered epochs.

    Document fields: url, title, snippet, body (text or HTML), keywords, from_epoch,
    until_epoch, status, and `versions` ([{from_epoch, body, title?, snippet?}]).
    """

    def __init__(self, docs: list[dict[str, Any]], epoch: int = 0):
        self.docs = docs
        self.epoch = epoch
        self.search_log: list[str] = []
        self.fetch_log: list[str] = []
        self.rate_limit_next = 0
        self.fail_queries: set[str] = set()

    @classmethod
    def from_file(cls, path: str | Path, epoch: int = 0) -> "SimulatedWeb":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["docs"] if isinstance(data, dict) else data, epoch)

    def _visible(self, doc: dict[str, Any]) -> bool:
        if self.epoch < int(doc.get("from_epoch", 0)):
            return False
        until = doc.get("until_epoch")
        return until is None or self.epoch <= int(until)

    def current(self, doc: dict[str, Any]) -> dict[str, Any]:
        view = dict(doc)
        for version in sorted(doc.get("versions", []), key=lambda v: int(v.get("from_epoch", 0))):
            if self.epoch >= int(version.get("from_epoch", 0)):
                view.update({k: v for k, v in version.items() if k != "from_epoch"})
        return view

    def search(self, query: str, max_results: int = 8) -> list[SearchResult]:
        self.search_log.append(query)
        if self.rate_limit_next > 0:
            self.rate_limit_next -= 1
            raise SearchRateLimited("simulated throttling", retry_after=0.0)
        if normalize_text(query) in self.fail_queries:
            raise SearchError("simulated provider failure")
        wanted = token_set(query)
        scored = []
        for doc in self.docs:
            if not self._visible(doc) or doc.get("unlisted"):
                continue
            view = self.current(doc)
            haystack = token_set(" ".join([view.get("title", ""), view.get("snippet", ""), " ".join(view.get("keywords", []))]))
            overlap = len(wanted & haystack)
            if overlap:
                scored.append((overlap / (len(wanted) or 1), view))
        scored.sort(key=lambda pair: (-pair[0], pair[1]["url"]))
        return [
            SearchResult(rank=i + 1, url=view["url"], title=view.get("title", ""), snippet=view.get("snippet", ""))
            for i, (_, view) in enumerate(scored[:max_results])
        ]

    def fetch(self, url: str) -> FetchResult:
        self.fetch_log.append(url)
        try:
            wanted = canonicalize_url(url)
        except ValueError:
            return FetchResult(url=url, ok=False, outcome="BLOCKED_URL", error="malformed url")
        for doc in self.docs:
            if canonicalize_url(doc["url"]) != wanted:
                continue
            if not self._visible(doc):
                return FetchResult(url=url, ok=False, outcome="HTTP_ERROR", http_status=404, error="HTTP 404")
            view = self.current(doc)
            status = int(view.get("status", 200))
            if status >= 400:
                return FetchResult(url=url, ok=False, outcome="HTTP_ERROR", http_status=status, error=f"HTTP {status}")
            body = view.get("body", "")
            if "<" in body and ">" in body and re.search(r"<(html|body|p|div|h1)\b", body, re.I):
                extracted = extract_html(body, view["url"])
                content_type = "text/html"
            else:
                extracted = extract_plain(body)
                content_type = "text/plain"
            return FetchResult(
                url=url, ok=bool(extracted.text), outcome="OK" if extracted.text else "EMPTY_TEXT", final_url=view["url"],
                http_status=200, content_type=content_type, title=extracted.title or view.get("title"),
                published_at=view.get("published_at"), text=extracted.text, flags=list(extracted.flags),
                bytes_read=len(body),
            )
        return FetchResult(url=url, ok=False, outcome="HTTP_ERROR", http_status=404, error="HTTP 404")


class FixtureSearch:
    name = "fixture"

    def __init__(self, web: SimulatedWeb):
        self.web = web

    def search(self, query: str, max_results: int = 8) -> list[SearchResult]:
        return self.web.search(query, max_results)


class FixtureFetcher:
    def __init__(self, web: SimulatedWeb):
        self.web = web

    def fetch(self, url: str) -> FetchResult:
        return self.web.fetch(url)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 25]


class SimulatedLLM:
    """Rule-based stand-in for a language model. Deterministic, and it never invents text:
    extraction copies whole sentences from the page, so its quotes always verify.

    `overrides` maps a purpose to a callable(payload) -> dict (or a dict) to script
    specific behaviour such as a fabricated quote or a guidance request.
    """

    model = "simulated"

    def __init__(self, overrides: dict[str, Any] | None = None, tokens_per_call: int = 40):
        self.overrides = overrides or {}
        self.tokens_per_call = tokens_per_call
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def generate(self, *, purpose: str, system: str, user: str, schema: type[T], max_tokens: int,
                 payload: dict[str, Any] | None = None) -> tuple[T, LLMUsage]:
        payload = payload or {}
        self.calls.append((purpose, {"system": system, "user": user, "payload": payload}))
        override = self.overrides.get(purpose)
        if override is not None:
            data = override(payload) if callable(override) else override
            if data is not None:
                return schema.model_validate(data), self._usage(user)
        handler: Callable[[dict[str, Any]], dict[str, Any]] = getattr(self, f"_{purpose}", self._empty)
        return schema.model_validate(handler(payload)), self._usage(user)

    def _usage(self, user: str) -> LLMUsage:
        return LLMUsage(prompt_tokens=len(user) // 4, output_tokens=self.tokens_per_call, duration_ms=1)

    @staticmethod
    def _empty(payload: dict[str, Any]) -> dict[str, Any]:
        return {}

    @staticmethod
    def _structure_watch(payload: dict[str, Any]) -> dict[str, Any]:
        objective = payload.get("objective", "")
        return {"title": " ".join(objective.split()[:8]), "key_questions": [objective], "include": [], "exclude": [],
                "priorities": []}

    @staticmethod
    def _plan(payload: dict[str, Any]) -> dict[str, Any]:
        tried = {normalize_text(q) for q in payload.get("tried_queries", [])}
        seeds = [(q["text"], q.get("id")) for q in payload.get("open_questions", [])]
        seeds += [(lead, None) for lead in payload.get("leads", [])]
        seeds.append((payload.get("objective", ""), None))
        session = int(payload.get("session_number", 1))
        if session > 1:
            seeds += [(f"{q} new announced session{session}", None) for q in payload.get("standing_questions", [])]
        queries = []
        for text, qid in seeds:
            query = " ".join(tokenize(text)[:9])
            if query and normalize_text(query) not in tried and all(query != q["query"] for q in queries):
                queries.append({"query": query, "rationale": f"targets: {text[:80]}", "question_id": qid})
        return {"queries": queries[: int(payload.get("max_queries", 4))]}

    @staticmethod
    def _triage(payload: dict[str, Any]) -> dict[str, Any]:
        focus = token_set(payload.get("objective", "") + " " + " ".join(payload.get("focus", [])))
        excluded = [token_set(x) for x in payload.get("exclude", [])]
        ratings = []
        for cand in payload.get("candidates", []):
            words = token_set(cand.get("title", "") + " " + cand.get("snippet", ""))
            overlap = len(words & focus)
            score = 3 if overlap >= 4 else 2 if overlap >= 2 else 1 if overlap == 1 else 0
            out_of_scope = any(ex and ex <= words for ex in excluded)
            if out_of_scope:
                score = min(score, 1)
            ratings.append({"index": cand["index"], "score": score, "out_of_scope": out_of_scope,
                            "reason": "simulated lexical overlap"})
        return {"ratings": ratings}

    @staticmethod
    def _extract(payload: dict[str, Any]) -> dict[str, Any]:
        focus = token_set(payload.get("objective", "") + " " + " ".join(payload.get("focus", [])))
        scored = []
        for sentence in _sentences(payload.get("text", "")):
            overlap = len(token_set(sentence) & focus)
            if overlap >= 2 and 40 <= len(sentence) <= 400:
                scored.append((overlap, sentence))
        scored.sort(key=lambda pair: -pair[0])
        claims = [{"text": s, "quote": s} for _, s in scored[:4]]
        if not claims:
            return {"relevance": 0, "items": []}
        title = payload.get("title") or claims[0]["text"][:80]
        return {"relevance": 3 if len(claims) >= 2 else 2,
                "items": [{"title": title, "subject": title, "claims": claims, "uncertainty": None}]}

    @staticmethod
    def _support(payload: dict[str, Any]) -> dict[str, Any]:
        verdicts = []
        for pair in payload.get("pairs", []):
            claim, quote = token_set(pair["claim"]), token_set(pair["quote"])
            ratio = len(claim & quote) / (len(claim) or 1)
            verdicts.append({"index": pair["index"], "bearing": "direct",
                             "verdict": "supported" if ratio >= 0.6 else "partial" if ratio >= 0.3 else "unsupported"})
        return {"verdicts": verdicts}

    @staticmethod
    def _match(payload: dict[str, Any]) -> dict[str, Any]:
        matches = []
        for item in payload.get("items", []):
            new = token_set(item.get("text", ""))
            best, best_score = None, 0.0
            for cand in payload.get("candidates", []):
                other = token_set(cand.get("text", ""))
                score = len(new & other) / (len(new | other) or 1)
                if score > best_score:
                    best, best_score = cand["index"], score
            matches.append({"item": item["index"], "same_as": best if best_score >= 0.4 else None})
        return {"matches": matches}

    @staticmethod
    def _novelty(payload: dict[str, Any]) -> dict[str, Any]:
        known = [token_set(c) for c in payload.get("known_claims", [])]
        verdicts = []
        for item in payload.get("new_claims", []):
            words = token_set(item["text"])
            best = max((len(words & k) / (len(words | k) or 1) for k in known), default=0.0)
            verdicts.append({"index": item["index"], "adds_information": best < 0.6})
        return {"verdicts": verdicts}

    @staticmethod
    def _reflect(payload: dict[str, Any]) -> dict[str, Any]:
        findings = payload.get("findings", [])
        resolved = []
        for question in payload.get("open_questions", []):
            q_words = token_set(question["text"])
            for finding in findings:
                f_words = token_set(finding.get("text", ""))
                if q_words and len(q_words & f_words) / len(q_words) >= 0.6:
                    resolved.append({"question_id": question["id"], "finding_ids": [finding["id"]],
                                     "answer": f"Addressed by: {finding.get('title', '')}"[:200]})
                    break
        return {"summary": f"Simulated session reviewed {len(findings)} findings.", "resolved": resolved,
                "new_questions": [], "next_directions": [], "guidance": None}
