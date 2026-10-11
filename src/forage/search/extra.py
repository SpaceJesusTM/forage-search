"""Optional search adapters: the `ddgs` metasearch library and a self-hosted SearXNG instance.

Neither is required. `ddgs` needs `pip install forage-search[ddgs]`; SearXNG needs a
URL the user runs themselves with the JSON format enabled.
"""

from __future__ import annotations

import httpx

from forage.search.base import SearchError, SearchRateLimited, SearchResult


class DdgsSearch:
    name = "ddgs"

    def __init__(self, *, backend: str = "auto", timeout: int = 30):
        try:
            from ddgs import DDGS  # noqa: F401
        except ImportError as exc:
            raise SearchError("the 'ddgs' package is not installed (pip install 'forage-search[ddgs]')") from exc
        self.backend = backend
        self.timeout = timeout

    def search(self, query: str, max_results: int = 8) -> list[SearchResult]:
        from ddgs import DDGS
        from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

        try:
            rows = DDGS(timeout=self.timeout).text(query, max_results=max_results, backend=self.backend)
        except RatelimitException as exc:
            raise SearchRateLimited(f"ddgs rate limited: {exc}") from exc
        except TimeoutException as exc:
            raise SearchError(f"ddgs timed out: {str(exc)[:160]}") from exc
        except DDGSException as exc:
            if "no results" in str(exc).lower():
                return []
            raise SearchError(f"ddgs failed: {str(exc)[:160]}") from exc
        out = []
        for row in rows or []:
            url = row.get("href") or ""
            if url.startswith(("http://", "https://")):
                out.append(SearchResult(rank=len(out) + 1, url=url, title=(row.get("title") or "")[:300],
                                        snippet=" ".join((row.get("body") or "").split())[:500]))
        return out[:max_results]


class SearxngSearch:
    name = "searxng"

    def __init__(self, base_url: str, *, timeout: float = 15.0, client: httpx.Client | None = None):
        if not base_url:
            raise SearchError("searxng_url is not configured")
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout, trust_env=False)

    def search(self, query: str, max_results: int = 8) -> list[SearchResult]:
        try:
            resp = self._client.get(f"{self.base_url}/search", params={"q": query, "format": "json"})
        except httpx.HTTPError as exc:
            raise SearchError(f"searxng request failed: {type(exc).__name__}: {exc}") from exc
        if resp.status_code == 429:
            raise SearchRateLimited("searxng rate limited the request")
        if resp.status_code >= 400:
            raise SearchError(f"searxng returned HTTP {resp.status_code} (is the JSON format enabled?)")
        try:
            rows = resp.json().get("results", [])
        except ValueError as exc:
            raise SearchError("searxng did not return JSON") from exc
        out = []
        for row in rows:
            url = row.get("url") or ""
            if url.startswith(("http://", "https://")):
                out.append(SearchResult(rank=len(out) + 1, url=url, title=(row.get("title") or "")[:300],
                                        snippet=" ".join((row.get("content") or "").split())[:500]))
        return out[:max_results]
