"""Keyless search via DuckDuckGo's HTML endpoint.

This scrapes a public page, so it is the least reliable part of Forage: the
endpoint throttles bursts (HTTP 202 challenge) and its markup can change. Queries
are paced, throttling is reported as `SearchRateLimited`, and nothing is retried
in a loop. Swap in a keyed API or SearXNG adapter for unattended production use.
"""

from __future__ import annotations

import time
from typing import Callable
from urllib.parse import parse_qs, urlsplit

import httpx

from forage.fetch.http import guarded_client
from forage.search.base import SearchError, SearchRateLimited, SearchResult

ENDPOINT = "https://html.duckduckgo.com/html/"
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36"
)


def parse_results(html: str, max_results: int = 8) -> list[SearchResult]:
    from lxml import etree
    from lxml import html as lxml_html

    try:
        doc = lxml_html.fromstring(html)
    except (etree.ParserError, ValueError):
        return []
    out: list[SearchResult] = []
    seen: set[str] = set()
    for block in doc.xpath('//div[contains(concat(" ", normalize-space(@class), " "), " result ")]'):
        if "result--ad" in (block.get("class") or ""):
            continue
        links = block.xpath('.//a[contains(@class, "result__a")]')
        if not links:
            continue
        url = _unwrap(links[0].get("href") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        snippets = block.xpath('.//*[contains(@class, "result__snippet")]')
        out.append(
            SearchResult(
                rank=len(out) + 1,
                url=url,
                title=" ".join(links[0].text_content().split())[:300],
                snippet=" ".join(snippets[0].text_content().split())[:500] if snippets else "",
            )
        )
        if len(out) >= max_results:
            break
    return out


def _unwrap(href: str) -> str:
    """DuckDuckGo sometimes wraps targets as //duckduckgo.com/l/?uddg=<url>."""
    if href.startswith("//"):
        href = "https:" + href
    parts = urlsplit(href)
    if parts.netloc.endswith("duckduckgo.com"):
        if parts.path.startswith("/l/"):
            target = parse_qs(parts.query).get("uddg", [""])[0]
            return target if target.startswith(("http://", "https://")) else ""
        return ""  # ads and internal links
    return href if parts.scheme in ("http", "https") else ""


class DuckDuckGoSearch:
    name = "duckduckgo"

    def __init__(
        self, *, min_interval: float = 12.0, timeout: float = 15.0, region: str = "wt-wt",
        client: httpx.Client | None = None, sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.min_interval = min_interval
        self.region = region
        self._client = client or guarded_client(user_agent=_BROWSER_UA, timeout=timeout)
        self._sleep = sleep
        self._monotonic = monotonic
        self._last: float | None = None
        self._cooldown_until: float = 0.0

    def search(self, query: str, max_results: int = 8) -> list[SearchResult]:
        now = self._monotonic()
        if now < self._cooldown_until:
            raise SearchRateLimited("duckduckgo is cooling down after throttling", retry_after=self._cooldown_until - now)
        if self._last is not None:
            wait = self.min_interval - (now - self._last)
            if wait > 0:
                self._sleep(wait)
        self._last = self._monotonic()
        try:
            resp = self._client.post(ENDPOINT, data={"q": query, "b": "", "kl": self.region},
                                     headers={"Referer": "https://html.duckduckgo.com/"})
        except httpx.HTTPError as exc:
            raise SearchError(f"duckduckgo request failed: {type(exc).__name__}: {exc}") from exc
        if resp.status_code in (202, 403, 429) or "anomaly-modal" in resp.text[:20000]:
            self._cooldown_until = self._monotonic() + 90.0
            raise SearchRateLimited(f"duckduckgo throttled the request (HTTP {resp.status_code})", retry_after=90.0)
        if resp.status_code >= 400:
            raise SearchError(f"duckduckgo returned HTTP {resp.status_code}")
        return parse_results(resp.text, max_results)
