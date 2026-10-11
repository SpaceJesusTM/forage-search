"""HTTP fetcher with a connect-time address guard, robots.txt and size/time limits.

The guard runs where the TCP connection is opened, so it applies to every hop
of a redirect chain and to the address actually dialled, not a prior lookup.
"""

from __future__ import annotations

import socket
import time
from typing import Callable
from urllib import robotparser
from urllib.parse import urlsplit

import httpcore
import httpx

from forage.fetch.base import FetchResult
from forage.fetch.extract import extract_html, extract_pdf, extract_plain
from forage.fetch.urls import check_fetchable, is_public_ip


class BlockedAddressError(Exception):
    """The target resolves to an address Forage refuses to contact."""


def resolve_guarded(host: str, allow_private: bool = False) -> str:
    """Resolve `host` (IPv4 first) and return one address, refusing non-public targets."""
    addresses: list[str] = []
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            infos = socket.getaddrinfo(host, None, family, socket.SOCK_STREAM)
        except socket.gaierror:
            continue
        addresses = list(dict.fromkeys(info[4][0] for info in infos))
        if addresses:
            break
    if not addresses:
        raise httpcore.ConnectError(f"could not resolve host '{host}'")
    if not allow_private:
        for address in addresses:
            if not is_public_ip(address):
                raise BlockedAddressError(f"'{host}' resolves to non-public address {address}")
    return addresses[0]


class _GuardedBackend(httpcore.SyncBackend):
    def __init__(self, allow_private: bool):
        self._allow_private = allow_private

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        address = resolve_guarded(host, self._allow_private)
        return super().connect_tcp(address, port, timeout=timeout, local_address=local_address, socket_options=socket_options)


class GuardedTransport(httpx.HTTPTransport):
    """httpx transport whose connections all pass through the address guard. No proxy support by design."""

    def __init__(self, allow_private: bool = False, **kwargs):
        super().__init__(trust_env=False, **kwargs)
        pool = getattr(self, "_pool", None)
        if pool is None or not hasattr(pool, "_network_backend"):
            raise RuntimeError("httpx/httpcore internals changed; refusing to fetch without the address guard")
        pool._network_backend = _GuardedBackend(allow_private)


def guarded_client(*, user_agent: str, timeout: float, allow_private: bool = False, max_redirects: int = 5) -> httpx.Client:
    return httpx.Client(
        transport=GuardedTransport(allow_private=allow_private, retries=0),
        timeout=httpx.Timeout(timeout, connect=min(10.0, timeout)),
        follow_redirects=True,
        max_redirects=max_redirects,
        trust_env=False,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,text/plain;q=0.8,*/*;q=0.5",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )


class HttpFetcher:
    def __init__(
        self, *, user_agent: str, timeout: float = 20.0, max_bytes: int = 4_000_000, respect_robots: bool = True,
        min_interval_per_host: float = 2.0, allow_private: bool = False, client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep, monotonic: Callable[[], float] = time.monotonic,
    ):
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.respect_robots = respect_robots
        self.min_interval_per_host = min_interval_per_host
        self._client = client or guarded_client(user_agent=user_agent, timeout=timeout, allow_private=allow_private)
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_request: dict[str, float] = {}
        self._robots: dict[str, robotparser.RobotFileParser | None] = {}

    # -- politeness -------------------------------------------------------

    def _pace(self, host: str) -> None:
        last = self._last_request.get(host)
        if last is not None:
            wait = self.min_interval_per_host - (self._monotonic() - last)
            if wait > 0:
                self._sleep(wait)
        self._last_request[host] = self._monotonic()

    def _robots_for(self, scheme: str, netloc: str) -> robotparser.RobotFileParser | None:
        key = f"{scheme}://{netloc}"
        if key in self._robots:
            return self._robots[key]
        parser: robotparser.RobotFileParser | None = robotparser.RobotFileParser()
        try:
            resp = self._client.get(f"{key}/robots.txt", timeout=min(8.0, self.timeout))
            if resp.status_code in (401, 403):
                parser.parse(["User-agent: *", "Disallow: /"])  # type: ignore[union-attr]
            elif resp.status_code >= 400:
                parser = None  # no robots file: everything allowed
            else:
                parser.parse(resp.text[:500_000].splitlines())  # type: ignore[union-attr]
        except BlockedAddressError:
            raise
        except (httpx.HTTPError, httpcore.ConnectError, UnicodeError):
            parser = None  # unreachable robots file is treated as absent
        self._robots[key] = parser
        return parser

    def allowed_by_robots(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parts = urlsplit(url)
        parser = self._robots_for(parts.scheme, parts.netloc)
        return True if parser is None else parser.can_fetch(self.user_agent, url)

    # -- fetch ------------------------------------------------------------

    def fetch(self, url: str) -> FetchResult:
        started = self._monotonic()

        def done(result: FetchResult) -> FetchResult:
            result.duration_ms = int((self._monotonic() - started) * 1000)
            return result

        reason = check_fetchable(url)
        if reason:
            return done(FetchResult(url=url, ok=False, outcome="BLOCKED_URL", error=reason))
        host = urlsplit(url).netloc
        try:
            if not self.allowed_by_robots(url):
                return done(FetchResult(url=url, ok=False, outcome="BLOCKED_ROBOTS", error="disallowed by robots.txt"))
            self._pace(host)
            with self._client.stream("GET", url) as resp:
                final_url = str(resp.url)
                content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
                if check_fetchable(final_url):
                    return done(FetchResult(url=url, ok=False, outcome="BLOCKED_URL", final_url=final_url,
                                            error=f"redirect target refused: {check_fetchable(final_url)}"))
                if resp.status_code >= 400:
                    return done(FetchResult(url=url, ok=False, outcome="HTTP_ERROR", final_url=final_url,
                                            http_status=resp.status_code, content_type=content_type,
                                            error=f"HTTP {resp.status_code}"))
                kind = _classify(content_type, final_url)
                if kind is None:
                    return done(FetchResult(url=url, ok=False, outcome="UNSUPPORTED_TYPE", final_url=final_url,
                                            http_status=resp.status_code, content_type=content_type,
                                            error=f"unsupported content type '{content_type}'"))
                chunks, size, truncated = [], 0, False
                for chunk in resp.iter_bytes():
                    size += len(chunk)
                    if size > self.max_bytes:
                        truncated = True
                        break
                    chunks.append(chunk)
                    if self._monotonic() - started > self.timeout * 2:
                        return done(FetchResult(url=url, ok=False, outcome="TIMEOUT", final_url=final_url,
                                                http_status=resp.status_code, error="body read exceeded time limit"))
                data = b"".join(chunks)
                encoding = resp.encoding or "utf-8"
        except BlockedAddressError as exc:
            return done(FetchResult(url=url, ok=False, outcome="BLOCKED_ADDRESS", error=str(exc)))
        except httpx.TimeoutException as exc:
            return done(FetchResult(url=url, ok=False, outcome="TIMEOUT", error=type(exc).__name__))
        except httpx.TooManyRedirects:
            return done(FetchResult(url=url, ok=False, outcome="NETWORK_ERROR", error="too many redirects"))
        except (httpx.HTTPError, httpcore.ConnectError, UnicodeError, ValueError) as exc:
            return done(FetchResult(url=url, ok=False, outcome="NETWORK_ERROR", error=f"{type(exc).__name__}: {str(exc)[:200]}"))

        if urlsplit(final_url).netloc != host:
            try:
                if not self.allowed_by_robots(final_url):
                    return done(FetchResult(url=url, ok=False, outcome="BLOCKED_ROBOTS", final_url=final_url,
                                            error="redirect target disallowed by robots.txt"))
            except BlockedAddressError as exc:
                return done(FetchResult(url=url, ok=False, outcome="BLOCKED_ADDRESS", final_url=final_url, error=str(exc)))
        if truncated and kind == "pdf":
            return done(FetchResult(url=url, ok=False, outcome="TOO_LARGE", final_url=final_url, http_status=200,
                                    content_type=content_type, error=f"PDF larger than {self.max_bytes} bytes"))
        if kind == "pdf":
            extracted = extract_pdf(data)
        else:
            try:
                text = data.decode(encoding, errors="replace")
            except LookupError:
                text = data.decode("utf-8", errors="replace")
            extracted = extract_html(text, final_url) if kind == "html" else extract_plain(text)
        flags = list(extracted.flags) + (["body_truncated"] if truncated else [])
        result = FetchResult(
            url=url, ok=bool(extracted.text), outcome="OK" if extracted.text else "EMPTY_TEXT", final_url=final_url,
            http_status=200, content_type=content_type, title=extracted.title, published_at=extracted.published_at,
            text=extracted.text, flags=flags, bytes_read=len(data),
            error=None if extracted.text else "no readable text extracted",
        )
        return done(result)


def _classify(content_type: str, url: str) -> str | None:
    if content_type in ("text/html", "application/xhtml+xml", ""):
        return "pdf" if urlsplit(url).path.lower().endswith(".pdf") and content_type == "" else "html"
    if content_type == "application/pdf":
        return "pdf"
    if content_type in ("text/plain", "text/markdown", "application/json", "application/xml", "text/xml",
                        "application/rss+xml", "application/atom+xml"):
        return "text"
    return None
