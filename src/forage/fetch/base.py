"""Fetch adapter contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class FetchResult:
    url: str
    ok: bool
    outcome: str  # OK | HTTP_ERROR | BLOCKED_ROBOTS | BLOCKED_ADDRESS | BLOCKED_URL | UNSUPPORTED_TYPE | TOO_LARGE | TIMEOUT | NETWORK_ERROR | EMPTY_TEXT
    final_url: str | None = None
    http_status: int | None = None
    content_type: str | None = None
    title: str | None = None
    published_at: str | None = None
    text: str = ""
    error: str | None = None
    flags: list[str] = field(default_factory=list)
    bytes_read: int = 0
    duration_ms: int = 0

    @property
    def blocked(self) -> bool:
        return self.outcome.startswith("BLOCKED") or self.outcome == "UNSUPPORTED_TYPE"


class Fetcher(Protocol):
    def fetch(self, url: str) -> FetchResult: ...
