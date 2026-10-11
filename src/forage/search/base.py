"""Search adapter contract. Result titles and snippets are untrusted data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol


@dataclass
class SearchResult:
    rank: int
    url: str
    title: str
    snippet: str

    def as_dict(self) -> dict:
        return asdict(self)


class SearchError(Exception):
    """The provider failed for this query."""


class SearchRateLimited(SearchError):
    """The provider is throttling us; back off before retrying."""

    def __init__(self, message: str, retry_after: float = 60.0):
        super().__init__(message)
        self.retry_after = retry_after


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int = 8) -> list[SearchResult]: ...


class FallbackSearch:
    """Try providers in order; move on when one errors or is rate limited."""

    def __init__(self, providers: list[SearchProvider]):
        if not providers:
            raise ValueError("at least one provider is required")
        self.providers = providers
        self.name = "+".join(p.name for p in providers)
        self.last_used: str | None = None

    def search(self, query: str, max_results: int = 8) -> list[SearchResult]:
        last: SearchError | None = None
        for provider in self.providers:
            try:
                results = provider.search(query, max_results)
            except SearchError as exc:
                last = exc
                continue
            self.last_used = provider.name
            if results:
                return results
        if last is not None:
            raise last
        return []
