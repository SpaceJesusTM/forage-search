"""Shared domain vocabulary: statuses, the watch specification and session budgets."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class WatchStatus(StrEnum):
    DRAFT = "DRAFT"
    PENDING_CONFIRMATION = "PENDING_CONFIRMATION"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"  # user-requested
    AWAITING_REVIEW = "AWAITING_REVIEW"  # agent-requested checkpoint
    REVIEW_PROPOSED = "REVIEW_PROPOSED"  # resume/reopen plan waiting for user confirmation
    CLOSED = "CLOSED"


class RevisionState(StrEnum):
    PROPOSED = "PROPOSED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"


class RunStatus(StrEnum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"


class SourceStatus(StrEnum):
    DISCOVERED = "DISCOVERED"  # seen in search results, not fetched yet
    FETCHED = "FETCHED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"  # robots.txt, address guard or unsupported content
    IRRELEVANT = "IRRELEVANT"  # triaged or read and judged off-objective


class ChangeType(StrEnum):
    NEW = "NEW"
    UPDATED = "UPDATED"  # an existing finding gained new information
    CORROBORATED = "CORROBORATED"  # an independent source repeated what was already known
    UNCHANGED = "UNCHANGED"  # re-observed; nothing new
    STALE = "STALE"  # evidence could no longer be confirmed

    @property
    def rank(self) -> int:
        return {"NEW": 4, "UPDATED": 3, "STALE": 2, "CORROBORATED": 1, "UNCHANGED": 0}[self.value]


class Verification(StrEnum):
    QUOTE_EXACT = "QUOTE_EXACT"
    QUOTE_FUZZY = "QUOTE_FUZZY"
    QUOTE_NOT_FOUND = "QUOTE_NOT_FOUND"


class Support(StrEnum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"
    UNCHECKED = "UNCHECKED"


class ClaimStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"  # the source changed and no longer contains the quote


class QuestionStatus(StrEnum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"
    DROPPED = "DROPPED"


def _clean_list(values: list[str] | None, limit: int, max_len: int = 300) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values or []:
        text = " ".join(str(value).split())[:max_len]
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out[:limit]


class WatchSpec(BaseModel):
    """Structured form of a user-approved objective. Only a human can approve changes to it."""

    key_questions: list[str] = Field(default_factory=list, description="Concrete questions the research should answer")
    include: list[str] = Field(default_factory=list, description="What counts as in scope")
    exclude: list[str] = Field(default_factory=list, description="What is out of scope")
    priorities: list[str] = Field(default_factory=list, description="What matters most when ranking results")
    geography: str | None = None
    freshness_days: int | None = Field(default=None, description="Re-check sources older than this many days")
    source_hints: list[str] = Field(default_factory=list, description="Sites or URLs the user suggested")
    notes: str | None = None

    @field_validator("key_questions", "include", "exclude", "priorities", "source_hints", mode="before")
    @classmethod
    def _lists(cls, value):
        return _clean_list(value, limit=12)

    @field_validator("freshness_days", mode="before")
    @classmethod
    def _freshness(cls, value):
        if value in (None, "", 0):
            return None
        return max(1, min(int(value), 3650))


class Budget(BaseModel):
    """Hard limits for one bounded session."""

    max_duration_seconds: int = Field(900, ge=10, le=6 * 3600)
    max_search_requests: int = Field(6, ge=0, le=200)
    max_fetched_pages: int = Field(8, ge=0, le=500)
    max_llm_output_tokens: int = Field(9000, ge=0, le=2_000_000)
