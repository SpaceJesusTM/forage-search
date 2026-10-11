"""Versioned report contract. The JSON form is the authority; Markdown is a rendering of it.

Adapters (a CLI, a web UI, a future Discord bridge) should depend on this schema
and nothing else from the engine.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


class ReportClaim(BaseModel):
    claim_id: str
    text: str = Field(description="What the source states, as written up by the local model")
    quote: str = Field(description="The source page's own words at the location that was matched")
    evidence_url: str
    source_id: str
    source_title: str | None = None
    source_domain: str
    checked_at: str
    verification_status: Literal["quote_verified", "quote_verified_approximate"]
    support: Literal["supported", "partial", "unchecked"]
    new_this_session: bool
    note: str | None = None


class ReportFinding(BaseModel):
    finding_id: str
    change_type: Literal["NEW", "UPDATED", "CORROBORATED", "UNCHANGED", "STALE"]
    title: str
    subject: str
    claims: list[ReportClaim]
    independent_sources: int = Field(description="Distinct registrable domains with a verified claim")
    uncertainties: list[str] = Field(default_factory=list)
    change_note: str | None = None
    retired_claims: list[str] = Field(default_factory=list, description="Claims this session could no longer confirm")
    first_seen_session: int | None = None
    first_seen_at: str
    last_confirmed_at: str


class ReportQuestion(BaseModel):
    question_id: str
    text: str
    origin: str
    status: str
    searches_so_far: int = 0
    answer: str | None = Field(default=None, description="Model-written; an inference from the cited findings")
    answered_by: list[str] = Field(default_factory=list)


class ReportDirection(BaseModel):
    query: str
    rationale: str | None = None


class ReportSource(BaseModel):
    source_id: str
    url: str
    title: str | None = None
    domain: str
    outcome: str = Field(description="read | unchanged | duplicate_content | failed | blocked | not_relevant")
    fetched_at: str | None = None
    http_status: int | None = None
    claims_kept: int = 0
    claims_rejected: int = 0
    flags: list[str] = Field(default_factory=list)
    note: str | None = None


class ReportSearch(BaseModel):
    query: str
    rationale: str | None = None
    provider: str
    status: str
    results: int = 0
    new_urls: int = 0
    known_urls: int = 0


class ReportComparison(BaseModel):
    """How this session relates to the research that existed when it started."""

    previous_run_id: str | None = None
    prior_sessions: int = 0
    known_findings_before: int = 0
    known_findings_after: int = 0
    open_questions_before: int = 0
    open_questions_after: int = 0
    queries_already_tried_before: int = 0
    queries_repeated: int = 0
    queries_rejected_as_repeats: int = 0
    known_urls_skipped: int = 0
    saved_leads_followed: int = 0
    sources_rechecked: int = 0
    duplicate_claims_suppressed: int = 0


class PendingReview(BaseModel):
    request_id: str
    policy: str
    signal: str
    question: str
    choices: list[dict[str, Any]]
    safe_default: str
    created_at: str


class ResearchReport(BaseModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    watch_id: str
    watch_title: str
    objective: str
    session_number: int
    status: Literal["completed", "cancelled", "failed", "interrupted", "running"]
    stop_reason: str | None = None
    error: str | None = None
    started_at: str
    finished_at: str | None = None
    model: str | None = None
    search_provider: str | None = None
    simulated: bool = Field(description="True when a fixture web or simulated model produced this report")
    has_meaningful_update: bool
    headline: str
    summary: str | None = Field(default=None, description="Model-written synthesis; inference, not source text")
    findings: list[ReportFinding]
    unchanged_findings: int = 0
    known_findings_total: int = 0
    questions_answered: list[ReportQuestion] = Field(default_factory=list)
    questions_opened: list[ReportQuestion] = Field(default_factory=list)
    questions_open: list[ReportQuestion] = Field(default_factory=list)
    next_directions: list[ReportDirection] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    sources: list[ReportSource] = Field(default_factory=list)
    searches: list[ReportSearch] = Field(default_factory=list)
    comparison: ReportComparison = Field(default_factory=ReportComparison)
    pending_review: PendingReview | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    budget: dict[str, Any] = Field(default_factory=dict)
