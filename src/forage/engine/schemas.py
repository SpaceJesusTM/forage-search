"""Shapes the local model must answer in. Validators clamp rather than reject where safe,
because a small model that returns five items when four were asked for is still useful."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


def _text(value, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


class WatchDraft(BaseModel):
    title: str = ""
    key_questions: list[str] = Field(default_factory=list)
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    priorities: list[str] = Field(default_factory=list)
    freshness_days: int | None = None

    @field_validator("title", mode="before")
    @classmethod
    def _title(cls, value):
        return _text(value, 90)

    @field_validator("freshness_days", mode="before")
    @classmethod
    def _freshness(cls, value):
        try:
            return max(1, min(365, int(value))) if value not in (None, "", 0) else None
        except (TypeError, ValueError):
            return None


class PlannedQuery(BaseModel):
    query: str
    rationale: str = ""
    question_id: str | None = None

    @field_validator("query", mode="before")
    @classmethod
    def _query(cls, value):
        return _text(value, 160)

    @field_validator("rationale", mode="before")
    @classmethod
    def _rationale(cls, value):
        return _text(value, 240)


class Plan(BaseModel):
    queries: list[PlannedQuery] = Field(default_factory=list)


class TriageRating(BaseModel):
    index: int
    score: int = 0
    out_of_scope: bool = False
    reason: str = ""

    @field_validator("score", mode="before")
    @classmethod
    def _score(cls, value):
        try:
            return max(0, min(3, int(value)))
        except (TypeError, ValueError):
            return 0

    @field_validator("reason", mode="before")
    @classmethod
    def _reason(cls, value):
        return _text(value, 160)


class Triage(BaseModel):
    ratings: list[TriageRating] = Field(default_factory=list)


class ExtractedClaim(BaseModel):
    text: str
    quote: str = ""

    @field_validator("text", mode="before")
    @classmethod
    def _claim(cls, value):
        return _text(value, 500)

    @field_validator("quote", mode="before")
    @classmethod
    def _quote(cls, value):
        return str(value or "").strip()[:700]


class ExtractedItem(BaseModel):
    title: str = ""
    subject: str = ""
    claims: list[ExtractedClaim] = Field(default_factory=list)
    uncertainty: str | None = None

    @field_validator("title", "subject", mode="before")
    @classmethod
    def _short(cls, value):
        return _text(value, 160)

    @field_validator("uncertainty", mode="before")
    @classmethod
    def _uncertainty(cls, value):
        text = _text(value, 300)
        return text if text and text.lower() not in {"none", "null", "n/a", "no"} else None


class Extraction(BaseModel):
    relevance: int = 0
    items: list[ExtractedItem] = Field(default_factory=list)

    @field_validator("relevance", mode="before")
    @classmethod
    def _relevance(cls, value):
        try:
            return max(0, min(3, int(value)))
        except (TypeError, ValueError):
            return 0


class SupportVerdict(BaseModel):
    index: int
    verdict: Literal["supported", "partial", "unsupported"] = "partial"
    bearing: Literal["direct", "background", "none"] = "direct"

    @field_validator("verdict", mode="before")
    @classmethod
    def _verdict(cls, value):
        text = str(value or "").strip().lower()
        return text if text in {"supported", "partial", "unsupported"} else "partial"

    @field_validator("bearing", mode="before")
    @classmethod
    def _bearing(cls, value):
        text = str(value or "").strip().lower()
        return text if text in {"direct", "background", "none"} else "direct"


class SupportCheck(BaseModel):
    verdicts: list[SupportVerdict] = Field(default_factory=list)


class ItemMatch(BaseModel):
    item: int
    same_as: int | None = None


class MatchBatch(BaseModel):
    matches: list[ItemMatch] = Field(default_factory=list)


class NoveltyVerdict(BaseModel):
    index: int
    adds_information: bool = True


class NoveltyCheck(BaseModel):
    verdicts: list[NoveltyVerdict] = Field(default_factory=list)


class ResolvedQuestion(BaseModel):
    question_id: str
    answer: str = ""
    finding_ids: list[str] = Field(default_factory=list)

    @field_validator("finding_ids", mode="before")
    @classmethod
    def _ids(cls, value):
        return [str(v).strip().strip("[]") for v in (value or [])][:6]

    @field_validator("answer", mode="before")
    @classmethod
    def _answer(cls, value):
        return _text(value, 400)


class NextDirection(BaseModel):
    query: str
    rationale: str = ""

    @field_validator("query", mode="before")
    @classmethod
    def _query(cls, value):
        return _text(value, 160)

    @field_validator("rationale", mode="before")
    @classmethod
    def _rationale(cls, value):
        return _text(value, 240)


class GuidanceProposal(BaseModel):
    """A decision the model thinks belongs to the user. Only a checkpoint policy can act on it."""

    needed: bool = False
    kind: Literal["scope", "priority", "strategy", "stalled", "other"] = "other"
    question: str = ""
    options: list[str] = Field(default_factory=list)
    reason: str = ""

    @field_validator("kind", mode="before")
    @classmethod
    def _kind(cls, value):
        text = str(value or "").strip().lower()
        return text if text in {"scope", "priority", "strategy", "stalled", "other"} else "other"

    @field_validator("question", "reason", mode="before")
    @classmethod
    def _short(cls, value):
        return _text(value, 400)

    @field_validator("options", mode="before")
    @classmethod
    def _options(cls, value):
        return [_text(v, 200) for v in (value or []) if _text(v, 200)][:4]


class Reflection(BaseModel):
    # Field order is the generation order under schema-constrained decoding: essentials first,
    # so a truncated reply still carries the summary and the next directions.
    summary: str = ""
    next_directions: list[NextDirection] = Field(default_factory=list)
    new_questions: list[str] = Field(default_factory=list)
    resolved: list[ResolvedQuestion] = Field(default_factory=list)
    guidance: GuidanceProposal | None = None

    @field_validator("summary", mode="before")
    @classmethod
    def _summary(cls, value):
        return _text(value, 900)

    @field_validator("new_questions", mode="before")
    @classmethod
    def _questions(cls, value):
        return [_text(v, 240) for v in (value or []) if len(_text(v, 240)) >= 12][:4]
