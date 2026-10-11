"""Deterministic helpers: clocks, identifiers and text normalisation."""

from __future__ import annotations

import hashlib
import re
import secrets
import unicodedata
from datetime import datetime, timezone
from typing import Callable, Iterable

Clock = Callable[[], datetime]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    """UTC ISO-8601 with a trailing Z, second precision."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def sha1_hex(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(prefix: str, *parts: str, length: int = 12) -> str:
    """Identifier derived from content, so retries produce the same id."""
    return f"{prefix}-{sha1_hex(*parts)[:length]}"


def random_id(prefix: str, length: int = 10) -> str:
    return f"{prefix}-{secrets.token_hex(length // 2)}"


def slugify(text: str, max_len: int = 32) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text[:max_len].strip("-") or "watch"


PUNCT_MAP = str.maketrans(
    {
        "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
        "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
        "\u2013": "-", "\u2014": "-", "\u2010": "-", "\u2011": "-", "\u2212": "-",
        "\u00a0": " ", "\u202f": " ", "\u2009": " ", "\u200b": "", "\ufeff": "", "\u00ad": "",
    }
)


def normalize_text(text: str) -> str:
    """Canonical form for comparing text: NFKC, ASCII punctuation, single spaces, casefolded."""
    text = unicodedata.normalize("NFKC", text).translate(PUNCT_MAP)
    return re.sub(r"\s+", " ", text).strip().casefold()


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def truncate(text: str, limit: int, marker: str = "…") -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(marker))].rstrip() + marker


_STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in into is it
    its may might more most must no not of on or our should so some such than that the their them then there
    these they this those to was we were what when where which who why will with would you your about also
    any all each other over under up out new use used using via per vs""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.\-]*[a-z0-9+#]|[a-z0-9]")


def tokenize(text: str) -> list[str]:
    """Lowercased content tokens with stopwords removed and trivial plural folding."""
    out: list[str] = []
    for tok in _TOKEN_RE.findall(normalize_text(text)):
        if tok in _STOPWORDS:
            continue
        if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


def token_set(text: str) -> frozenset[str]:
    return frozenset(tokenize(text))


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def estimate_tokens(text: str) -> int:
    """Rough token estimate used only for prompt budgeting, never for accounting."""
    return max(1, len(text) // 4)
