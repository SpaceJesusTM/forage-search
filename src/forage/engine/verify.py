"""Deterministic evidence checks. A claim is reportable only if these pass.

1. The quote the model gave must be located in the stored page text. The text kept
   as evidence is the page's own wording at the matched span, not the model's copy.
2. Every number in the claim must appear somewhere on the page.
3. Every proper name in the claim must appear somewhere on the page.
4. Claim and quote must share vocabulary, and neither may read like an instruction.

Names or numbers that are on the page but outside the quoted passage are allowed,
flagged, and cap the claim at "partly supported".

Passing means "this source says this", not "this is true".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from forage.engine.safety import scan_for_injection
from forage.types import Verification
from forage.util import PUNCT_MAP, token_set

MIN_QUOTE_CHARS = 12
MIN_QUOTE_TOKENS = 3
MAX_QUOTE_CHARS = 600
FUZZY_MIN_TOKENS = 6
FUZZY_MIN_RATIO = 0.86
MAX_SPAN_CHARS = 1500


@dataclass
class QuoteMatch:
    status: Verification
    start: int = -1
    end: int = -1
    text: str = ""  # the page's own wording
    ratio: float = 0.0


@dataclass
class ClaimCheck:
    accepted: bool
    verification: Verification
    quote: str = ""
    offset: int = -1
    reason: str | None = None  # why it was rejected
    notes: list[str] = field(default_factory=list)  # caveats on an accepted claim
    numbers_outside_quote: bool = False
    names_outside_quote: bool = False

    @property
    def details_outside_quote(self) -> bool:
        return self.numbers_outside_quote or self.names_outside_quote


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalise like `util.normalize_text` while recording each output char's source index."""
    out: list[str] = []
    index: list[int] = []
    pending_space = False
    for i, ch in enumerate(text):
        for piece in unicodedata.normalize("NFKC", ch).translate(PUNCT_MAP).casefold():
            if piece.isspace():
                pending_space = bool(out)
                continue
            if pending_space:
                out.append(" ")
                index.append(i)
                pending_space = False
            out.append(piece)
            index.append(i)
    return "".join(out), index


def _norm(text: str) -> str:
    return normalize_with_map(text)[0]


def _strip_wrapping(quote: str) -> str:
    quote = quote.strip().strip("\"'“”‘’`").strip()
    return re.sub(r"^(?:\.{3}|…)\s*|\s*(?:\.{3}|…)$", "", quote).strip()


def _span(text: str, index: list[int], start: int, end: int) -> tuple[int, int, str]:
    o_start = index[start]
    o_end = index[end - 1] + 1
    return o_start, o_end, text[o_start:o_end]


def locate_quote(quote: str, text: str, _pre: tuple[str, list[int]] | None = None) -> QuoteMatch:
    """Find `quote` in `text`: exactly, as ordered segments around ellipses, or as a close token match."""
    norm_text, index = _pre or normalize_with_map(text)
    quote = _strip_wrapping(quote or "")[:MAX_QUOTE_CHARS]
    norm_quote = _norm(quote)
    if len(norm_quote) < MIN_QUOTE_CHARS or len(norm_quote.split()) < MIN_QUOTE_TOKENS:
        return QuoteMatch(Verification.QUOTE_NOT_FOUND)

    pos = norm_text.find(norm_quote)
    if pos >= 0:
        start, end, found = _span(text, index, pos, pos + len(norm_quote))
        return QuoteMatch(Verification.QUOTE_EXACT, start, end, found, 1.0)

    # Trailing punctuation the model added or dropped.
    trimmed = norm_quote.strip(" .,;:!?\"'()[]-")
    if trimmed != norm_quote and len(trimmed) >= MIN_QUOTE_CHARS:
        pos = norm_text.find(trimmed)
        if pos >= 0:
            start, end, found = _span(text, index, pos, pos + len(trimmed))
            return QuoteMatch(Verification.QUOTE_EXACT, start, end, found, 1.0)

    # Elided or stitched quotes: every segment must appear, in order, within a bounded span.
    segments = [s.strip(" .,;:") for s in re.split(r"\s*(?:\[\s*)?(?:\.{3,}|…)(?:\s*\])?\s*", norm_quote)]
    segments = [s for s in segments if len(s.split()) >= MIN_QUOTE_TOKENS]
    if len(segments) < 2:
        sentence_parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", norm_quote)]
        segments = [s.strip(" .,;:!?") for s in sentence_parts if len(s.split()) >= MIN_QUOTE_TOKENS]
        if len(segments) < 2 or len(segments) != len([s for s in sentence_parts if s]):
            segments = []
    if len(segments) >= 2:
        cursor, first, last = 0, -1, -1
        for segment in segments:
            pos = norm_text.find(segment, cursor)
            if pos < 0:
                first = -1
                break
            first = pos if first < 0 else first
            last = pos + len(segment)
            cursor = last
        if first >= 0 and last - first <= MAX_SPAN_CHARS:
            start, end, found = _span(text, index, first, last)
            return QuoteMatch(Verification.QUOTE_EXACT, start, end, found, 1.0)

    if "|" in quote:
        row = _table_row(quote, text)
        if row is not None:
            return row

    return _fuzzy(norm_quote, norm_text, text, index)


def _table_row(quote: str, text: str) -> QuoteMatch | None:
    """Match a quote made of table cells against one table row of the page.

    Models often quote the label cell and the value cell they need, skipping cells in
    between. The evidence kept is the whole row, preceded by the table's header row.
    """
    wanted = [_norm(cell) for cell in quote.split("|")]
    wanted = [cell for cell in wanted if cell]
    if len(wanted) < 2:
        return None
    offset = 0
    header: str | None = None
    previous_was_row = False
    for line in text.split("\n"):
        is_row = "|" in line
        if is_row:
            cells = [_norm(cell) for cell in line.split("|")]
            is_separator = all(not cell.strip("-: ") for cell in cells)
            if not previous_was_row:
                header = line.strip()
            if not is_separator:
                position = 0
                for cell in wanted:
                    try:
                        position = cells.index(cell, position) + 1
                    except ValueError:
                        position = -1
                        break
                if position > 0:
                    row = line.strip()
                    shown = row if header in (None, row) else f"{header} // {row}"
                    return QuoteMatch(Verification.QUOTE_EXACT, offset, offset + len(line), shown, 1.0)
        previous_was_row = is_row
        offset += len(line) + 1
    return None


def claim_coverage(claim_text: str, quote: str) -> float:
    """Share of the claim's content words that also occur in the quote (0..1)."""
    claim = token_set(claim_text)
    if not claim:
        return 0.0
    return len(claim & token_set(quote)) / len(claim)


def _fuzzy(norm_quote: str, norm_text: str, text: str, index: list[int]) -> QuoteMatch:
    q_tokens = norm_quote.split()
    if len(q_tokens) < FUZZY_MIN_TOKENS:
        return QuoteMatch(Verification.QUOTE_NOT_FOUND)
    spans = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", norm_text)]
    positions: dict[str, list[int]] = {}
    for i, (tok, _, _) in enumerate(spans):
        positions.setdefault(tok, []).append(i)

    anchors = sorted(
        (i for i, tok in enumerate(q_tokens) if tok in positions and len(positions[tok]) <= 40),
        key=lambda i: (len(positions[q_tokens[i]]), -len(q_tokens[i])),
    )[:4]
    best: tuple[float, int, int] | None = None
    seen: set[int] = set()
    slack = max(2, len(q_tokens) // 6)
    for anchor in anchors:
        for page_pos in positions[q_tokens[anchor]]:
            lo = max(0, page_pos - anchor - slack)
            if lo in seen:
                continue
            seen.add(lo)
            hi = min(len(spans), page_pos - anchor + len(q_tokens) + slack)
            window = [tok for tok, _, _ in spans[lo:hi]]
            blocks = [b for b in SequenceMatcher(None, q_tokens, window, autojunk=False).get_matching_blocks() if b.size]
            if not blocks:
                continue
            ratio = sum(b.size for b in blocks) / len(q_tokens)
            if best is None or ratio > best[0]:
                best = (ratio, lo + blocks[0].b, lo + blocks[-1].b + blocks[-1].size)
    if best is None or best[0] < FUZZY_MIN_RATIO:
        return QuoteMatch(Verification.QUOTE_NOT_FOUND, ratio=best[0] if best else 0.0)
    ratio, tok_start, tok_end = best
    # The matched page span must not be much longer than the quote (no stitching across the page).
    if tok_end - tok_start > len(q_tokens) * 1.4 + 2:
        return QuoteMatch(Verification.QUOTE_NOT_FOUND, ratio=ratio)
    start, end, found = _span(text, index, spans[tok_start][1], spans[tok_end - 1][2])
    return QuoteMatch(Verification.QUOTE_FUZZY, start, end, found, ratio)


_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> list[str]:
    """Digit sequences with thousands separators removed, e.g. '1,200' -> '1200', '3.50' -> '3.50'."""
    out = []
    for raw in _NUMBER_RE.findall(text or ""):
        value = raw.replace(",", "").rstrip(".")
        if value and value not in out:
            out.append(value)
    return out


def _has_number(number: str, haystack_numbers: set[str]) -> bool:
    if number in haystack_numbers:
        return True
    # '2026' also matches '2026.0'-style renderings and vice versa; '05' matches '5'.
    stripped = number.lstrip("0") or "0"
    return any((h.lstrip("0") or "0") == stripped for h in haystack_numbers)


_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]*[A-Za-z0-9]|[A-Za-z]")
_MONTHS = {m: m[:3] for m in ("january february march april june july august september october november december").split()}
_MONTHS["sept"] = "sep"


def _squash(text: str) -> str:
    """Lowercase letters and digits only, so 'Llama 3.1 8B' can be found inside 'llama3.1:8b'."""
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKC", text).casefold())


# Sentence openers that are capitalised only because they come first.
_OPENERS = frozenset("""the a an this that these those it its in on at for with when if by to from as both each most
some all no not there their they we our one two three four five many several other another such after before during
while since although however under over between using according based only also more less new any every how what why
where who which research data evidence output input performance accuracy support use usage access cost price work
staff faculty funding demand study page source paper article report author site team group company model method
approach technique system tool result test benchmark program programme course event job role position""".split())
_COMMON_WORD_ENDINGS = ("ing", "tion", "sion", "ment", "ance", "ence", "ity", "ness", "ship", "ism", "ers", "ors",
                        "ed", "ly", "ive", "able", "ible", "ous", "ful", "less")


def _looks_like_sentence_opener(word: str) -> bool:
    """True for a first word that is probably an ordinary word rather than a proper name."""
    lower = word.casefold()
    if lower in _OPENERS or lower.rstrip("s") in _OPENERS:
        return True
    if lower.endswith("s") and not lower.endswith("ss"):  # plural common noun far more often than a name
        return True
    return lower.endswith(_COMMON_WORD_ENDINGS)


def names_in(claim_text: str) -> list[str]:
    """Words in a claim that look like proper names or identifiers.

    A word counts if it has a capital letter after its first character, mixes letters
    and digits (model sizes, versions, product codes), or is capitalised. The first
    word of the sentence is included only when it does not look like an ordinary
    capitalised opener.
    """
    out: list[str] = []
    for i, match in enumerate(_NAME_RE.finditer(claim_text or "")):
        word = match.group(0).rstrip(".")
        has_digit = any(ch.isdigit() for ch in word)
        inner_capital = any(ch.isupper() for ch in word[1:])
        capitalised = word[0].isupper() and (i > 0 or not _looks_like_sentence_opener(word))
        if not (has_digit or inner_capital or capitalised):
            continue
        if len(_squash(word)) >= 3 and word not in out:
            out.append(word)
    return out


def _name_present(name: str, squashed_haystack: str) -> bool:
    key = _squash(name)
    if key in squashed_haystack:
        return True
    short = _MONTHS.get(key)
    return bool(short and short in squashed_haystack)


def check_claim(claim_text: str, quote: str, page_text: str, _pre: tuple[str, list[int]] | None = None,
                context: str = "") -> ClaimCheck:
    """Run every deterministic check for one extracted claim.

    `context` is trusted-enough page metadata (title, site) in which names may also be found.
    """
    claim_text = " ".join((claim_text or "").split())
    if not 15 <= len(claim_text) <= 500:
        return ClaimCheck(False, Verification.QUOTE_NOT_FOUND, reason="claim_length")
    if scan_for_injection(claim_text) or scan_for_injection(quote or ""):
        return ClaimCheck(False, Verification.QUOTE_NOT_FOUND, reason="instruction_like_text")

    match = locate_quote(quote, page_text, _pre)
    if match.status is Verification.QUOTE_NOT_FOUND:
        return ClaimCheck(False, match.status, reason="quote_not_found")

    check = ClaimCheck(True, match.status, quote=" ".join(match.text.split()), offset=match.start)
    if match.status is Verification.QUOTE_FUZZY:
        check.notes.append(f"quote matched approximately ({match.ratio:.0%} of words)")

    claim_numbers = numbers_in(claim_text)
    if claim_numbers:
        quote_numbers = set(numbers_in(match.text))
        missing = [n for n in claim_numbers if not _has_number(n, quote_numbers)]
        if missing:
            page_numbers = set(numbers_in(page_text))
            absent = [n for n in missing if not _has_number(n, page_numbers)]
            if absent:
                return ClaimCheck(False, match.status, reason="number_not_in_source",
                                  notes=[f"number(s) not on the page: {', '.join(absent[:4])}"])
            check.numbers_outside_quote = True
            check.notes.append(f"number(s) appear on the page but not in the quoted passage: {', '.join(missing[:4])}")

    if not token_set(claim_text) & token_set(match.text):
        return ClaimCheck(False, match.status, reason="claim_unrelated_to_quote")

    names = names_in(claim_text)
    if names:
        page_squashed = _squash(page_text + " " + context)
        absent = [n for n in names if not _name_present(n, page_squashed)]
        # Short all-caps words are usually generic acronyms the model supplied (API, LLM); anything
        # else that the page never mentions is an attribution the source did not make.
        fabricated = [n for n in absent if not (n.isupper() and len(n) <= 5)]
        if fabricated:
            return ClaimCheck(False, match.status, reason="name_not_in_source",
                              notes=[f"name(s) not on the page: {', '.join(fabricated[:4])}"])
        quote_squashed = _squash(match.text)
        outside = [n for n in names if n not in absent and not _name_present(n, quote_squashed)]
        if outside:
            check.names_outside_quote = True
            check.notes.append(f"name(s) appear on the page but not in the quoted passage: {', '.join(outside[:5])}")

    return check
