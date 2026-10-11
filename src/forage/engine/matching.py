"""Lexical similarity for deduplication.

No embedding model is assumed. Similarity is IDF-weighted token overlap, which is
exact enough to decide clear cases; the ambiguous middle band is handed to the
LLM by the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from forage.util import jaccard, normalize_text, token_set, tokenize

# Finding-level thresholds (IDF-weighted score in [0, 1]).
SAME_FINDING = 0.62
MAYBE_FINDING = 0.10  # below this, an item is filed as new without asking the model
# Claim-level thresholds.
SAME_CLAIM = 0.80
MAYBE_CLAIM = 0.42
# Query-level threshold.
REPEAT_QUERY = 0.75


class Idf:
    """Inverse document frequency over a small in-memory corpus of token sets."""

    def __init__(self, docs: list[frozenset[str]]):
        self.n = len(docs)
        self.df: dict[str, int] = {}
        for doc in docs:
            for tok in doc:
                self.df[tok] = self.df.get(tok, 0) + 1

    def weight(self, token: str) -> float:
        return math.log((self.n + 1) / (self.df.get(token, 0) + 0.5)) + 0.2

    def score(self, a: frozenset[str], b: frozenset[str]) -> float:
        """Blend of weighted cosine and weighted containment; 1.0 for identical sets."""
        if not a or not b:
            return 0.0
        shared = a & b
        if not shared:
            return 0.0
        w_shared = sum(self.weight(t) ** 2 for t in shared)
        w_a = sum(self.weight(t) ** 2 for t in a)
        w_b = sum(self.weight(t) ** 2 for t in b)
        cosine = w_shared / math.sqrt(w_a * w_b)
        containment = w_shared / min(w_a, w_b)
        return 0.6 * cosine + 0.4 * containment


def text_similarity(a: str, b: str) -> float:
    """Unweighted similarity of two short texts: mean of Jaccard and containment."""
    ta, tb = token_set(a), token_set(b)
    if not ta or not tb:
        return 0.0
    shared = len(ta & tb)
    return 0.5 * shared / len(ta | tb) + 0.5 * shared / min(len(ta), len(tb))


def is_repeat_query(query: str, tried: list[str]) -> str | None:
    """Return the earlier query this one repeats, if any.

    Repeats are: the same normalised text, the same word set in another order, a
    high word overlap, or one query that only adds or drops a single word.
    """
    norm = normalize_text(query)
    words = set(tokenize(query))
    if not words:
        return query
    for old in tried:
        if normalize_text(old) == norm:
            return old
        old_words = set(tokenize(old))
        if not old_words:
            continue
        if jaccard(words, old_words) >= REPEAT_QUERY:
            return old
        small, large = sorted((words, old_words), key=len)
        if small <= large and len(large) - len(small) <= 1 and len(small) >= 3:
            return old
    return None


@dataclass
class Candidate:
    key: str
    score: float


def rank_candidates(item_text: str, item_title: str, existing: list[tuple[str, str, str]], limit: int = 3) -> list[Candidate]:
    """Rank existing entries (key, title, full_text) by similarity to a new item, best first."""
    if not existing:
        return []
    item_doc = token_set(item_text)
    item_head = token_set(item_title)
    docs = [token_set(text) for _, _, text in existing]
    idf = Idf(docs + [item_doc])
    ranked = []
    for (key, title, _), doc in zip(existing, docs):
        body = idf.score(item_doc, doc)
        head = idf.score(item_head, token_set(title)) if item_head else 0.0
        ranked.append(Candidate(key, max(body, 0.5 * body + 0.5 * head)))
    ranked.sort(key=lambda c: -c.score)
    return ranked[:limit]
