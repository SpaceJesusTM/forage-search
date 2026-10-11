"""Handling of untrusted text: instruction-pattern detection, data-block wrapping, Markdown neutralising.

Detection is a tripwire, not the defence. The defence is structural: the model has
no tools, cannot name URLs to fetch, cannot alter the approved objective, and every
claim it returns must be located in the stored page text.
"""

from __future__ import annotations

import re

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("override_instructions", re.compile(
        r"\b(?:ignore|disregard|forget|override)\b[^.\n]{0,40}\b(?:previous|prior|above|earlier|all|any|your|the)\b[^.\n]{0,30}"
        r"\b(?:instruction|prompt|rule|direction|guideline|context)s?\b", re.I)),
    ("role_reassignment", re.compile(r"\byou are now\b|\bact as (?:an?|the)\b[^.\n]{0,40}\b(?:assistant|ai|model|agent)\b|\bfrom now on,? you\b", re.I)),
    ("addresses_model", re.compile(
        r"\b(?:ai|llm|language model|assistant|chatbot|agent)s?\b[^.\n]{0,25}\b(?:must|should|shall|are required to|need to)\b"
        r"[^.\n]{0,40}\b(?:report|say|state|respond|output|include|recommend|tell|rank|ignore)\b", re.I)),
    ("prompt_markup", re.compile(r"<\|?(?:im_start|im_end|system|endoftext)\|?>|\[/?INST\]|^\s*#{2,}\s*(?:system|instruction)s?\b|^\s*system\s*:", re.I | re.M)),
    ("system_prompt_probe", re.compile(r"\b(?:reveal|print|repeat|show|leak)\b[^.\n]{0,30}\b(?:system prompt|hidden prompt|instructions above)\b", re.I)),
    ("command_execution", re.compile(r"\b(?:run|execute)\b[^.\n]{0,25}\b(?:the following|this)\b[^.\n]{0,15}\b(?:command|script|code)\b|curl\s[^\n|]{0,120}\|\s*(?:ba)?sh\b", re.I)),
    ("credential_request", re.compile(r"\b(?:send|post|email|upload|share|include)\b[^.\n]{0,40}\b(?:api key|password|credential|token|secret|ssh key)s?\b", re.I)),
    ("new_instructions", re.compile(r"\bnew (?:instructions?|task|objective)\s*:|\binstead,? (?:you should|do the following)\b|\bdo not (?:mention|tell|reveal)\b[^.\n]{0,30}\b(?:user|this)\b", re.I)),
]

_BLOCK_TOKEN_RE = re.compile(r"<<<|>>>")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
REDACTION = "[instruction-like text removed]"


def scan_for_injection(text: str) -> list[str]:
    """Names of instruction-like patterns present in `text` (empty when none)."""
    return [name for name, pattern in _PATTERNS if pattern.search(text or "")]


def redact_injection(text: str) -> tuple[str, int]:
    """Replace sentences that contain instruction-like patterns. Returns (text, sentences_redacted).

    Only the copy shown to the model is redacted; the stored snapshot keeps the original
    so the page can still be audited.
    """
    lines = text.split("\n")
    redacted = 0
    for i, line in enumerate(lines):
        if not any(pattern.search(line) for _, pattern in _PATTERNS):
            continue
        # Redact sentence by sentence so facts sharing a paragraph with the offending text survive.
        sentences = _SENTENCE_RE.split(line)
        for j, sentence in enumerate(sentences):
            if any(pattern.search(sentence) for _, pattern in _PATTERNS):
                sentences[j] = REDACTION
                redacted += 1
        if not any(s == REDACTION for s in sentences):  # the pattern spanned a sentence boundary
            sentences = [REDACTION]
            redacted += 1
        lines[i] = " ".join(sentences)
    return "\n".join(lines), redacted


def data_block(label: str, text: str) -> str:
    """Wrap untrusted text in delimiters the content itself cannot reproduce."""
    safe = _BLOCK_TOKEN_RE.sub("", text or "")
    return f"<<<{label}\n{safe}\n{label}>>>"


def one_line(text: str, limit: int = 300) -> str:
    """Collapse untrusted text to a single bounded line for use inside prompts."""
    text = _BLOCK_TOKEN_RE.sub("", " ".join((text or "").split()))
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


_MD_SPECIAL_RE = re.compile(r"([\\`*_{}\[\]<>#|~])")


def md_text(text: str, limit: int = 600) -> str:
    """Render untrusted text inert in Markdown: one line, no markup, no links, no images."""
    text = " ".join((text or "").split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    text = _MD_SPECIAL_RE.sub(r"\\\1", text)
    text = text.replace("](", "]\\(").replace("![", "!\\[")
    return re.sub(r"(?i)\b(https?)://", r"\1:\\/\\/", text)


def md_url(url: str) -> str:
    """A URL safe to place inside <...> autolink brackets; empty if it is not http(s)."""
    url = (url or "").strip()
    if not re.match(r"(?i)^https?://", url):
        return ""
    return re.sub(r"[\s<>\"`\\]", lambda m: "%{:02X}".format(ord(m.group(0))), url)
