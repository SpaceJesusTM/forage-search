"""Turn fetched bytes into plain text. Everything returned here is untrusted data."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

MAX_TEXT_CHARS = 120_000
MAX_PDF_PAGES = 20

_HIDDEN_STYLE_RE = re.compile(
    r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(?:px|pt|em|rem|%)?\s*(?:;|$)|opacity\s*:\s*0(?:\.0+)?\s*(?:;|$)",
    re.I,
)
_HIDDEN_CLASS_RE = re.compile(r"(?:^|\s)(?:sr-only|visually-hidden|screen-reader-text|hidden|d-none)(?:\s|$)", re.I)
_DROP_TAGS = ("script", "style", "noscript", "template", "iframe", "object", "embed", "svg", "canvas")
_BLOCK_TAGS = ("p", "div", "li", "tr", "br", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "blockquote",
               "pre", "table", "ul", "ol", "dd", "dt", "figcaption", "title")


@dataclass
class Extracted:
    text: str = ""
    title: str | None = None
    published_at: str | None = None
    flags: list[str] = field(default_factory=list)


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def strip_hidden_html(html: str) -> tuple[str, int]:
    """Remove content a human reader would not see. Returns (html, removed_element_count)."""
    from lxml import etree
    from lxml import html as lxml_html

    try:
        root = lxml_html.fromstring(html)
    except (etree.ParserError, ValueError):
        return html, 0
    removed = 0
    for comment in root.xpath("//comment()"):
        parent = comment.getparent()
        if parent is not None:
            parent.remove(comment)
    for tag in _DROP_TAGS:
        for el in root.xpath(f"//{tag}"):
            el.drop_tree()
    for el in root.xpath("//*[@hidden or @aria-hidden='true' or @style or @class]"):
        if el.getparent() is None:
            continue
        hidden = (
            el.get("hidden") is not None
            or el.get("aria-hidden") == "true"
            or bool(_HIDDEN_STYLE_RE.search(el.get("style") or ""))
            or bool(_HIDDEN_CLASS_RE.search(el.get("class") or ""))
        )
        if hidden:
            el.drop_tree()
            removed += 1
    return lxml_html.tostring(root, encoding="unicode"), removed


def extract_html(html: str, url: str | None = None) -> Extracted:
    import trafilatura
    from lxml import etree
    from lxml import html as lxml_html

    out = Extracted()
    if not html or not html.strip():
        return out
    cleaned, removed = strip_hidden_html(html)
    if removed:
        out.flags.append("hidden_content_removed")

    try:
        meta = trafilatura.extract_metadata(html, default_url=url)
    except Exception:  # third-party parser; a metadata failure must not lose the page
        meta = None
    if meta is not None:
        out.title = (meta.title or None) and " ".join(meta.title.split())[:300]
        out.published_at = meta.date or None

    text = None
    try:
        text = trafilatura.extract(
            cleaned, url=url, include_comments=False, include_tables=True, include_links=False,
            favor_recall=True, deduplicate=True,
        )
    except Exception:
        text = None
    if not text or len(text) < 200:
        try:
            body = lxml_html.fromstring(cleaned)
            for el in body.xpath("//nav|//footer|//header|//aside"):
                el.drop_tree()
            for el in body.iter(*_BLOCK_TAGS):  # text_content() would otherwise glue paragraphs together
                el.tail = "\n" + (el.tail or "")
            fallback = _clean_text(body.text_content())
        except (etree.ParserError, ValueError):
            fallback = ""
        if len(fallback) > len(text or ""):
            text = fallback
            out.flags.append("fallback_extraction")
    text = _clean_text(text or "")
    if out.title is None:
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
        if match:
            out.title = " ".join(re.sub(r"<[^>]+>", "", match.group(1)).split())[:300] or None
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
        out.flags.append("text_truncated")
    out.text = text
    return out


def extract_pdf(data: bytes) -> Extracted:
    from pypdf import PdfReader

    out = Extracted()
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for index, page in enumerate(reader.pages):
            if index >= MAX_PDF_PAGES:
                out.flags.append("pdf_pages_truncated")
                break
            pages.append(page.extract_text() or "")
        info = reader.metadata
        if info and info.title:
            out.title = " ".join(str(info.title).split())[:300] or None
    except Exception as exc:  # pypdf raises many types on malformed files
        out.flags.append(f"pdf_error:{type(exc).__name__}")
        return out
    text = _clean_text("\n\n".join(pages))
    # PDF text wraps mid-sentence; join single line breaks so quotes can be matched.
    text = re.sub(r"(?<![.!?:\n])\n(?!\n)", " ", text)
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
        out.flags.append("text_truncated")
    out.text = text
    out.flags.append("pdf")
    return out


def extract_plain(text: str) -> Extracted:
    out = Extracted(text=_clean_text(text)[:MAX_TEXT_CHARS])
    return out
