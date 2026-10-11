"""URL canonicalisation and classification. Pure functions, no network."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit

_TRACKING_PREFIXES = ("utm_", "pk_", "mtm_", "hsa_", "vero_", "oly_")
_TRACKING_PARAMS = frozenset(
    """fbclid gclid gclsrc dclid msclkid yclid igshid mc_cid mc_eid ref ref_src ref_url referrer source
    src share spm scid cmpid campaign trk trkinfo tracking_id _hsenc _hsmi _ga _gl s_cid ncid cid
    amp amp_js_v usqp""".split()
)
_SKIP_EXTENSIONS = frozenset(
    """jpg jpeg png gif webp svg ico bmp tiff mp3 mp4 m4a m4v mov avi wmv flv webm ogg wav zip tar gz
    tgz bz2 7z rar exe dmg pkg msi deb rpm apk iso bin css js woff woff2 ttf eot otf pptx ppt xlsx xls
    docx doc epub mobi""".split()
)
_ALLOWED_PORTS = frozenset({None, 80, 443, 8080, 8443})
_TWO_LEVEL_SUFFIXES = frozenset(
    """co.uk ac.uk gov.uk org.uk com.au net.au org.au edu.au gov.au co.nz co.jp ne.jp or.jp ac.jp
    com.br com.cn com.hk com.sg com.tw com.mx co.in co.kr co.za on.ca qc.ca bc.ca ab.ca gc.ca
    github.io gitlab.io substack.com medium.com blogspot.com wordpress.com readthedocs.io""".split()
)
_ARXIV_RE = re.compile(r"^/(?:abs|pdf|html)/(?P<id>(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[a-z]{2})?/\d{7}))(?:v\d+)?(?:\.pdf)?/?$", re.I)


class UrlError(ValueError):
    pass


def _host(parts) -> str:
    host = (parts.hostname or "").rstrip(".").lower()
    try:
        return host.encode("idna").decode("ascii") if host and not host.isascii() else host
    except UnicodeError:
        return host


def canonicalize_url(url: str) -> str:
    """Identity form of a URL for exact deduplication.

    Drops fragments, tracking parameters, default ports, `www.` and scheme differences,
    sorts the remaining query and normalises trailing slashes. Not meant for fetching.
    """
    url = (url or "").strip()
    if "://" not in url:
        url = "https://" + url.lstrip("/")
    parts = urlsplit(url)
    if parts.scheme.lower() not in ("http", "https"):
        raise UrlError(f"unsupported scheme: {parts.scheme!r}")
    host = _host(parts)
    if not host:
        raise UrlError("missing host")
    if host.startswith("www."):
        host = host[4:]
    if re.fullmatch(r"[a-z]{2,3}\.m\.wikipedia\.org", host):
        host = host.replace(".m.", ".")
    try:
        port = parts.port
    except ValueError as exc:
        raise UrlError("invalid port") from exc
    netloc = host if port in (None, 80, 443) else f"{host}:{port}"

    path = quote(unquote(parts.path or "/"), safe="/:@!$&'()*+,;=-._~%")
    path = re.sub(r"/{2,}", "/", path)
    path = re.sub(r"/(?:index|default)\.(?:html?|php|aspx?)$", "/", path, flags=re.I)
    if len(path) > 1:
        path = path.rstrip("/")

    query_pairs = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if k.lower() not in _TRACKING_PARAMS and not k.lower().startswith(_TRACKING_PREFIXES)
    ]

    if host in ("arxiv.org", "export.arxiv.org"):
        match = _ARXIV_RE.match(path)
        if match:
            return f"https://arxiv.org/abs/{match.group('id').lower()}"

    query = urlencode(sorted(query_pairs))
    return urlunsplit(("https", netloc, path or "/", query, ""))


def registrable_domain(url_or_host: str) -> str:
    """Approximate eTLD+1, used to count independent sources and to cap per-site fetches."""
    host = url_or_host
    if "://" in url_or_host:
        host = _host(urlsplit(url_or_host))
    host = host.lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    if ".".join(labels[-2:]) in _TWO_LEVEL_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def check_fetchable(url: str) -> str | None:
    """Return None if the URL may be requested, otherwise a short reason."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return "malformed url"
    if parts.scheme.lower() not in ("http", "https"):
        return f"scheme '{parts.scheme}' not allowed"
    if parts.username or parts.password:
        return "credentials in url"
    host = _host(parts)
    if not host or "." not in host and ":" not in host:
        return "missing or unqualified host"
    try:
        if parts.port not in _ALLOWED_PORTS:
            return f"port {parts.port} not allowed"
    except ValueError:
        return "invalid port"
    ext = parts.path.rsplit(".", 1)[-1].lower() if "." in parts.path.rsplit("/", 1)[-1] else ""
    if ext in _SKIP_EXTENSIONS:
        return f"unsupported file type .{ext}"
    if len(url) > 2000:
        return "url too long"
    return None


def is_public_ip(address: str) -> bool:
    """True only for globally routable addresses (rejects private, loopback, link-local, CGNAT, reserved)."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return bool(ip.is_global) and not ip.is_multicast
