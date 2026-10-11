import pytest

from forage.fetch.urls import UrlError, canonicalize_url, check_fetchable, is_public_ip, registrable_domain


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("HTTP://www.Example.com:80/a//b/index.html?utm_source=x&b=2&a=1#frag", "https://example.com/a/b?a=1&b=2"),
        ("https://example.com/path/", "https://example.com/path"),
        ("https://example.com", "https://example.com/"),
        ("https://example.com/?fbclid=abc&gclid=def", "https://example.com/"),
        ("example.com/page?ref=home", "https://example.com/page"),
        ("https://EXAMPLE.com:8443/x", "https://example.com:8443/x"),
        ("https://en.m.wikipedia.org/wiki/Tide", "https://en.wikipedia.org/wiki/Tide"),
        ("https://arxiv.org/pdf/2605.02363v1", "https://arxiv.org/abs/2605.02363"),
        ("https://arxiv.org/abs/2605.02363v3", "https://arxiv.org/abs/2605.02363"),
        ("https://example.com/a?id=7&page=2", "https://example.com/a?id=7&page=2"),
    ],
)
def test_canonicalize(raw, expected):
    assert canonicalize_url(raw) == expected


def test_canonical_form_is_stable():
    once = canonicalize_url("https://www.example.com/a/?utm_campaign=x&z=1&y=2")
    assert canonicalize_url(once) == once


def test_meaningful_query_parameters_are_kept_distinct():
    assert canonicalize_url("https://example.com/job?id=1") != canonicalize_url("https://example.com/job?id=2")


@pytest.mark.parametrize("raw", ["ftp://example.com/x", "javascript:alert(1)", "file:///etc/passwd", "https:///nohost"])
def test_canonicalize_rejects_non_web_urls(raw):
    with pytest.raises(UrlError):
        canonicalize_url(raw)


@pytest.mark.parametrize(
    "url, domain",
    [
        ("https://mscac.utoronto.ca/x", "utoronto.ca"),
        ("https://www.cs.toronto.edu/", "toronto.edu"),
        ("https://news.bbc.co.uk/a", "bbc.co.uk"),
        ("https://someone.github.io/project", "someone.github.io"),
        ("https://example.com", "example.com"),
        ("http://203.0.113.9/x", "203.0.113.9"),
    ],
)
def test_registrable_domain(url, domain):
    assert registrable_domain(url) == domain


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/a",
        "http://user:secret@example.com/",
        "http://localhost/admin",
        "http://example.com:22/",
        "http://127.0.0.1:11434/api/tags",
        "https://example.com/photo.jpg",
        "https://example.com/setup.exe",
        "https://example.com/" + "a" * 2100,
    ],
)
def test_unfetchable_urls_are_refused(url):
    assert check_fetchable(url) is not None


@pytest.mark.parametrize("url", ["https://example.com/", "http://example.org/a/b?c=d", "https://example.com/paper.pdf"])
def test_ordinary_urls_are_fetchable(url):
    assert check_fetchable(url) is None


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "10.1.2.3", "192.168.1.10", "172.16.5.5", "169.254.169.254", "100.123.121.96", "0.0.0.0",
     "::1", "fe80::1", "fd7a:115c:a1e0::53", "::ffff:127.0.0.1", "224.0.0.1", "not-an-ip"],
)
def test_non_public_addresses_are_rejected(address):
    assert is_public_ip(address) is False


@pytest.mark.parametrize("address", ["8.8.8.8", "208.80.154.224", "2606:4700:4700::1111"])
def test_public_addresses_are_accepted(address):
    assert is_public_ip(address) is True
