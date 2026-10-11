"""LIVE smoke tests: the real local Ollama server and the public web.

Deselected by default. Run with:  pytest -m live
Each test is small and bounded. A failure here can mean the environment is
unavailable (model server down, search provider throttling), not a code defect.
"""

import pytest
from pydantic import BaseModel

from forage.config import load_settings

pytestmark = pytest.mark.live


class Reply(BaseModel):
    answer: str


def test_live_local_model_returns_schema_valid_output():
    from forage.app import build_llm

    settings = load_settings()
    llm = build_llm(settings)
    out, usage = llm.generate(purpose="live_smoke", system="Reply with JSON only.",
                              user='Return {"answer": "ready"}.', schema=Reply, max_tokens=20)
    assert out.answer
    assert usage.output_tokens > 0 and usage.prompt_tokens > 0


def test_live_search_returns_real_results():
    from forage.search.duckduckgo import DuckDuckGoSearch

    results = DuckDuckGoSearch(min_interval=0).search("University of Toronto MScAC program", 5)
    assert results, "no results (the provider may be throttling this address)"
    assert all(r.url.startswith("http") for r in results)


def test_live_fetch_retrieves_and_extracts_a_public_page():
    from forage.fetch.http import HttpFetcher

    settings = load_settings()
    result = HttpFetcher(user_agent=settings.user_agent).fetch("https://en.wikipedia.org/wiki/Tidal_power")
    assert result.ok and result.http_status == 200
    assert "tidal" in result.text.lower() and len(result.text) > 2000


def test_live_fetch_refuses_local_and_tailnet_addresses():
    from forage.fetch.http import HttpFetcher

    fetcher = HttpFetcher(user_agent="ForageTest/0", respect_robots=False)
    assert fetcher.fetch("http://127.0.0.1/").outcome == "BLOCKED_ADDRESS"
    assert fetcher.fetch("http://100.100.100.100/").outcome == "BLOCKED_ADDRESS"
