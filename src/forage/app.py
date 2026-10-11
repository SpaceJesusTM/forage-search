"""Composition root: builds adapters from settings and wires the layers together.

Interfaces (the CLI, a future web UI or Tari bridge) talk to an `App`. Tests pass
their own adapters in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from forage.config import Settings, load_settings
from forage.controller.reviews import ReviewController
from forage.controller.watches import WatchController
from forage.engine.session import ResearchEngine
from forage.fetch.base import Fetcher
from forage.fetch.http import HttpFetcher
from forage.llm.base import LLMClient
from forage.llm.ollama import OllamaClient
from forage.scheduler.runner import RunManager
from forage.search.base import FallbackSearch, SearchError, SearchProvider
from forage.simulated import FixtureFetcher, FixtureSearch, SimulatedLLM, SimulatedWeb
from forage.storage.store import Store


@dataclass
class App:
    settings: Settings
    store: Store
    llm: LLMClient
    search: SearchProvider
    fetcher: Fetcher
    engine: ResearchEngine
    runner: RunManager
    watches: WatchController
    reviews: ReviewController

    def close(self) -> None:
        self.store.close()


def build_llm(settings: Settings) -> LLMClient:
    if settings.model == "simulated":
        return SimulatedLLM()
    client = OllamaClient(
        settings.ollama_url, settings.model, num_ctx=settings.num_ctx, temperature=settings.temperature,
        timeout=settings.llm_timeout_seconds, think=settings.llm_think,
    )
    client.ensure_model(settings.model, settings.max_model_gb)
    return client


def _fixture_web(settings: Settings) -> SimulatedWeb:
    if not settings.fixture_web:
        raise SearchError("search_provider 'fixture' needs FORAGE_FIXTURE_WEB to point at a corpus file")
    return SimulatedWeb.from_file(settings.fixture_web, settings.fixture_epoch)


def build_search(settings: Settings, web: SimulatedWeb | None = None) -> SearchProvider:
    providers: list[SearchProvider] = []
    for name in [n.strip() for n in settings.search_provider.split("+") if n.strip()]:
        if name == "duckduckgo":
            from forage.search.duckduckgo import DuckDuckGoSearch

            providers.append(DuckDuckGoSearch(min_interval=settings.search_min_interval_seconds))
        elif name == "ddgs":
            from forage.search.extra import DdgsSearch

            providers.append(DdgsSearch())
        elif name == "searxng":
            from forage.search.extra import SearxngSearch

            providers.append(SearxngSearch(settings.searxng_url))
        elif name == "fixture":
            providers.append(FixtureSearch(web or _fixture_web(settings)))
        else:
            raise SearchError(f"unknown search provider '{name}'")
    if not providers:
        raise SearchError("no search provider configured")
    return providers[0] if len(providers) == 1 else FallbackSearch(providers)


def build_fetcher(settings: Settings, web: SimulatedWeb | None = None) -> Fetcher:
    if settings.search_provider == "fixture":
        return FixtureFetcher(web or _fixture_web(settings))
    return HttpFetcher(
        user_agent=settings.user_agent, timeout=settings.fetch_timeout_seconds, max_bytes=settings.fetch_max_bytes,
        respect_robots=settings.respect_robots, min_interval_per_host=settings.fetch_min_interval_per_host_seconds,
        allow_private=settings.allow_private_addresses,
    )


def build_app(settings: Settings | None = None, *, llm: LLMClient | None = None, search: SearchProvider | None = None,
              fetcher: Fetcher | None = None, progress: Callable[[str], None] | None = None, clock=None,
              sleep=None, monotonic=None) -> App:
    settings = settings or load_settings()
    kwargs = {k: v for k, v in (("clock", clock), ("sleep", sleep), ("monotonic", monotonic)) if v is not None}
    store = Store(settings.db_path, **({"clock": clock} if clock else {}))
    web = None
    if settings.search_provider == "fixture" and (search is None or fetcher is None):
        web = _fixture_web(settings)
    llm = llm or build_llm(settings)
    search = search or build_search(settings, web)
    fetcher = fetcher or build_fetcher(settings, web)
    engine = ResearchEngine(store, llm, search, fetcher, settings, progress=progress, **kwargs)
    runner = RunManager(store, settings, engine, **({"clock": clock} if clock else {}))
    return App(settings, store, llm, search, fetcher, engine, runner, runner.watches, runner.reviews)
