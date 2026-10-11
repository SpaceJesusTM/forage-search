"""Shared test fixtures.

Everything here is SIMULATED: a fixture web corpus, a rule-based stand-in for the
model, and a controllable clock. No test in the default run touches the network
or Ollama. Live checks are in test_live.py behind the `live` marker.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from forage.app import App, build_app
from forage.config import Settings
from forage.simulated import FixtureFetcher, FixtureSearch, SimulatedLLM, SimulatedWeb
from forage.types import WatchSpec

CORPUS = Path(__file__).resolve().parent.parent / "examples" / "simulated_web.json"
OBJECTIVE = "How much tidal turbine energy capacity is installed and what does tidal energy cost?"
QUESTIONS = ["What tidal turbine capacity is installed?", "What does tidal energy cost?"]


class FakeClock:
    def __init__(self, start: datetime | None = None):
        self.now = start or datetime(2026, 10, 12, 6, 30, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


class Harness:
    """A Forage app wired to the simulated web and model, with helpers for common steps."""

    def __init__(self, tmp_path: Path, llm: SimulatedLLM | None = None, **overrides):
        self.tmp_path = tmp_path
        self.web = SimulatedWeb.from_file(CORPUS)
        self.llm = llm or SimulatedLLM()
        self.clock = FakeClock()
        self.progress: list[str] = []
        self.settings = Settings(data_dir=tmp_path / "data", search_provider="fixture", model="simulated",
                                 search_min_interval_seconds=0.0, **overrides)
        self.app = self.open()

    def open(self) -> App:
        self.app = build_app(self.settings, llm=self.llm, search=FixtureSearch(self.web), fetcher=FixtureFetcher(self.web),
                             progress=self.progress.append, clock=self.clock, sleep=lambda s: None)
        return self.app

    def reopen(self) -> App:
        """Simulate a process restart: drop every in-memory object and reopen the database file."""
        self.app.close()
        return self.open()

    def watch(self, objective: str = OBJECTIVE, approve: bool = True, **spec) -> str:
        spec.setdefault("key_questions", QUESTIONS)
        watch = self.app.watches.create(objective, spec=WatchSpec(**spec))
        if approve:
            self.app.watches.approve(watch["id"])
        return watch["id"]

    def run(self, watch_id: str, **overrides):
        return self.app.runner.run_watch(watch_id, overrides=overrides or None)

    def report(self, run_id: str) -> dict:
        import json

        return json.loads(self.app.store.get_report(run_id)["json"])


@pytest.fixture
def harness(tmp_path: Path):
    h = Harness(tmp_path)
    yield h
    h.app.close()


@pytest.fixture
def make_harness(tmp_path: Path):
    created: list[Harness] = []

    def factory(llm: SimulatedLLM | None = None, **overrides) -> Harness:
        h = Harness(tmp_path / f"h{len(created)}", llm, **overrides)
        created.append(h)
        return h

    yield factory
    for h in created:
        h.app.close()
