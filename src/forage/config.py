"""Settings: defaults, then an optional TOML file, then FORAGE_* environment variables.

Nothing here is host-specific. The data directory, Ollama address, model and search
provider are all configurable so the same code runs on a laptop or the MSI.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class Settings:
    # Storage
    data_dir: Path = Path("data")

    # Local LLM
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = ""  # empty: pick from installed models at runtime
    max_model_gb: float = 12.0  # upper bound when auto-selecting an installed model
    num_ctx: int = 8192
    temperature: float = 0.1
    llm_timeout_seconds: float = 180.0
    llm_think: bool = False

    # Search
    search_provider: str = "duckduckgo"  # duckduckgo | ddgs | searxng | fixture
    search_min_interval_seconds: float = 12.0
    search_results_per_query: int = 8
    searxng_url: str = ""
    fixture_web: str = ""  # path to a SIMULATED web corpus (search_provider = "fixture")
    fixture_epoch: int = 0

    # Fetch
    user_agent: str = "ForageResearchBot/0.1 (local research agent; respects robots.txt)"
    fetch_timeout_seconds: float = 20.0
    fetch_max_bytes: int = 4_000_000
    fetch_min_interval_per_host_seconds: float = 2.0
    respect_robots: bool = True
    allow_private_addresses: bool = False  # tests only

    # Per-session budget defaults
    max_duration_seconds: int = 900
    max_search_requests: int = 6
    max_fetched_pages: int = 8
    max_llm_output_tokens: int = 9000

    # Engine behaviour
    queries_per_session: int = 4
    fetch_per_query: int = 3
    max_per_domain_per_session: int = 2
    page_chars_per_call: int = 7000
    extract_calls_per_page: int = 1
    max_rechecks_per_session: int = 2
    support_check: bool = True

    # Scheduler
    night_start: str = "02:00"
    night_end: str = "06:00"
    night_timezone: str = "America/Toronto"
    scheduler_authority: str = "standalone"  # standalone | external

    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "forage.sqlite3"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"


def _coerce(current: Any, raw: Any) -> Any:
    if isinstance(current, bool):
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return bool(raw)
    if isinstance(current, Path):
        return Path(str(raw)).expanduser()
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def load_settings(config_path: str | os.PathLike[str] | None = None, env: dict[str, str] | None = None) -> Settings:
    env = dict(os.environ if env is None else env)
    settings = Settings()
    names = {f.name for f in fields(Settings)} - {"extra"}

    path = Path(config_path or env.get("FORAGE_CONFIG") or "forage.toml")
    if path.is_file():
        with path.open("rb") as fh:
            data = tomllib.load(fh)
        for key, value in data.items():
            if key in names:
                setattr(settings, key, _coerce(getattr(settings, key), value))
            else:
                settings.extra[key] = value

    for name in names:
        raw = env.get(f"FORAGE_{name.upper()}")
        if raw is not None and raw != "":
            setattr(settings, name, _coerce(getattr(settings, name), raw))

    settings.data_dir = Path(settings.data_dir).expanduser()
    return settings
