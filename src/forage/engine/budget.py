"""Per-session budget accounting. Checked before every search, fetch and model call."""

from __future__ import annotations

import time
from typing import Callable

from forage.llm.base import LLMUsage
from forage.types import Budget


class BudgetTracker:
    def __init__(self, budget: Budget, monotonic: Callable[[], float] = time.monotonic):
        self.budget = budget
        self._monotonic = monotonic
        self._started = monotonic()
        self.search_requests = 0
        self.fetched_pages = 0
        self.llm_calls = 0
        self.llm_prompt_tokens = 0
        self.llm_output_tokens = 0
        self.llm_ms = 0
        self.exhausted: list[str] = []

    # -- time -------------------------------------------------------------

    def elapsed(self) -> float:
        return self._monotonic() - self._started

    def remaining_seconds(self) -> float:
        return self.budget.max_duration_seconds - self.elapsed()

    @property
    def wrapup_reserve_seconds(self) -> float:
        """Time held back so the session can still reflect and write its report."""
        return min(120.0, max(20.0, 0.12 * self.budget.max_duration_seconds))

    @property
    def wrapup_reserve_tokens(self) -> int:
        return min(700, self.budget.max_llm_output_tokens // 6)

    def flag(self, name: str) -> None:
        if name not in self.exhausted:
            self.exhausted.append(name)

    def time_left_for_work(self) -> bool:
        if self.remaining_seconds() > self.wrapup_reserve_seconds:
            return True
        self.flag("time")
        return False

    # -- gates ------------------------------------------------------------

    def can_search(self) -> bool:
        if self.search_requests >= self.budget.max_search_requests:
            self.flag("search_requests")
            return False
        return self.time_left_for_work()

    def can_fetch(self) -> bool:
        if self.fetched_pages >= self.budget.max_fetched_pages:
            self.flag("fetched_pages")
            return False
        return self.time_left_for_work()

    def can_llm(self, expected_tokens: int = 0, *, wrapup: bool = False) -> bool:
        """Work-phase calls must leave the wrap-up reserve intact; wrap-up calls may spend it."""
        reserve = 0 if wrapup else self.wrapup_reserve_tokens
        if self.llm_output_tokens + expected_tokens + reserve > self.budget.max_llm_output_tokens:
            self.flag("llm_output_tokens")
            return False
        if wrapup:
            if self.remaining_seconds() <= 0:
                self.flag("time")
                return False
            return True
        return self.time_left_for_work()

    def max_tokens_for_call(self, wanted: int, *, wrapup: bool = False) -> int:
        """Clamp a call's output allowance to what the session can still afford."""
        reserve = 0 if wrapup else self.wrapup_reserve_tokens
        left = self.budget.max_llm_output_tokens - self.llm_output_tokens - reserve
        return max(0, min(wanted, left))

    # -- accounting -------------------------------------------------------

    def note_search(self) -> None:
        self.search_requests += 1

    def note_fetch(self) -> None:
        self.fetched_pages += 1

    def note_llm(self, usage: LLMUsage) -> None:
        self.llm_calls += 1
        self.llm_prompt_tokens += usage.prompt_tokens
        self.llm_output_tokens += usage.output_tokens
        self.llm_ms += usage.duration_ms

    def limits_reached(self) -> list[str]:
        """Budget dimensions that are used up, whether or not a gate was asked after the last unit was spent."""
        reached = list(self.exhausted)
        b = self.budget
        checks = (
            ("search_requests", b.max_search_requests > 0 and self.search_requests >= b.max_search_requests),
            ("fetched_pages", b.max_fetched_pages > 0 and self.fetched_pages >= b.max_fetched_pages),
            ("llm_output_tokens", self.llm_output_tokens + self.wrapup_reserve_tokens >= b.max_llm_output_tokens
             and self.llm_calls > 0),
            ("time", self.remaining_seconds() <= self.wrapup_reserve_seconds),
        )
        for name, hit in checks:
            if hit and name not in reached:
                reached.append(name)
        return reached

    def snapshot(self) -> dict:
        return {
            "search_requests": self.search_requests,
            "fetched_pages": self.fetched_pages,
            "llm_calls": self.llm_calls,
            "llm_prompt_tokens": self.llm_prompt_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "llm_seconds": round(self.llm_ms / 1000, 1),
            "wall_seconds": round(self.elapsed(), 1),
            "budget_exhausted": self.limits_reached(),
        }
