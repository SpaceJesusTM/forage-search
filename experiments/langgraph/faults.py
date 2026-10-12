"""Fault injection for the engine comparison: make the process "die" at an exact adapter call.

`SimulatedCrash` derives from BaseException so nothing in the engines catches it, which is
what a killed process looks like to the code: no cleanup, no finalisation, whatever was
committed to SQLite stays, whatever was only in memory is gone.
"""

from __future__ import annotations

from typing import Any, Callable


class SimulatedCrash(BaseException):
    pass


class Faults:
    """Counts adapter calls (search, fetch, model) in order and can crash before or after the n-th one."""

    def __init__(self, harness):
        self.sequence: list[str] = []
        self._armed: tuple[int, str] | None = None
        self._on_crash: Callable[[], None] | None = None
        self._wrap(harness.web, "search", "search")
        self._wrap(harness.web, "fetch", "fetch")
        self._wrap_llm(harness.llm)

    def arm(self, index: int, when: str, on_crash: Callable[[], None] | None = None) -> None:
        """Crash `when` ('before' or 'after') the adapter call with this 0-based position in the session."""
        self._armed, self._on_crash = (index, when), on_crash

    def disarm(self) -> None:
        self._armed = None

    def _maybe(self, position: int, when: str) -> None:
        if self._armed == (position, when):
            self._armed = None
            if self._on_crash:
                self._on_crash()
            raise SimulatedCrash(f"crash {when} adapter call {position} ({self.sequence[position]})")

    def _call(self, kind: str, real: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        position = len(self.sequence)
        self.sequence.append(kind)
        self._maybe(position, "before")
        result = real(*args, **kwargs)
        self._maybe(position, "after")
        return result

    def _wrap(self, target: Any, name: str, kind: str) -> None:
        real = getattr(target, name)
        setattr(target, name, lambda *a, **k: self._call(kind, real, *a, **k))

    def _wrap_llm(self, llm: Any) -> None:
        real = llm.generate
        llm.generate = lambda **k: self._call(f"llm:{k.get('purpose')}", real, **k)
