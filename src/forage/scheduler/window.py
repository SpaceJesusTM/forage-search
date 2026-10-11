"""The nightly admission window. Pure time arithmetic, no I/O.

Admission is decided on local wall-clock time in the configured zone, so daylight
saving changes need no special cases: on the spring-forward night local 02:00-02:59
does not exist and the window is three hours long; on the fall-back night it is four.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


def _parse_clock(text: str) -> time:
    hour, _, minute = text.strip().partition(":")
    return time(int(hour), int(minute or 0))


@dataclass(frozen=True)
class NightWindow:
    start: time
    end: time
    tz: ZoneInfo

    @classmethod
    def from_settings(cls, start: str, end: str, tz_name: str) -> "NightWindow":
        window = cls(_parse_clock(start), _parse_clock(end), ZoneInfo(tz_name))
        if window.start == window.end:
            raise ValueError("night window start and end must differ")
        return window

    def local(self, now: datetime) -> datetime:
        return now.astimezone(self.tz)

    def is_open(self, now: datetime) -> bool:
        """True from `start` inclusive to `end` exclusive, local time. A window may span midnight."""
        clock = self.local(now).time().replace(tzinfo=None)
        if self.start < self.end:
            return self.start <= clock < self.end
        return clock >= self.start or clock < self.end

    def _at(self, day, clock: time) -> datetime:
        # A wall time skipped by daylight saving resolves to the instant the clocks jump.
        return datetime.combine(day, clock, tzinfo=self.tz).astimezone(timezone.utc)

    def next_open(self, now: datetime) -> datetime:
        """The current instant if the window is open, otherwise the next opening (UTC)."""
        if self.is_open(now):
            return now.astimezone(timezone.utc)
        local = self.local(now)
        candidate = self._at(local.date(), self.start)
        if candidate <= now:
            candidate = self._at(local.date() + timedelta(days=1), self.start)
        return candidate

    def closes_at(self, now: datetime) -> datetime | None:
        """When the window that is currently open ends (UTC); None if it is closed."""
        if not self.is_open(now):
            return None
        local = self.local(now)
        end = self._at(local.date(), self.end)
        if end <= now:  # window spans midnight and we are in its evening part
            end = self._at(local.date() + timedelta(days=1), self.end)
        return end
