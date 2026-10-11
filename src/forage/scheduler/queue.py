"""Multi-watch scheduler (Boundary D): fair round-robin over eligible watches, one session at a time.

Everything here is deterministic. The cursor, per-watch turn counts, failure backoff
and every admission decision are stored, so rotation continues correctly across
nights and restarts. Only one scheduler may hold the lease; nothing here installs
or touches an operating-system scheduler.
"""

from __future__ import annotations

import os
import socket
import time
from datetime import datetime, timedelta
from typing import Any, Callable

from forage.config import Settings
from forage.scheduler.runner import RunManager, RunRefused
from forage.scheduler.window import NightWindow
from forage.storage.store import Store
from forage.types import RunStatus
from forage.util import iso, parse_iso, utcnow

BACKOFF_BASE_MINUTES = 15
BACKOFF_MAX_MINUTES = 6 * 60


class SchedulerRefused(Exception):
    """This process may not schedule: another authority or another scheduler owns the queue."""


class Scheduler:
    def __init__(self, store: Store, settings: Settings, runner: RunManager, *, clock: Callable[[], datetime] = utcnow,
                 sleep: Callable[[float], None] = time.sleep, owner: str | None = None):
        self.store = store
        self.settings = settings
        self.runner = runner
        self.clock = clock
        self.sleep = sleep
        self.owner = owner or f"{socket.gethostname()}:{os.getpid()}"
        self.window = NightWindow.from_settings(settings.night_start, settings.night_end, settings.night_timezone)
        self._stop = False

    # ------------------------------------------------------------- control

    def request_stop(self) -> None:
        """Finish cleanly: no new admissions, and the running session stops at its next checkpoint."""
        self._stop = True

    @property
    def stopping(self) -> bool:
        return self._stop

    # --------------------------------------------------------------- lease

    def _check_authority(self) -> None:
        if self.settings.scheduler_authority != "standalone":
            raise SchedulerRefused(
                f"scheduler_authority is '{self.settings.scheduler_authority}': another controller owns scheduling, "
                "so the standalone queue runner is disabled")

    def acquire_lease(self, ttl_seconds: int) -> None:
        self._check_authority()
        now = self.clock()
        with self.store.tx():
            lease = self.store.one("SELECT * FROM scheduler_lease WHERE id = 1")
            if lease and lease["owner"] != self.owner and parse_iso(lease["expires_at"]) > now:
                raise SchedulerRefused(
                    f"another scheduler ({lease['owner']}) holds the queue until {lease['expires_at']}")
            self.store.execute("DELETE FROM scheduler_lease WHERE id = 1")
            self.store.insert("scheduler_lease", {
                "id": 1, "owner": self.owner, "acquired_at": lease["acquired_at"] if lease and lease["owner"] == self.owner else iso(now),
                "heartbeat_at": iso(now), "expires_at": iso(now + timedelta(seconds=ttl_seconds))})

    def release_lease(self) -> None:
        self.store.execute("DELETE FROM scheduler_lease WHERE id = 1 AND owner = ?", (self.owner,))

    def _lease_ttl(self) -> int:
        return int(self.settings.max_duration_seconds * 2 + 600)

    # ---------------------------------------------------------- eligibility

    def _queue_row(self, watch_id: str) -> dict[str, Any]:
        row = self.store.one("SELECT * FROM watch_queue WHERE watch_id = ?", (watch_id,))
        if row is None:
            self.store.insert("watch_queue", {"watch_id": watch_id}, or_ignore=True)
            row = self.store.one("SELECT * FROM watch_queue WHERE watch_id = ?", (watch_id,))
        return row  # type: ignore[return-value]

    def eligibility(self, watch: dict[str, Any], now: datetime) -> tuple[bool, str, datetime | None]:
        """(eligible, reason, eligible_at). `eligible_at` is set when only time stands in the way."""
        if not self.runner.watches.is_researchable(watch):
            return False, f"status {watch['status']}", None
        row = self._queue_row(watch["id"])
        if row["cooldown_until"] and parse_iso(row["cooldown_until"]) > now:
            return False, f"backing off after {row['consecutive_failures']} failure(s) until {row['cooldown_until']}", \
                parse_iso(row["cooldown_until"])
        gap = timedelta(minutes=self.settings.min_minutes_between_sessions)
        if row["last_admitted_at"] and gap.total_seconds() > 0:
            ready = parse_iso(row["last_admitted_at"]) + gap
            if ready > now:
                return False, f"ran recently; next turn after {iso(ready)}", ready
        return True, "eligible", None

    def cursor(self) -> int:
        row = self.store.one("SELECT cursor_position FROM queue_state WHERE id = 1")
        return int(row["cursor_position"]) if row else 0

    def rotation(self) -> list[dict[str, Any]]:
        """All watches in service order, starting just after the cursor and wrapping round."""
        watches = self.store.list_watches()
        cursor = self.cursor()
        return [w for w in watches if w["queue_position"] > cursor] + [w for w in watches if w["queue_position"] <= cursor]

    def pick_next(self, now: datetime) -> tuple[dict[str, Any] | None, list[tuple[str, str]]]:
        """The next eligible watch in rotation, plus the watches passed over and why."""
        skipped: list[tuple[str, str]] = []
        for watch in self.rotation():
            eligible, reason, _ = self.eligibility(watch, now)
            if eligible:
                return watch, skipped
            skipped.append((watch["id"], reason))
        return None, skipped

    def next_eligible_at(self, now: datetime) -> datetime | None:
        times = [at for watch in self.store.list_watches() if (at := self.eligibility(watch, now)[2]) is not None]
        return min(times) if times else None

    # ------------------------------------------------------------- logging

    def _log(self, decision: str, trigger: str, watch_id: str | None = None, run_id: str | None = None,
             reason: str | None = None) -> None:
        self.store.insert("queue_admission", {"at": iso(self.clock()), "watch_id": watch_id, "run_id": run_id,
                                              "decision": decision, "reason": reason, "trigger": trigger})

    # ----------------------------------------------------------- execution

    def run_next(self, *, trigger: str = "queue", respect_window: bool = True) -> dict[str, Any] | None:
        """Admit and run one session for the next watch in rotation. Returns a summary, or None if nothing ran."""
        now = self.clock()
        if respect_window and not self.window.is_open(now):
            self._log("WINDOW_CLOSED", trigger, reason=f"local time {self.window.local(now).strftime('%H:%M %Z')}")
            return None
        self.runner.recover()
        watch, skipped = self.pick_next(now)
        if watch is None:
            self._log("NONE_ELIGIBLE", trigger, reason="; ".join(f"{w}: {r}" for w, r in skipped)[:500] or "no watches")
            return None
        previous_cursor = self.cursor()
        try:
            with self.store.tx():
                run = self.runner.admit(watch["id"], trigger)
                # Cursor and admission commit together: a crash cannot serve a watch twice or skip one.
                self.store.execute("DELETE FROM queue_state WHERE id = 1")
                total = int(self.store.scalar("SELECT COUNT(*) FROM queue_admission WHERE decision = 'ADMITTED'") or 0) + 1
                self.store.insert("queue_state", {"id": 1, "cursor_position": watch["queue_position"],
                                                  "cursor_watch_id": watch["id"], "admitted_total": total,
                                                  "updated_at": iso(now)})
                self._queue_row(watch["id"])
                self.store.execute(
                    "UPDATE watch_queue SET turns = turns + 1, last_admitted_at = ?, last_run_id = ? WHERE watch_id = ?",
                    (iso(now), run["id"], watch["id"]))
                for watch_id, reason in skipped:
                    self._log("SKIPPED", trigger, watch_id, reason=reason)
                self._log("ADMITTED", trigger, watch["id"], run["id"])
        except RunRefused as exc:
            self._log("REFUSED", trigger, watch["id"], reason=str(exc)[:300])
            return None

        outcome = self.runner.execute_admitted(run["id"], should_stop=lambda: self._stop)
        self._after_run(watch["id"], outcome.status, previous_cursor)
        return {"watch_id": watch["id"], "run_id": outcome.run_id, "status": str(outcome.status),
                "meaningful": outcome.has_meaningful_update, "error": outcome.error}

    def _after_run(self, watch_id: str, status: RunStatus, previous_cursor: int) -> None:
        now = self.clock()
        if status is RunStatus.COMPLETED:
            self.store.execute("UPDATE watch_queue SET consecutive_failures = 0, cooldown_until = NULL WHERE watch_id = ?",
                               (watch_id,))
        elif status in (RunStatus.FAILED, RunStatus.INTERRUPTED):
            failures = int(self._queue_row(watch_id)["consecutive_failures"]) + 1
            minutes = min(BACKOFF_BASE_MINUTES * 2 ** (failures - 1), BACKOFF_MAX_MINUTES)
            self.store.execute("UPDATE watch_queue SET consecutive_failures = ?, cooldown_until = ? WHERE watch_id = ?",
                               (failures, iso(now + timedelta(minutes=minutes)), watch_id))
        elif status is RunStatus.CANCELLED and self._stop:
            # Our own shutdown cut the turn short: give it back so the watch is first in line next time.
            self.store.execute("UPDATE queue_state SET cursor_position = ?, updated_at = ? WHERE id = 1",
                               (previous_cursor, iso(now)))
            self.store.execute("UPDATE watch_queue SET turns = turns - 1, last_admitted_at = NULL WHERE watch_id = ?",
                               (watch_id,))

    def run_batch(self, *, max_sessions: int = 1, respect_window: bool = True, trigger: str = "queue") -> list[str]:
        """Run up to `max_sessions` sessions back to back, then stop. Returns human-readable lines."""
        lines: list[str] = []
        self.acquire_lease(self._lease_ttl())
        try:
            for _ in range(max(0, max_sessions)):
                if self._stop:
                    lines.append("stop requested; no further sessions admitted")
                    break
                self.acquire_lease(self._lease_ttl())  # renew
                now = self.clock()
                if respect_window and not self.window.is_open(now):
                    lines.append(f"night window is closed (local time {self.window.local(now).strftime('%H:%M %Z')}); "
                                 f"next opens {iso(self.window.next_open(now))}")
                    self._log("WINDOW_CLOSED", trigger)
                    break
                result = self.run_next(trigger=trigger, respect_window=respect_window)
                if result is None:
                    lines.append("no eligible watch right now")
                    break
                lines.append(f"{result['watch_id']}: run {result['run_id']} {result['status']}"
                             + (" (meaningful update)" if result["meaningful"] else "")
                             + (f" — {result['error']}" if result["error"] else ""))
        finally:
            self.release_lease()
        return lines

    def run_night(self, *, max_sessions: int = 12, wait: bool = False, poll_seconds: float = 60.0) -> list[str]:
        """Foreground runner for one night window. Admits nothing at or after the window's end."""
        lines: list[str] = []
        self.acquire_lease(self._lease_ttl())
        try:
            now = self.clock()
            if not self.window.is_open(now):
                opens = self.window.next_open(now)
                if not wait:
                    return [f"night window is closed; it next opens {iso(opens)} "
                            f"({self.window.local(opens).strftime('%Y-%m-%d %H:%M %Z')})"]
                lines.append(f"waiting for the night window to open at {iso(opens)}")
                while self.clock() < opens and not self._stop:
                    self.acquire_lease(self._lease_ttl())
                    self.sleep(min(poll_seconds, max(1.0, (opens - self.clock()).total_seconds())))
            sessions = 0
            while sessions < max_sessions and not self._stop:
                self.acquire_lease(self._lease_ttl())
                now = self.clock()
                if not self.window.is_open(now):
                    lines.append(f"window closed at {iso(now)}; no further sessions admitted")
                    break
                result = self.run_next(trigger="night", respect_window=True)
                if result is not None:
                    sessions += 1
                    lines.append(f"{result['watch_id']}: run {result['run_id']} {result['status']}"
                                 + (" (meaningful update)" if result["meaningful"] else ""))
                    continue
                closes = self.window.closes_at(self.clock())
                ready = self.next_eligible_at(self.clock())
                if closes is None or ready is None or ready >= closes:
                    lines.append("no watch will become eligible before the window closes")
                    break
                self.sleep(min(poll_seconds, max(1.0, (ready - self.clock()).total_seconds())))
            if self._stop:
                lines.append("stopped on request")
            lines.append(f"{sessions} session(s) run")
        finally:
            self.release_lease()
        return lines

    # -------------------------------------------------------------- status

    def status(self) -> dict[str, Any]:
        now = self.clock()
        lease = self.store.one("SELECT * FROM scheduler_lease WHERE id = 1")
        state = self.store.one("SELECT * FROM queue_state WHERE id = 1") or {}
        watches = []
        for watch in self.rotation():
            eligible, reason, _ = self.eligibility(watch, now)
            row = self._queue_row(watch["id"])
            watches.append({
                "watch_id": watch["id"], "status": watch["status"], "policy": watch["checkpoint_policy"],
                "queue_position": watch["queue_position"], "eligible_now": eligible, "reason": reason,
                "turns": row["turns"], "last_admitted_at": row["last_admitted_at"],
                "consecutive_failures": row["consecutive_failures"], "cooldown_until": row["cooldown_until"],
            })
        return {
            "now_utc": iso(now),
            "scheduler_authority": self.settings.scheduler_authority,
            "night_window": {
                "local": f"{self.settings.night_start}-{self.settings.night_end} {self.settings.night_timezone}",
                "local_time_now": self.window.local(now).strftime("%Y-%m-%d %H:%M %Z"),
                "open_now": self.window.is_open(now),
                "next_open_utc": iso(self.window.next_open(now)),
                "closes_utc": iso(self.window.closes_at(now)) if self.window.is_open(now) else None,
            },
            "cursor": {"position": state.get("cursor_position", 0), "last_served": state.get("cursor_watch_id"),
                       "admitted_total": state.get("admitted_total", 0)},
            "lease": lease,
            "running_session": (self.store.running_run() or {}).get("id"),
            "next_in_rotation": watches,
        }
