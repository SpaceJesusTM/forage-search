"""Multi-watch scheduling: night window, round-robin fairness, persistence, backoff, lease. SIMULATED adapters."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from forage.scheduler.queue import Scheduler, SchedulerRefused
from forage.scheduler.window import NightWindow
from forage.simulated import SimulatedLLM
from forage.types import RunStatus, WatchStatus

TORONTO = ZoneInfo("America/Toronto")
WINDOW = NightWindow.from_settings("02:00", "06:00", "America/Toronto")


def toronto(year, month, day, hour, minute=0, fold=0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=TORONTO, fold=fold).astimezone(timezone.utc)


# ---------------------------------------------------------------- window


@pytest.mark.parametrize("hour, minute, expected", [
    (1, 59, False), (2, 0, True), (3, 30, True), (5, 59, True), (6, 0, False), (6, 1, False), (14, 0, False), (23, 59, False),
])
def test_window_boundaries_start_inclusive_end_exclusive(hour, minute, expected):
    assert WINDOW.is_open(toronto(2026, 10, 12, hour, minute)) is expected


def test_window_is_evaluated_in_toronto_time_not_utc():
    assert WINDOW.is_open(datetime(2026, 10, 12, 7, 0, tzinfo=timezone.utc)) is True  # 03:00 EDT
    assert WINDOW.is_open(datetime(2026, 10, 12, 3, 0, tzinfo=timezone.utc)) is False  # 23:00 EDT the day before
    assert WINDOW.is_open(datetime(2026, 12, 12, 7, 0, tzinfo=timezone.utc)) is True  # 02:00 EST in winter
    assert WINDOW.is_open(datetime(2026, 12, 12, 6, 30, tzinfo=timezone.utc)) is False  # 01:30 EST


def test_spring_forward_night_has_a_three_hour_window():
    # 2027-03-14: clocks jump from 02:00 EST to 03:00 EDT, so local 02:00-02:59 never happens.
    before = datetime(2027, 3, 14, 6, 59, tzinfo=timezone.utc)  # 01:59 EST
    jump = datetime(2027, 3, 14, 7, 0, tzinfo=timezone.utc)  # 03:00 EDT
    assert not WINDOW.is_open(before) and WINDOW.is_open(jump)
    assert WINDOW.next_open(before) == jump
    closes = WINDOW.closes_at(jump)
    assert closes == datetime(2027, 3, 14, 10, 0, tzinfo=timezone.utc)  # 06:00 EDT
    assert closes - jump == timedelta(hours=3)
    assert not WINDOW.is_open(closes)


def test_fall_back_night_has_a_four_hour_window_and_no_double_opening():
    # 2026-11-01: 01:00-01:59 happens twice (EDT then EST); the window starts at 02:00 EST.
    first_0130 = toronto(2026, 11, 1, 1, 30, fold=0)
    second_0130 = toronto(2026, 11, 1, 1, 30, fold=1)
    assert second_0130 - first_0130 == timedelta(hours=1)
    assert not WINDOW.is_open(first_0130) and not WINDOW.is_open(second_0130)
    opens = datetime(2026, 11, 1, 7, 0, tzinfo=timezone.utc)  # 02:00 EST
    assert WINDOW.next_open(first_0130) == opens and WINDOW.next_open(second_0130) == opens
    assert WINDOW.is_open(opens) and WINDOW.closes_at(opens) - opens == timedelta(hours=4)


def test_next_open_rolls_to_tomorrow_after_close():
    after_close = toronto(2026, 10, 12, 6, 0)
    assert WINDOW.next_open(after_close) == toronto(2026, 10, 13, 2, 0)
    inside = toronto(2026, 10, 12, 4, 0)
    assert WINDOW.next_open(inside) == inside
    assert WINDOW.closes_at(after_close) is None


def test_window_spanning_midnight():
    late = NightWindow.from_settings("23:00", "03:00", "America/Toronto")
    assert late.is_open(toronto(2026, 10, 12, 23, 30)) and late.is_open(toronto(2026, 10, 13, 1, 0))
    assert not late.is_open(toronto(2026, 10, 12, 22, 59)) and not late.is_open(toronto(2026, 10, 13, 3, 0))
    assert late.closes_at(toronto(2026, 10, 12, 23, 30)) == toronto(2026, 10, 13, 3, 0)


# ----------------------------------------------------------------- queue


def night_harness(make_harness, llm=None, **overrides):
    harness = make_harness(llm, min_minutes_between_sessions=0, **overrides)
    harness.clock.now = toronto(2026, 10, 12, 2, 5)  # inside the night window
    harness.scheduler = Scheduler(harness.app.store, harness.settings, harness.app.runner, clock=harness.clock,
                                  sleep=lambda s: harness.clock.advance(seconds=s), owner="test-scheduler")
    return harness


def three_watches(harness) -> list[str]:
    return [harness.watch(f"Objective {name}: installed tidal turbine capacity and tidal energy cost")
            for name in ("alpha", "beta", "gamma")]


def served(harness) -> list[str]:
    return [row["watch_id"] for row in harness.app.store.all(
        "SELECT watch_id FROM queue_admission WHERE decision = 'ADMITTED' ORDER BY id")]


def test_round_robin_serves_watches_in_order_and_wraps(make_harness):
    harness = night_harness(make_harness)
    a, b, c = three_watches(harness)
    lines = harness.scheduler.run_batch(max_sessions=5)
    assert served(harness) == [a, b, c, a, b]
    assert len(lines) == 5 and all("COMPLETED" in line for line in lines)
    turns = {row["watch_id"]: row["turns"] for row in harness.app.store.all("SELECT * FROM watch_queue")}
    assert turns == {a: 2, b: 2, c: 1}


def test_cursor_survives_restart_and_continues_next_night(make_harness):
    harness = night_harness(make_harness)
    a, b, c = three_watches(harness)
    harness.scheduler.run_batch(max_sessions=2)
    assert served(harness) == [a, b]

    harness.reopen()  # process restart
    harness.clock.now = toronto(2026, 10, 13, 2, 0)  # the following night
    fresh = Scheduler(harness.app.store, harness.settings, harness.app.runner, clock=harness.clock, owner="another-process")
    assert fresh.cursor() == harness.app.store.get_watch(b)["queue_position"]
    fresh.run_batch(max_sessions=2)
    assert served(harness) == [a, b, c, a]  # resumes with the watch that was next, not from the top


def test_paused_closed_unconfirmed_and_reviewing_watches_are_skipped_without_losing_their_place(make_harness):
    harness = night_harness(make_harness)
    a, b, c = three_watches(harness)
    unconfirmed = harness.watch("Objective delta: tidal energy cost", approve=False)
    harness.app.watches.pause(b)
    harness.scheduler.run_batch(max_sessions=3)
    assert served(harness) == [a, c, a]
    skips = harness.app.store.all("SELECT watch_id, reason FROM queue_admission WHERE decision = 'SKIPPED'")
    assert {(s["watch_id"], s["reason"]) for s in skips} >= {(b, "status PAUSED"), (unconfirmed, "status PENDING_CONFIRMATION")}

    harness.app.watches.request_resume(b)
    harness.app.watches.approve(b)
    harness.app.watches.close(c)
    harness.scheduler.run_batch(max_sessions=3)
    assert served(harness) == [a, c, a, b, a, b]  # b rejoins at its own position; closed c is passed over


def test_no_admission_outside_the_window(make_harness):
    harness = night_harness(make_harness)
    three_watches(harness)
    harness.clock.now = toronto(2026, 10, 12, 6, 0)  # exactly 06:00: closed
    lines = harness.scheduler.run_batch(max_sessions=3)
    assert served(harness) == [] and "night window is closed" in lines[0]
    assert harness.web.search_log == []
    assert harness.app.store.scalar("SELECT COUNT(*) FROM queue_admission WHERE decision = 'WINDOW_CLOSED'") >= 1
    # Manual override exists for daytime testing and is explicit.
    harness.scheduler.run_batch(max_sessions=1, respect_window=False)
    assert len(served(harness)) == 1


def test_admission_stops_at_cutoff_but_an_admitted_session_finishes(make_harness):
    harness = night_harness(make_harness)
    a, b, c = three_watches(harness)
    harness.clock.now = toronto(2026, 10, 12, 5, 58)
    original_fetch = harness.web.fetch

    def slow_fetch(url):
        harness.clock.advance(minutes=3)  # each page takes three simulated minutes, crossing 06:00
        return original_fetch(url)

    harness.web.fetch = slow_fetch
    lines = harness.scheduler.run_night(max_sessions=10)
    assert served(harness) == [a]  # admitted at 05:58; nothing new admitted after 06:00
    run = harness.app.store.list_runs(a)[-1]
    assert run["status"] == RunStatus.COMPLETED  # the admitted bounded session was allowed to finish
    assert run["finished_at"] > "2026-10-12T10:00:00Z"  # ...after 06:00 local (10:00 UTC)
    assert any("window closed" in line for line in lines) and "1 session(s) run" in lines[-1]


def test_night_runner_reports_when_window_is_closed_and_can_wait(make_harness):
    harness = night_harness(make_harness)
    a, _, _ = three_watches(harness)
    harness.clock.now = toronto(2026, 10, 12, 1, 0)
    (line,) = harness.scheduler.run_night(max_sessions=1)
    assert "night window is closed" in line and "2026-10-12T06:00:00Z" in line  # 02:00 EDT
    assert served(harness) == []
    lines = harness.scheduler.run_night(max_sessions=1, wait=True)  # sleeps (simulated) until 02:00, then runs
    assert served(harness) == [a] and lines[0].startswith("waiting for the night window")
    assert harness.clock.now >= toronto(2026, 10, 12, 2, 0)


def test_failed_sessions_back_off_and_do_not_starve_others(make_harness):
    from forage.llm.base import LLMUnavailable

    class DownForAlpha(SimulatedLLM):
        def generate(self, **kwargs):
            if "alpha" in kwargs.get("user", ""):
                raise LLMUnavailable("simulated outage")
            return super().generate(**kwargs)

    harness = night_harness(make_harness, DownForAlpha())
    a, b, c = three_watches(harness)
    harness.scheduler.run_batch(max_sessions=5)
    assert served(harness) == [a, b, c, b, c]  # after failing once, alpha is skipped while b and c keep their turns
    row = harness.app.store.one("SELECT * FROM watch_queue WHERE watch_id = ?", (a,))
    assert row["consecutive_failures"] == 1 and row["cooldown_until"] is not None
    assert harness.app.store.list_runs(a)[-1]["status"] == RunStatus.FAILED

    harness.clock.advance(minutes=16)  # first backoff is 15 minutes
    harness.scheduler.run_batch(max_sessions=1)
    assert served(harness)[-1] == a
    row = harness.app.store.one("SELECT * FROM watch_queue WHERE watch_id = ?", (a,))
    assert row["consecutive_failures"] == 2  # failed again: backoff doubles
    assert harness.scheduler.eligibility(harness.app.store.get_watch(a), harness.clock() + timedelta(minutes=20))[0] is False
    assert harness.scheduler.eligibility(harness.app.store.get_watch(a), harness.clock() + timedelta(minutes=31))[0] is True


def test_minimum_gap_prevents_one_watch_from_looping(make_harness):
    harness = night_harness(make_harness)
    harness.settings.min_minutes_between_sessions = 60
    a = harness.watch()
    lines = harness.scheduler.run_night(max_sessions=10)
    # One watch, four-hour window, one-hour gap: sessions at about 02:05, 03:05, 04:05 and 05:05.
    assert served(harness) == [a, a, a, a]
    assert "4 session(s) run" in lines[-1]
    starts = [parse["started_at"] for parse in harness.app.store.list_runs(a)]
    assert starts == sorted(starts) and starts[1] >= "2026-10-12T07:05:00Z"


def test_only_one_scheduler_may_hold_the_queue(make_harness):
    harness = night_harness(make_harness)
    three_watches(harness)
    rival = Scheduler(harness.app.store, harness.settings, harness.app.runner, clock=harness.clock, owner="rival")
    harness.scheduler.acquire_lease(600)
    with pytest.raises(SchedulerRefused, match="another scheduler"):
        rival.run_batch(max_sessions=1)
    assert served(harness) == []
    harness.scheduler.release_lease()
    assert len(rival.run_batch(max_sessions=1)) == 1

    # A crashed holder's lease expires rather than blocking forever.
    harness.scheduler.acquire_lease(600)
    harness.clock.advance(seconds=601)
    assert len(rival.run_batch(max_sessions=1)) == 1


def test_standalone_scheduler_is_disabled_when_another_authority_owns_scheduling(make_harness):
    harness = night_harness(make_harness, scheduler_authority="external")
    a, _, _ = three_watches(harness)
    with pytest.raises(SchedulerRefused, match="another controller owns scheduling"):
        harness.scheduler.run_batch(max_sessions=1)
    with pytest.raises(SchedulerRefused):
        harness.scheduler.run_night(max_sessions=1)
    assert served(harness) == []
    assert harness.run(a).status is RunStatus.COMPLETED  # an explicit single run through the API still works


def test_stop_request_cancels_the_session_and_returns_its_turn(make_harness):
    harness = night_harness(make_harness)
    a, b, _ = three_watches(harness)
    original_fetch = harness.web.fetch

    def fetch_then_stop(url):
        harness.scheduler.request_stop()  # e.g. SIGTERM arrives during the first page
        return original_fetch(url)

    harness.web.fetch = fetch_then_stop
    lines = harness.scheduler.run_batch(max_sessions=3)
    assert harness.app.store.list_runs(a)[-1]["status"] == RunStatus.CANCELLED
    assert len(harness.web.fetch_log) == 1 and len(served(harness)) == 1
    assert any("stop requested" in line for line in lines)
    assert harness.app.store.running_run() is None
    assert harness.app.store.one("SELECT * FROM scheduler_lease WHERE id = 1") is None  # lease released

    harness.web.fetch = original_fetch
    again = Scheduler(harness.app.store, harness.settings, harness.app.runner, clock=harness.clock, owner="next-start")
    again.run_batch(max_sessions=2)
    assert served(harness)[1:] == [a, b]  # the interrupted watch goes first, then rotation continues


def test_user_pause_during_a_queued_session_does_not_penalise_or_requeue_it(make_harness):
    harness = night_harness(make_harness)
    a, b, c = three_watches(harness)
    original_fetch = harness.web.fetch

    def fetch_then_pause(url):
        if harness.app.store.get_watch(a)["status"] == WatchStatus.ACTIVE:
            harness.app.watches.pause(a)
        return original_fetch(url)

    harness.web.fetch = fetch_then_pause
    harness.scheduler.run_batch(max_sessions=3)
    assert harness.app.store.list_runs(a)[-1]["status"] == RunStatus.CANCELLED
    assert served(harness) == [a, b, c]
    row = harness.app.store.one("SELECT * FROM watch_queue WHERE watch_id = ?", (a,))
    assert row["consecutive_failures"] == 0 and row["cooldown_until"] is None
    assert harness.app.store.scalar("SELECT COUNT(*) FROM outbox WHERE watch_id = ?", (a,)) == 0  # nothing to publish


def test_outbox_gets_reports_only_for_meaningful_updates(make_harness):
    harness = night_harness(make_harness)
    a = harness.watch()
    harness.scheduler.run_batch(max_sessions=2)  # second session finds nothing new
    runs = harness.app.store.list_runs(a)
    assert [bool(r["has_meaningful_update"]) for r in runs] == [True, False]
    queued = harness.app.store.all("SELECT run_id, kind FROM outbox ORDER BY id")
    assert queued == [{"run_id": runs[0]["id"], "kind": "REPORT"}]
    assert harness.app.store.get_report(runs[1]["id"]) is not None  # the quiet session is still recorded


def test_status_describes_rotation_window_and_reasons(make_harness):
    harness = night_harness(make_harness)
    a, b, c = three_watches(harness)
    harness.app.watches.pause(b)
    harness.scheduler.run_batch(max_sessions=1)
    status = harness.scheduler.status()
    assert status["night_window"]["open_now"] is True and status["night_window"]["local"] == "02:00-06:00 America/Toronto"
    assert status["cursor"]["last_served"] == a and status["cursor"]["admitted_total"] == 1
    order = [w["watch_id"] for w in status["next_in_rotation"]]
    assert order == [b, c, a]
    reasons = {w["watch_id"]: (w["eligible_now"], w["reason"]) for w in status["next_in_rotation"]}
    assert reasons[b] == (False, "status PAUSED") and reasons[c] == (True, "eligible")
    assert status["scheduler_authority"] == "standalone" and status["running_session"] is None
