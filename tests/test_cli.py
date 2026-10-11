"""CLI smoke tests against the SIMULATED web and model (selected through FORAGE_* settings)."""

import json

import pytest

from forage.cli import main

from conftest import CORPUS, OBJECTIVE


@pytest.fixture
def cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FORAGE_SEARCH_PROVIDER", "fixture")
    monkeypatch.setenv("FORAGE_MODEL", "simulated")
    monkeypatch.setenv("FORAGE_FIXTURE_WEB", str(CORPUS))
    monkeypatch.setenv("FORAGE_SEARCH_MIN_INTERVAL_SECONDS", "0")
    monkeypatch.chdir(tmp_path)  # no stray forage.toml is picked up
    data_dir = str(tmp_path / "data")

    def run(*args: str) -> tuple[int, str]:
        code = main(["--data-dir", data_dir, *args])
        captured = capsys.readouterr()
        return code, captured.out + captured.err

    return run


def _watch_id(output: str) -> str:
    return next(line.split()[2] for line in output.splitlines() if line.startswith("Created watch"))


def test_full_cycle_from_the_command_line(cli):
    code, out = cli("watch", "create", OBJECTIVE, "--no-llm", "--question", "What tidal turbine capacity is installed?",
                    "--question", "What does tidal energy cost?")
    assert code == 0 and "PENDING_CONFIRMATION" in out and "Nothing is researched until you confirm" in out
    watch_id = _watch_id(out)

    code, out = cli("run", watch_id, "--quiet")
    assert code == 1 and "only confirmed ACTIVE watches are researched" in out

    assert cli("watch", "approve", watch_id)[0] == 0
    code, out = cli("run", watch_id, "--quiet")
    assert code == 0 and "COMPLETED" in out and "Findings:" in out

    code, out = cli("report", watch_id)
    assert code == 0 and "SIMULATED RUN" in out and "## Findings from sources" in out
    code, out = cli("report", watch_id, "--format", "json")
    report = json.loads(out)
    assert report["schema_version"] == "1.0" and report["session_number"] == 1 and report["findings"]

    code, out = cli("findings", watch_id, "--evidence")
    assert code == 0 and "6 MW" in out and "QUOTE_EXACT" in out and "https://" in out
    assert cli("sources", watch_id)[1].count("FETCHED") >= 3
    assert "[RESOLVED]" in cli("questions", watch_id)[1] or "[OPEN]" in cli("questions", watch_id)[1]
    assert "model calls by purpose" in cli("budget", watch_id)[1]
    assert "session_started" in cli("log", watch_id)[1]

    assert cli("run", watch_id, "--quiet")[0] == 0
    code, out = cli("compare", watch_id)
    assert code == 0 and "session 1 vs session 2" in out and "Searches repeated word for word: 0" in out
    code, out = cli("runs", watch_id)
    assert out.count("COMPLETED") == 2

    code, out = cli("watch", "list")
    assert watch_id in out and "sessions=2" in out


def test_pause_resume_and_close_from_the_command_line(cli):
    watch_id = _watch_id(cli("watch", "create", OBJECTIVE, "--no-llm")[1])
    cli("watch", "approve", watch_id)
    assert "Paused" in cli("watch", "pause", watch_id)[1]
    code, out = cli("run", watch_id, "--quiet")
    assert code == 1 and "PAUSED" in out

    code, out = cli("watch", "resume", watch_id, "--exclude", "wave energy")
    assert code == 0 and "REVIEW_PROPOSED" in out and "wave energy" in out
    assert cli("run", watch_id, "--quiet")[0] == 1  # still not confirmed
    assert "ACTIVE" in cli("watch", "approve", watch_id)[1]
    assert cli("run", watch_id, "--quiet")[0] == 0

    assert "Closed" in cli("watch", "close", watch_id)[1]
    code, out = cli("watch", "show", watch_id, "--history")
    assert "[CLOSED]" in out and "paused" in out and "resume_proposed" in out and "closed" in out


def test_unknown_watch_is_a_clean_error(cli):
    code, out = cli("run", "does-not-exist")
    assert code == 1 and "no watch matches" in out


def test_doctor_reports_simulated_configuration(cli):
    code, out = cli("doctor")
    assert code == 0 and "simulated (no real inference)" in out and "search provider fixture" in out


def test_queue_commands_from_the_command_line(cli, monkeypatch):
    monkeypatch.setenv("FORAGE_MIN_MINUTES_BETWEEN_SESSIONS", "0")
    first = _watch_id(cli("watch", "create", OBJECTIVE, "--no-llm")[1])
    second = _watch_id(cli("watch", "create", "What does tidal energy cost per megawatt hour today?", "--no-llm")[1])
    cli("watch", "approve", first)
    cli("watch", "approve", second)

    code, out = cli("queue", "status")
    status = json.loads(out)
    assert code == 0 and [w["watch_id"] for w in status["next_in_rotation"]] == [first, second]
    assert status["night_window"]["local"] == "02:00-06:00 America/Toronto"

    code, out = cli("queue", "run", "--max-sessions", "3", "--ignore-window", "--quiet")
    assert code == 0 and out.count("COMPLETED") == 3
    lines = [line.split(":")[0] for line in out.splitlines() if "COMPLETED" in line]
    assert lines == [first, second, first]

    monkeypatch.setenv("FORAGE_SCHEDULER_AUTHORITY", "external")
    code, out = cli("queue", "run", "--ignore-window", "--quiet")
    assert code == 1 and "another controller owns scheduling" in out
