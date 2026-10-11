"""SQLite connection handling and numbered migrations."""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Any, Iterator

from forage.util import iso, utcnow

_MIGRATION_RE = re.compile(r"^(\d{4})_.+\.sql$")


def _migration_files() -> list[tuple[int, str, str]]:
    out = []
    root = resources.files("forage.storage") / "migrations"
    for entry in root.iterdir():
        match = _MIGRATION_RE.match(entry.name)
        if match:
            out.append((int(match.group(1)), entry.name, entry.read_text(encoding="utf-8")))
    return sorted(out)


class Database:
    """One SQLite connection in autocommit mode; use `tx()` for multi-statement atomicity."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, isolation_level=None, timeout=10.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA busy_timeout = 10000")
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute("PRAGMA synchronous = NORMAL")
        self._tx_depth = 0
        self.migrate()

    # -- migrations -------------------------------------------------------

    def migrate(self) -> list[str]:
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migration (version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)"
        )
        applied = {row["version"] for row in self.conn.execute("SELECT version FROM schema_migration")}
        done = []
        for version, name, sql in _migration_files():
            if version in applied:
                continue
            # executescript commits implicitly, so the bookkeeping row goes in the same script.
            stamp = iso(utcnow())
            script = (
                "BEGIN;\n" + sql
                + f"\nINSERT INTO schema_migration (version, name, applied_at) VALUES ({version}, '{name}', '{stamp}');\nCOMMIT;"
            )
            try:
                self.conn.executescript(script)
            except sqlite3.Error:
                if self.conn.in_transaction:
                    self.conn.execute("ROLLBACK")
                raise
            done.append(name)
        return done

    def schema_version(self) -> int:
        row = self.conn.execute("SELECT MAX(version) AS v FROM schema_migration").fetchone()
        return int(row["v"] or 0)

    # -- transactions -----------------------------------------------------

    @contextmanager
    def tx(self) -> Iterator[None]:
        """Atomic block. Nested use joins the outer transaction."""
        if self._tx_depth > 0:
            self._tx_depth += 1
            try:
                yield
            finally:
                self._tx_depth -= 1
            return
        self.conn.execute("BEGIN IMMEDIATE")
        self._tx_depth = 1
        try:
            yield
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")
        finally:
            self._tx_depth = 0

    # -- helpers ----------------------------------------------------------

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def one(self, sql: str, params: tuple | dict = ()) -> dict[str, Any] | None:
        row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row is not None else None

    def all(self, sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
        return [dict(row) for row in self.conn.execute(sql, params).fetchall()]

    def scalar(self, sql: str, params: tuple | dict = ()) -> Any:
        row = self.conn.execute(sql, params).fetchone()
        return row[0] if row is not None else None

    def insert(self, table: str, values: dict[str, Any], *, or_ignore: bool = False) -> sqlite3.Cursor:
        cols = ", ".join(values)
        marks = ", ".join(f":{k}" for k in values)
        verb = "INSERT OR IGNORE" if or_ignore else "INSERT"
        return self.conn.execute(f"{verb} INTO {table} ({cols}) VALUES ({marks})", _encode(values))

    def update(self, table: str, key: dict[str, Any], values: dict[str, Any]) -> int:
        if not values:
            return 0
        sets = ", ".join(f"{k} = :v_{k}" for k in values)
        where = " AND ".join(f"{k} = :k_{k}" for k in key)
        params = {f"v_{k}": v for k, v in _encode(values).items()} | {f"k_{k}": v for k, v in key.items()}
        return self.conn.execute(f"UPDATE {table} SET {sets} WHERE {where}", params).rowcount

    def close(self) -> None:
        self.conn.close()


def _encode(values: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in values.items():
        if isinstance(value, (dict, list, tuple)):
            out[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        elif isinstance(value, bool):
            out[key] = int(value)
        else:
            out[key] = value
    return out


def loads(value: str | None, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    return json.loads(value)
