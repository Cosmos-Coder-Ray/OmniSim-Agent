import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from omnisim.config import Settings

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "llm_usage.sqlite3"


class LLMError(RuntimeError):
    pass


class BudgetExceeded(LLMError):
    pass


class LLMUnavailable(LLMError):
    pass


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _day(now: datetime | None) -> str:
    current = now or datetime.now(UTC)
    return current.astimezone(UTC).date().isoformat()


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=30)
    connection.execute(
        """CREATE TABLE IF NOT EXISTS usage (
            day TEXT NOT NULL,
            role TEXT NOT NULL,
            calls INTEGER NOT NULL DEFAULT 0,
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (day, role)
        )"""
    )
    connection.commit()
    return connection


@contextmanager
def request_slot(
    role: str,
    *,
    cap: int | None = None,
    db_path: Path = DB_PATH,
    now: datetime | None = None,
) -> Iterator[sqlite3.Connection]:
    connection = _connect(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        check(role, cap=cap, db_path=db_path, now=now, connection=connection)
        yield connection
        connection.commit()
    finally:
        if connection.in_transaction:
            connection.rollback()
        connection.close()


def _total(connection: sqlite3.Connection, day: str) -> int:
    return connection.execute(
        "SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0) FROM usage WHERE day = ?",
        (day,),
    ).fetchone()[0]


def _record(
    connection: sqlite3.Connection,
    role: str,
    prompt_tokens: int,
    completion_tokens: int,
    day: str,
) -> None:
    connection.execute(
        """INSERT INTO usage (day, role, calls, prompt_tokens, completion_tokens)
        VALUES (?, ?, 1, ?, ?)
        ON CONFLICT(day, role) DO UPDATE SET
            calls = calls + 1,
            prompt_tokens = prompt_tokens + excluded.prompt_tokens,
            completion_tokens = completion_tokens + excluded.completion_tokens""",
        (day, role, prompt_tokens, completion_tokens),
    )


def check(
    role: str,
    *,
    cap: int | None = None,
    db_path: Path = DB_PATH,
    now: datetime | None = None,
    connection: sqlite3.Connection | None = None,
) -> None:
    limit = Settings.from_env().OMNISIM_DAILY_TOKEN_CAP if cap is None else cap
    if connection is None:
        with closing(_connect(db_path)) as db, db:
            total = _total(db, _day(now))
    else:
        total = _total(connection, _day(now))
    if total >= limit:
        raise BudgetExceeded(f"daily token cap reached ({total}/{limit})")


def record(
    role: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    db_path: Path = DB_PATH,
    now: datetime | None = None,
    connection: sqlite3.Connection | None = None,
) -> None:
    if connection is None:
        with closing(_connect(db_path)) as db, db:
            _record(db, role, prompt_tokens, completion_tokens, _day(now))
    else:
        _record(connection, role, prompt_tokens, completion_tokens, _day(now))


def get_usage_summary(
    *,
    cap: int | None = None,
    db_path: Path = DB_PATH,
    now: datetime | None = None,
) -> dict:
    limit = Settings.from_env().OMNISIM_DAILY_TOKEN_CAP if cap is None else cap
    day = _day(now)
    with closing(_connect(db_path)) as connection, connection:
        rows = connection.execute(
            "SELECT role, calls, prompt_tokens, completion_tokens FROM usage WHERE day = ?",
            (day,),
        ).fetchall()
    by_role = {
        role: {
            "calls": calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        for role, calls, prompt_tokens, completion_tokens in rows
    }
    calls = sum(item["calls"] for item in by_role.values())
    prompt_tokens = sum(item["prompt_tokens"] for item in by_role.values())
    completion_tokens = sum(item["completion_tokens"] for item in by_role.values())
    total = prompt_tokens + completion_tokens
    return {
        "day": day,
        "calls": calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total": total,
        "cap": limit,
        "remaining": max(0, limit - total),
        "by_role": by_role,
    }
