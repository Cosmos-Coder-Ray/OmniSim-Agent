from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class JobRecord:
    campaign_id: str
    config: dict[str, Any]
    status: str
    kind: str
    created_at: str
    started_at: str | None
    finished_at: str | None


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


class ApiStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=5) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                    campaign_id TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                )"""
            )

    def create(
        self,
        campaign_id: str,
        config: dict[str, Any],
        *,
        kind: str = "live",
        status: str = "queued",
        created_at: str | None = None,
    ) -> JobRecord:
        created = created_at or _timestamp()
        encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
        with sqlite3.connect(self.path, timeout=5) as db:
            db.execute(
                "INSERT INTO jobs (campaign_id, config_json, status, kind, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (campaign_id, encoded, status, kind, created),
            )
        return JobRecord(campaign_id, config, status, kind, created, None, None)

    def set_status(
        self,
        campaign_id: str,
        status: str,
        *,
        at: str | None = None,
    ) -> None:
        timestamp = at or _timestamp()
        if status == "running":
            query = "UPDATE jobs SET status = ?, started_at = COALESCE(started_at, ?) WHERE campaign_id = ?"
            values = (status, timestamp, campaign_id)
        elif status in {
            "completed",
            "cancelled",
            "error",
            "interrupted",
            "token_cap",
            "no_failures_found",
        }:
            query = "UPDATE jobs SET status = ?, finished_at = ? WHERE campaign_id = ?"
            values = (status, timestamp, campaign_id)
        else:
            query = "UPDATE jobs SET status = ? WHERE campaign_id = ?"
            values = (status, campaign_id)
        with sqlite3.connect(self.path, timeout=5) as db:
            db.execute(query, values)

    def delete(self, campaign_id: str) -> None:
        with sqlite3.connect(self.path, timeout=5) as db:
            db.execute("DELETE FROM jobs WHERE campaign_id = ?", (campaign_id,))

    def get(self, campaign_id: str) -> JobRecord | None:
        with sqlite3.connect(self.path, timeout=5) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM jobs WHERE campaign_id = ?", (campaign_id,)).fetchone()
        return self._record(row) if row is not None else None

    def unfinished(self) -> list[JobRecord]:
        with sqlite3.connect(self.path, timeout=5) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                "SELECT * FROM jobs WHERE status IN ('queued', 'running') ORDER BY created_at"
            ).fetchall()
        return [self._record(row) for row in rows]

    def recent_outcomes(self, since: str) -> list[tuple[str, str]]:
        with sqlite3.connect(self.path, timeout=5) as db:
            rows = db.execute(
                "SELECT status, finished_at FROM jobs WHERE finished_at >= ? ORDER BY finished_at DESC LIMIT 100",
                (since,),
            ).fetchall()
        return [(str(status), str(finished_at)) for status, finished_at in rows]

    @staticmethod
    def _record(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            campaign_id=row["campaign_id"],
            config=json.loads(row["config_json"]),
            status=row["status"],
            kind=row["kind"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )
