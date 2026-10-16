import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

CALL_LOG_PATH = Path(__file__).resolve().parents[3] / "data" / "knowledge_calls.jsonl"


def _utc_date(now: datetime | None) -> date:
    return (now or datetime.now(UTC)).astimezone(UTC).date()


def log_call(entry: dict[str, Any], *, path: Path = CALL_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        key: entry.get(key)
        for key in (
            "ts",
            "run_id",
            "op",
            "cache",
            "latency_s",
            "n_results",
            "ok",
            "error",
        )
    }
    row["ts"] = row["ts"] or datetime.now(UTC).isoformat()
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_knowledge_usage(
    *, path: Path = CALL_LOG_PATH, now: datetime | None = None
) -> dict[str, int | str]:
    today = _utc_date(now)
    usage: dict[str, int | str] = {
        "date": today.isoformat(),
        "searches": 0,
        "extracts": 0,
        "cache_hits": 0,
        "cache_misses": 0,
    }
    try:
        stream = path.open(encoding="utf-8")
    except FileNotFoundError:
        return usage
    with stream:
        for line in stream:
            try:
                row = json.loads(line)
                ts = datetime.fromisoformat(row["ts"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if ts.astimezone(UTC).date() != today:
                continue
            if row.get("cache") == "hit":
                usage["cache_hits"] = int(usage["cache_hits"]) + 1
            elif row.get("cache") == "miss":
                usage["cache_misses"] = int(usage["cache_misses"]) + 1
                op = row.get("op")
                if op == "search":
                    usage["searches"] = int(usage["searches"]) + 1
                elif op == "extract":
                    usage["extracts"] = int(usage["extracts"]) + 1
    return usage
