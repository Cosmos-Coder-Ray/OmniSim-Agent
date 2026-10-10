import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOG_PATH = Path(__file__).resolve().parents[3] / "data" / "llm_calls.jsonl"
FIELDS = (
    "ts",
    "run_id",
    "role",
    "model",
    "attempt",
    "latency_s",
    "prompt_tokens",
    "completion_tokens",
    "finish_reason",
    "ok",
    "error",
)


def write_call_log(entry: dict[str, Any], *, path: Path = LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {key: entry.get(key) for key in FIELDS}
    row["ts"] = row["ts"] or datetime.now(UTC).isoformat()
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
