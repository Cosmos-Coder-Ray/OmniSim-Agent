import hashlib
import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "tavily_cache"
SCHEMA_VERSION = 1


def cache_file(op: str, params: dict[str, Any], *, cache_dir: Path = CACHE_DIR) -> Path:
    payload = json.dumps([op, params], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{op}_{digest}.json"


def read_cache(path: Path) -> Any | None:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(
            "invalid tavily cache file", extra={"path": str(path), "error": type(exc).__name__}
        )
        return None
    if (
        not isinstance(row, dict)
        or row.get("schema_version") != SCHEMA_VERSION
        or "data" not in row
    ):
        log.warning(
            "invalid tavily cache file", extra={"path": str(path), "error": "invalid header"}
        )
        return None
    return row["data"]


def write_cache(path: Path, op: str, params: dict[str, Any], data: Any) -> None:
    row = {
        "op": op,
        "params": params,
        "fetched_at": datetime.now(UTC).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "data": data,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temp_path = Path(stream.name)
            json.dump(row, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
