import ipaddress
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from omnisim.audit_log import log_event

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
    "provider_host",
    "call_id",
    "event",
    "provider",
    "process_id",
    "status_code",
    "failure_kind",
)

_call_context: ContextVar[dict[str, Any]] = ContextVar("llm_call_context", default={})


def provider_host(base_url: str | None = None) -> str | None:
    if base_url is None:
        base_url = os.environ.get("NEBIUS_BASE_URL", "")
    try:
        host = urlsplit(base_url).hostname
    except ValueError:
        return None
    if host is None:
        return None
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return host
    return None


def provider_hosts_for_run(run_id: str, *, path: Path = LOG_PATH) -> list[str]:
    hosts: set[str] = set()
    try:
        stream = path.open(encoding="utf-8")
    except FileNotFoundError:
        return []
    with stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or row.get("run_id") != run_id:
                continue
            host = row.get("provider_host")
            if isinstance(host, str):
                normalized = provider_host(host if "://" in host else f"//{host}")
                if normalized:
                    hosts.add(normalized)
    return sorted(hosts)


@contextmanager
def call_context(**values: Any) -> Iterator[None]:
    token = _call_context.set(_call_context.get() | values)
    try:
        yield
    finally:
        _call_context.reset(token)


def write_call_log(entry: dict[str, Any], *, path: Path = LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    context = _call_context.get()
    row = {
        key: entry.get(key) if entry.get(key) is not None else context.get(key) for key in FIELDS
    }
    row["ts"] = row["ts"] or datetime.now(UTC).isoformat()
    row["process_id"] = row["process_id"] or os.getpid()
    if row["attempt"] is None and row["role"] == "models":
        row["attempt"] = 1
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    log_event(
        str(row["event"] or "provider_call"),
        run_id=row["run_id"],
        provider=row["provider"] or "nebius",
        provider_host=row["provider_host"],
        role=row["role"],
        attempt=row["attempt"],
        latency_s=row["latency_s"],
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
        ok=row["ok"],
        error_type=row["error"],
        status_code=row["status_code"],
        failure_kind=row["failure_kind"],
    )
