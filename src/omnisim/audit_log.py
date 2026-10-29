from __future__ import annotations

import ipaddress
import json
import logging
import math
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_CONTEXT_FIELDS = ("request_id", "run_id", "campaign_id", "stage")
_context: dict[str, ContextVar[str | None]] = {
    name: ContextVar(f"omnisim_{name}", default=None) for name in _CONTEXT_FIELDS
}
_STRING_FIELDS = {
    "event",
    "provider",
    "provider_host",
    "role",
    "status",
    "cache",
    "failure_kind",
    "error_type",
    "operation",
    "profile",
    "method",
}
_NUMBER_FIELDS = {
    "attempt",
    "status_code",
    "latency_s",
    "prompt_tokens",
    "completion_tokens",
    "process_id",
    "queued",
    "running",
}
_BOOL_FIELDS = {"ok", "forced"}
_SAFE_TEXT = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_REQUEST_ID = re.compile(r"^[0-9a-f]{32}$")


def _safe_text(value: object) -> str | None:
    if not isinstance(value, str) or not _SAFE_TEXT.fullmatch(value):
        return None
    return value


def _safe_host(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        host = urlsplit(value if "://" in value else f"//{value}").hostname
    except ValueError:
        return None
    if host is None or not re.fullmatch(r"[A-Za-z0-9.-]{1,253}", host):
        return None
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return host.lower()
    return None


def _safe_field(name: str, value: object) -> Any:
    if name == "provider_host":
        return _safe_host(value)
    if name in _CONTEXT_FIELDS:
        if name == "request_id":
            return value if isinstance(value, str) and _REQUEST_ID.fullmatch(value) else None
        return value if isinstance(value, str) and _SAFE_ID.fullmatch(value) else None
    if name in _STRING_FIELDS:
        return _safe_text(value)
    if name in _NUMBER_FIELDS:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if name in _BOOL_FIELDS:
        return value if isinstance(value, bool) else None
    return None


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        row: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname.lower(),
        }
        event = _safe_text(getattr(record, "event", None))
        row["event"] = event or "log_record"
        for name, variable in _context.items():
            value = _safe_field(name, variable.get())
            if value is not None:
                row[name] = value
        for name in set(_CONTEXT_FIELDS) | _STRING_FIELDS | _NUMBER_FIELDS | _BOOL_FIELDS:
            value = _safe_field(name, getattr(record, name, None))
            if value is not None:
                row[name] = value
        if record.exc_info and record.exc_info[0] is not None:
            error_type = _safe_text(record.exc_info[0].__name__)
            if error_type is not None:
                row["error_type"] = error_type
        return json.dumps(row, separators=(",", ":"), ensure_ascii=True)


def bind_log_context(**values: str | None) -> dict[str, Token[str | None]]:
    tokens: dict[str, Token[str | None]] = {}
    for name, value in values.items():
        if name in _context:
            tokens[name] = _context[name].set(value)
    return tokens


def set_log_context(**values: str | None) -> None:
    for name, value in values.items():
        if name in _context:
            _context[name].set(value)


def reset_log_context(tokens: dict[str, Token[str | None]]) -> None:
    for name, token in reversed(list(tokens.items())):
        _context[name].reset(token)


@contextmanager
def log_context(**values: str | None) -> Iterator[None]:
    tokens = bind_log_context(**values)
    try:
        yield
    finally:
        reset_log_context(tokens)


def log_event(event: str, **fields: object) -> None:
    logging.getLogger("omnisim.audit").info(
        "",
        extra={"event": event, **fields},
    )


def configure_logging(*, log_path: Path | None = None, **context: str | None) -> None:
    logger = logging.getLogger("omnisim")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        if getattr(handler, "_omnisim_console", False):
            logger.removeHandler(handler)
            handler.close()
    if log_path is None:
        handler = logging.StreamHandler(sys.stderr)
        handler._omnisim_console = True  # type: ignore[attr-defined]
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = add_audit_file_handler(log_path)
        if handler is None:
            return
    bind_log_context(**context)


def add_audit_file_handler(path: Path) -> logging.Handler | None:
    logger = logging.getLogger("omnisim")
    resolved = path.resolve()
    for handler in logger.handlers:
        if getattr(handler, "_omnisim_audit_path", None) == resolved:
            return None
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler._omnisim_audit_path = resolved  # type: ignore[attr-defined]
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return handler


def remove_audit_file_handler(handler: logging.Handler | None) -> None:
    if handler is None:
        return
    logger = logging.getLogger("omnisim")
    logger.removeHandler(handler)
    handler.close()
