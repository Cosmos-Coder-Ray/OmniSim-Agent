from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from omnisim.llm import models, nebius
from omnisim.llm.budget import Usage
from omnisim.llm.call_log import provider_host
from omnisim.llm.models import Role

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
ChatJSON = Callable[..., tuple[Any, Usage]]
ModelResolver = Callable[[Role], str]


class ReplayCacheError(RuntimeError):
    pass


def prompt_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_model(role: Role) -> str:
    return models.resolve_model(nebius.get_client(), role)


def _read_entry(path: Path) -> dict[str, Any] | None:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(
            "invalid benchmark cache entry", extra={"path": str(path), "error": type(exc).__name__}
        )
        return None
    if (
        not isinstance(row, dict)
        or not isinstance(row.get("resolved_model_id"), str)
        or not isinstance(row.get("salt"), str)
        or not isinstance(row.get("prompt_hash"), str)
        or not isinstance(row.get("parsed"), dict)
        or not isinstance(row.get("usage"), dict)
    ):
        log.warning(
            "invalid benchmark cache entry", extra={"path": str(path), "error": "invalid header"}
        )
        return None
    return row


def _atomic_write(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temp_path = Path(stream.name)
            json.dump(row, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


class BenchmarkLLMCache:
    def __init__(
        self,
        cache_dir: Path,
        *,
        salt: str,
        prompt_hash: str,
        replay_only: bool = False,
        chat_json: ChatJSON | None = None,
        model_resolver: ModelResolver | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.salt = salt
        self.prompt_hash = prompt_hash
        self.replay_only = replay_only
        self._chat_json = chat_json or nebius.chat_json
        self._model_resolver = model_resolver or _resolve_model
        self.last_model_id: str | None = None
        self.last_provider_host: str | None = None
        self.last_usage = Usage()
        self.last_cache_hit = False

    def _request_digest(
        self,
        role: Role,
        system: str,
        user: str,
        schema: type[BaseModel],
        temperature: float,
        max_tokens: int,
    ) -> str:
        return _digest(
            [
                self.salt,
                self.prompt_hash,
                role,
                system,
                user,
                schema.__module__,
                schema.__qualname__,
                temperature,
                max_tokens,
            ]
        )

    def cache_path(
        self,
        role: Role,
        system: str,
        user: str,
        schema: type[BaseModel],
        temperature: float,
        max_tokens: int,
        model_id: str,
    ) -> Path:
        request = self._request_digest(role, system, user, schema, temperature, max_tokens)
        model_digest = _digest(model_id)[:16]
        return self.cache_dir / f"{request}_{model_digest}.json"

    def _load(
        self,
        request_digest: str,
        schema: type[T],
    ) -> tuple[T, Usage, str, str | None] | None:
        matching_models: set[str] = set()
        for path in self.cache_dir.glob("*.json") if self.cache_dir.exists() else ():
            row = _read_entry(path)
            if row and row.get("salt") == self.salt and row.get("prompt_hash") == self.prompt_hash:
                matching_models.add(row["resolved_model_id"])
        if len(matching_models) > 1:
            raise ReplayCacheError(
                "replay cache has multiple model IDs for this salt and prompt hash"
            )

        if not matching_models:
            return None
        model_id = next(iter(matching_models))
        path = self.cache_dir / f"{request_digest}_{_digest(model_id)[:16]}.json"
        row = _read_entry(path)
        if not row or row.get("salt") != self.salt or row.get("prompt_hash") != self.prompt_hash:
            return None
        try:
            parsed = schema.model_validate(row["parsed"])
            usage = Usage(
                prompt_tokens=int(row["usage"]["prompt_tokens"]),
                completion_tokens=int(row["usage"]["completion_tokens"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            log.warning(
                "invalid benchmark cache entry",
                extra={"path": str(path), "error": type(exc).__name__},
            )
            return None
        host = row.get("provider_host")
        return parsed, usage, model_id, host if isinstance(host, str) else None

    def __call__(
        self,
        role: Role,
        system: str,
        user: str,
        schema: type[T],
        *,
        temperature: float,
        max_tokens: int,
        run_id: str | None = None,
    ) -> tuple[T, Usage]:
        request_digest = self._request_digest(role, system, user, schema, temperature, max_tokens)
        if self.replay_only:
            cached = self._load(request_digest, schema)
            if cached is None:
                raise ReplayCacheError("replay cache miss for this benchmark request")
            parsed, usage, model_id, host = cached
            self.last_model_id = model_id
            self.last_provider_host = host
            self.last_usage = usage
            self.last_cache_hit = True
            return parsed, usage

        model_id = self._model_resolver(role)
        path = self.cache_path(role, system, user, schema, temperature, max_tokens, model_id)
        row = _read_entry(path)
        if (
            row
            and row.get("resolved_model_id") == model_id
            and row.get("salt") == self.salt
            and row.get("prompt_hash") == self.prompt_hash
        ):
            try:
                parsed = schema.model_validate(row["parsed"])
                usage = Usage(
                    prompt_tokens=int(row["usage"]["prompt_tokens"]),
                    completion_tokens=int(row["usage"]["completion_tokens"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                log.warning(
                    "invalid benchmark cache entry",
                    extra={"path": str(path), "error": type(exc).__name__},
                )
            else:
                self.last_model_id = model_id
                host = row.get("provider_host")
                self.last_provider_host = host if isinstance(host, str) else None
                self.last_usage = usage
                self.last_cache_hit = True
                return parsed, usage

        parsed, usage = self._chat_json(
            role,
            system,
            user,
            schema,
            temperature=temperature,
            max_tokens=max_tokens,
            run_id=run_id,
        )
        self.last_model_id = model_id
        self.last_provider_host = provider_host()
        self.last_usage = usage
        self.last_cache_hit = False
        _atomic_write(
            path,
            {
                "resolved_model_id": model_id,
                "provider_host": self.last_provider_host,
                "salt": self.salt,
                "prompt_hash": self.prompt_hash,
                "parsed": parsed.model_dump(mode="json"),
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                },
            },
        )
        return parsed, usage


def cache_size(cache_dir: Path) -> int:
    return sum(path.stat().st_size for path in cache_dir.glob("*.json") if path.is_file())
