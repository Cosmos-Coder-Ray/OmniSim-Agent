import logging
from functools import lru_cache
from time import perf_counter
from typing import Any, Literal

import openai

from omnisim.config import Settings
from omnisim.llm.budget import LLMUnavailable
from omnisim.llm.call_log import provider_host, write_call_log

log = logging.getLogger(__name__)
Role = Literal["fast", "deep", "ultra"]


def _is_nemotron(model: str) -> bool:
    model_id = model.lower()
    return model_id.startswith("nvidia/") and "nemotron" in model_id


@lru_cache(maxsize=32)
def _list_models(client: Any) -> tuple[str, ...]:
    started = perf_counter()
    try:
        models = tuple(sorted(model.id for model in client.models.list()))
    except openai.OpenAIError as exc:
        status = exc.status_code if isinstance(exc, openai.APIStatusError) else None
        retryable = isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)) or (
            status == 429 or (status is not None and status >= 500)
        )
        write_call_log(
            {
                "role": "models",
                "latency_s": perf_counter() - started,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "ok": False,
                "error": type(exc).__name__,
                "provider_host": provider_host(),
                "status_code": status,
                "failure_kind": "transport" if retryable else "request",
            }
        )
        log.warning(
            "model listing failed",
            extra={"error": type(exc).__name__, "status": status},
        )
        raise
    write_call_log(
        {
            "role": "models",
            "latency_s": perf_counter() - started,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "ok": True,
            "provider_host": provider_host(),
        }
    )
    return models


def list_models(client: Any) -> list[str]:
    return list(_list_models(client))


def resolve_model(client: Any, role: Role, *, settings: Settings | None = None) -> str:
    settings = settings or Settings.from_env()
    configured = {
        "fast": settings.NEBIUS_FAST_MODEL,
        "deep": settings.NEBIUS_DEEP_MODEL,
        "ultra": settings.NEBIUS_ULTRA_MODEL,
    }[role]
    available = list_models(client)
    if configured and configured in available and _is_nemotron(configured):
        return configured

    if configured:
        reason = "configured model is unavailable"
        if configured in available:
            reason = "configured model is not an NVIDIA Nemotron model"
        log.warning(reason, extra={"role": role, "model": configured})
    else:
        log.warning("configured model is empty", extra={"role": role})

    nemotron = [model for model in available if _is_nemotron(model)]
    if not nemotron:
        raise LLMUnavailable(f"no Nemotron model available for role {role}")

    preferences = {
        "fast": ("lightning", "nano"),
        "deep": ("super",),
        "ultra": ("ultra", "super"),
    }[role]
    for keyword in preferences:
        match = next((model for model in nemotron if keyword in model.lower()), None)
        if match:
            return match
    return sorted(nemotron)[0]
