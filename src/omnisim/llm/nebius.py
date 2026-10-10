import logging
import time
from time import perf_counter
from typing import Any, Literal, TypeVar

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from omnisim.config import Settings
from omnisim.llm.budget import LLMError, Usage, check, record, request_slot
from omnisim.llm.call_log import write_call_log
from omnisim.llm.json_utils import _objects, _strip_think
from omnisim.llm.models import Role, resolve_model

log = logging.getLogger(__name__)
TIMEOUT_S = 30.0
MAX_RETRY_TOKENS = 8192
T = TypeVar("T", bound=BaseModel)
_client: OpenAI | None = None


class LLMOutputError(LLMError):
    def __init__(self, message: str, last_error: Exception) -> None:
        super().__init__(message)
        self.last_error = last_error


class LLMUnavailable(LLMError):
    pass


def get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    settings = Settings.from_env()
    if not settings.NEBIUS_API_KEY:
        raise ValueError("NEBIUS_API_KEY is required")
    if not settings.NEBIUS_BASE_URL:
        raise ValueError("NEBIUS_BASE_URL is required")
    _client = OpenAI(
        api_key=settings.NEBIUS_API_KEY,
        base_url=settings.NEBIUS_BASE_URL,
        max_retries=0,
        timeout=TIMEOUT_S,
    )
    return _client


def _content_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        text = content
    else:
        text = ""
    text = _strip_think(text)
    if text.strip():
        return text
    reasoning = getattr(message, "reasoning_content", None)
    return reasoning if isinstance(reasoning, str) else ""


def _validate_candidates(
    text: str, schema: type[T], *, reverse: bool
) -> tuple[T | None, str, Exception | None]:
    candidates = _objects(text)
    if reverse:
        candidates.reverse()
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return schema.model_validate(candidate), "", None
        except ValidationError as exc:
            last_error = exc
    if candidates:
        return None, "schema", last_error or ValueError("schema validation failed")
    return None, "invalid_json", ValueError("no balanced JSON object found")


def _usage(response: Any) -> Usage:
    usage = getattr(response, "usage", None)
    if usage is None:
        log.warning("llm response has no usage data")
        return Usage()
    return Usage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, (openai.APITimeoutError, openai.APIConnectionError)):
        return True
    if isinstance(error, openai.APIStatusError):
        return error.status_code == 429 or error.status_code >= 500
    return False


def _log_attempt(
    *,
    run_id: str | None,
    role: Role,
    model: str,
    attempt: int,
    latency_s: float,
    usage: Usage,
    finish_reason: str | None,
    ok: bool,
    error: str | None,
) -> None:
    write_call_log(
        {
            "run_id": run_id,
            "role": role,
            "model": model,
            "attempt": attempt,
            "latency_s": latency_s,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "finish_reason": finish_reason,
            "ok": ok,
            "error": error,
        }
    )


def chat_json(
    role: Literal["fast", "deep", "ultra"],
    system: str,
    user: str,
    schema: type[T],
    *,
    temperature: float,
    max_tokens: int,
    retries: int = 1,
    run_id: str | None = None,
) -> tuple[T, Usage]:
    settings = Settings.from_env()
    check(role, cap=settings.OMNISIM_DAILY_TOKEN_CAP)
    client = get_client()
    try:
        model = resolve_model(client, role)
    except openai.OpenAIError as exc:
        raise LLMUnavailable(f"model listing failed ({type(exc).__name__})") from exc
    total = Usage()
    attempt = 0
    output_attempt = 0
    transport_retried = False
    current_max_tokens = max_tokens
    repair = ""
    last_error: Exception = ValueError("no valid JSON response")

    while output_attempt <= retries:
        attempt += 1
        request_error: openai.OpenAIError | None = None
        with request_slot(role, cap=settings.OMNISIM_DAILY_TOKEN_CAP) as slot:
            started = perf_counter()
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {
                            "role": "user",
                            "content": user + (f"\n\n{repair}" if repair else ""),
                        },
                    ],
                    temperature=temperature,
                    max_tokens=current_max_tokens,
                )
            except (
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.APIStatusError,
            ) as exc:
                latency_s = perf_counter() - started
                record(role, 0, 0, connection=slot)
                request_error = exc
            else:
                latency_s = perf_counter() - started
                usage = _usage(response)
                record(role, usage.prompt_tokens, usage.completion_tokens, connection=slot)

        if request_error is not None:
            exc = request_error
            _log_attempt(
                run_id=run_id,
                role=role,
                model=model,
                attempt=attempt,
                latency_s=latency_s,
                usage=Usage(),
                finish_reason=None,
                ok=False,
                error=type(exc).__name__,
            )
            status = exc.status_code if isinstance(exc, openai.APIStatusError) else None
            log.warning(
                "llm request failed",
                extra={"role": role, "model": model, "error": type(exc).__name__, "status": status},
            )
            if _is_retryable(exc) and not transport_retried:
                transport_retried = True
                time.sleep(0.25)
                continue
            raise LLMUnavailable(f"llm request failed ({type(exc).__name__})") from exc

        total = Usage(
            prompt_tokens=total.prompt_tokens + usage.prompt_tokens,
            completion_tokens=total.completion_tokens + usage.completion_tokens,
        )
        choice = response.choices[0]
        finish_reason = choice.finish_reason
        message = choice.message
        content = getattr(message, "content", None)
        clean_content = content if isinstance(content, str) else ""
        clean_content = _strip_think(clean_content)
        reasoning = not clean_content.strip()
        text = _content_text(message)
        parsed, error_kind, candidate_error = _validate_candidates(text, schema, reverse=reasoning)
        if parsed is not None:
            _log_attempt(
                run_id=run_id,
                role=role,
                model=model,
                attempt=attempt,
                latency_s=latency_s,
                usage=usage,
                finish_reason=finish_reason,
                ok=True,
                error=None,
            )
            return parsed, total

        last_error = candidate_error or ValueError(error_kind)
        _log_attempt(
            run_id=run_id,
            role=role,
            model=model,
            attempt=attempt,
            latency_s=latency_s,
            usage=usage,
            finish_reason=finish_reason,
            ok=False,
            error=error_kind,
        )
        output_attempt += 1
        if output_attempt > retries:
            break
        if finish_reason == "length":
            current_max_tokens = min(current_max_tokens * 2, MAX_RETRY_TOKENS)
        else:
            message = str(last_error)[:200]
            repair = f"Return valid JSON matching the requested schema. Fix this error: {message}"

    raise LLMOutputError(
        "llm response did not match the requested schema", last_error
    ) from last_error
