import logging
import random
import time
from collections.abc import Callable
from time import perf_counter
from typing import Any, Literal, TypeVar
from uuid import uuid4

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from omnisim.config import Settings
from omnisim.llm.budget import LLMError, LLMUnavailable, Usage, check, record, request_slot
from omnisim.llm.call_log import call_context, provider_host, write_call_log
from omnisim.llm.json_utils import _objects, _strip_think
from omnisim.llm.models import Role, resolve_model
from omnisim.resilience import (
    CALL_DEADLINE_S,
    CircuitBreaker,
    CircuitOpenError,
    RetryDeadlineExceeded,
    parse_retry_after,
    retry_call,
)

log = logging.getLogger(__name__)
TIMEOUT_S = 30.0
MAX_RETRY_TOKENS = 8192
T = TypeVar("T", bound=BaseModel)
_client: OpenAI | None = None
_circuit = CircuitBreaker()


class LLMOutputError(LLMError):
    def __init__(self, message: str, last_error: Exception) -> None:
        super().__init__(message)
        self.last_error = last_error


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
    call_id: str,
    host: str | None,
    status_code: int | None = None,
    failure_kind: str | None = None,
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
            "provider_host": host,
            "call_id": call_id,
            "status_code": status_code,
            "failure_kind": failure_kind,
        }
    )


def _retry_after_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    value = headers.get("Retry-After") if headers is not None else None
    if value is None:
        return None
    return parse_retry_after(value)


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
    retry_sleep: Callable[[float], None] | None = None,
    retry_clock: Callable[[], float] | None = None,
    retry_rng: Callable[[], float] | None = None,
) -> tuple[T, Usage]:
    settings = Settings.from_env()
    check(role, cap=settings.OMNISIM_DAILY_TOKEN_CAP)
    client = get_client()
    call_id = uuid4().hex
    host = provider_host(settings.NEBIUS_BASE_URL)
    sleep = retry_sleep or time.sleep
    clock = retry_clock or time.monotonic
    rng = retry_rng or random.SystemRandom().random
    deadline = clock() + CALL_DEADLINE_S

    def transition(state: str) -> None:
        write_call_log(
            {
                "run_id": run_id,
                "role": "circuit",
                "call_id": call_id,
                "provider": "nebius",
                "provider_host": host,
                "event": f"circuit_{state}",
            }
        )
        log.warning("provider circuit %s", state, extra={"provider": "nebius"})

    def provider_reachable(error: Exception) -> None:
        if isinstance(error, openai.OpenAIError) and not _is_retryable(error):
            _circuit.success(on_transition=transition)

    try:
        _circuit.before_call()
    except CircuitOpenError as exc:
        write_call_log(
            {
                "run_id": run_id,
                "role": "circuit",
                "call_id": call_id,
                "provider": "nebius",
                "provider_host": host,
                "event": "circuit_rejected",
                "ok": False,
                "error": "circuit_open",
            }
        )
        raise LLMUnavailable("circuit open") from exc

    with call_context(run_id=run_id, call_id=call_id, provider_host=host):
        model_attempt = 0

        def list_model() -> str:
            nonlocal model_attempt
            model_attempt += 1
            with call_context(attempt=model_attempt):
                return resolve_model(client, role)

        try:
            model = retry_call(
                list_model,
                retryable=_is_retryable,
                retry_after=_retry_after_seconds,
                deadline=deadline,
                sleep=sleep,
                clock=clock,
                rng=rng,
                on_nonretryable=provider_reachable,
                on_exhausted=lambda error: _circuit.exhausted(on_transition=transition),
            )
        except CircuitOpenError as exc:
            raise LLMUnavailable("circuit open") from exc
        except (openai.OpenAIError, RetryDeadlineExceeded) as exc:
            raise LLMUnavailable(f"model listing failed ({type(exc).__name__})") from exc

    total = Usage()
    attempt = 0
    output_attempt = 0
    current_max_tokens = max_tokens
    repair = ""
    last_error: Exception = ValueError("no valid JSON response")

    while output_attempt <= retries:

        def request() -> tuple[Any, Usage, float]:
            nonlocal attempt
            attempt += 1
            started = perf_counter()
            with request_slot(role, cap=settings.OMNISIM_DAILY_TOKEN_CAP) as slot:
                try:
                    request_timeout = max(0.01, min(TIMEOUT_S, deadline - clock()))
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
                        timeout=request_timeout,
                    )
                except (
                    openai.APITimeoutError,
                    openai.APIConnectionError,
                    openai.APIStatusError,
                ) as exc:
                    latency_s = perf_counter() - started
                    record(role, 0, 0, connection=slot)
                    status = exc.status_code if isinstance(exc, openai.APIStatusError) else None
                    is_transport = _is_retryable(exc)
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
                        call_id=call_id,
                        host=host,
                        status_code=status,
                        failure_kind="transport" if is_transport else "request",
                    )
                    log.warning(
                        "llm request failed",
                        extra={
                            "role": role,
                            "model": model,
                            "error": type(exc).__name__,
                            "status": status,
                        },
                    )
                    raise
                latency_s = perf_counter() - started
                usage = _usage(response)
                record(role, usage.prompt_tokens, usage.completion_tokens, connection=slot)
            return response, usage, latency_s

        try:
            response, usage, latency_s = retry_call(
                request,
                retryable=_is_retryable,
                retry_after=_retry_after_seconds,
                deadline=deadline,
                sleep=sleep,
                clock=clock,
                rng=rng,
                on_success=lambda: _circuit.success(on_transition=transition),
                on_nonretryable=provider_reachable,
                on_exhausted=lambda error: _circuit.exhausted(on_transition=transition),
            )
        except CircuitOpenError as exc:
            raise LLMUnavailable("circuit open") from exc
        except openai.OpenAIError as exc:
            raise LLMUnavailable(f"llm request failed ({type(exc).__name__})") from exc
        except RetryDeadlineExceeded as exc:
            raise LLMUnavailable("llm request deadline exceeded") from exc

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
                call_id=call_id,
                host=host,
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
            call_id=call_id,
            host=host,
            failure_kind="invalid_output",
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
