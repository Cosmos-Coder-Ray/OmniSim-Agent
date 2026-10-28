from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from omnisim.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    RetryDeadlineExceeded,
    parse_retry_after,
    retry_call,
)


class TransportError(Exception):
    pass


def test_retry_call_uses_full_jitter_and_stops_after_four_attempts() -> None:
    attempts = 0
    delays: list[float] = []

    def call() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 4:
            raise TransportError
        return "ok"

    result = retry_call(
        call,
        retryable=lambda error: isinstance(error, TransportError),
        sleep=delays.append,
        rng=lambda: 0.5,
    )

    assert result == "ok"
    assert attempts == 4
    assert delays == [0.5, 1.0, 2.0]


def test_retry_after_is_honored_and_capped() -> None:
    delays: list[float] = []

    def call() -> None:
        raise TransportError

    with pytest.raises(TransportError):
        retry_call(
            call,
            retryable=lambda error: True,
            retry_after=lambda error: 45,
            max_retries=1,
            sleep=delays.append,
            rng=lambda: 0,
        )

    assert delays == [30.0]


def test_retry_deadline_stops_before_another_attempt() -> None:
    now = 0.0
    attempts = 0

    def call() -> None:
        nonlocal attempts
        attempts += 1
        raise TransportError

    def sleep(delay: float) -> None:
        nonlocal now
        now += delay

    with pytest.raises(TransportError):
        retry_call(
            call,
            retryable=lambda error: True,
            deadline=1.0,
            sleep=sleep,
            clock=lambda: now,
            rng=lambda: 1,
        )

    assert attempts == 1


def test_retry_rejects_non_retryable_error_without_sleep() -> None:
    delays: list[float] = []

    with pytest.raises(ValueError, match="bad request"):
        retry_call(
            lambda: (_ for _ in ()).throw(ValueError("bad request")),
            retryable=lambda error: isinstance(error, TransportError),
            sleep=delays.append,
        )

    assert delays == []


def test_circuit_breaker_opens_waits_for_one_probe_and_closes() -> None:
    now = 0.0
    breaker = CircuitBreaker(threshold=3, cooldown_s=60, clock=lambda: now)
    transitions: list[str] = []

    breaker.exhausted(on_transition=transitions.append)
    breaker.exhausted(on_transition=transitions.append)
    breaker.exhausted(on_transition=transitions.append)
    with pytest.raises(CircuitOpenError):
        breaker.before_call()

    now = 60.0
    assert breaker.before_call()
    with pytest.raises(CircuitOpenError):
        breaker.before_call()
    breaker.success(on_transition=transitions.append)

    assert breaker.snapshot()["open"] is False
    assert transitions == ["open", "closed"]


def test_retry_after_accepts_delta_seconds_and_http_date() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    retry_at = now + timedelta(seconds=25)

    assert parse_retry_after("12") == 12
    assert (
        parse_retry_after(retry_at.strftime("%a, %d %b %Y %H:%M:%S GMT"), now=now.timestamp()) == 25
    )
    assert parse_retry_after("unknown") is None


def test_expired_deadline_without_attempt_raises_deadline_error() -> None:
    with pytest.raises(RetryDeadlineExceeded):
        retry_call(
            lambda: "unused",
            retryable=lambda error: True,
            deadline=0,
            clock=lambda: 0,
        )
