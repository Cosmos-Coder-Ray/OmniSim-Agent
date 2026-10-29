from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import TypeVar

T = TypeVar("T")

BASE_DELAY_S = 1.0
MAX_DELAY_S = 20.0
MAX_RETRY_AFTER_S = 30.0
MAX_RETRIES = 3
CALL_DEADLINE_S = 90.0
CIRCUIT_FAILURES = 3
CIRCUIT_COOLDOWN_S = 60.0


class CircuitOpenError(RuntimeError):
    pass


class RetryDeadlineExceeded(TimeoutError):
    pass


def parse_retry_after(value: str | None, *, now: float | None = None) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        current = datetime.fromtimestamp(time.time() if now is None else now, UTC)
        return max(0.0, (retry_at - current).total_seconds())


class CircuitBreaker:
    def __init__(
        self,
        *,
        threshold: int = CIRCUIT_FAILURES,
        cooldown_s: float = CIRCUIT_COOLDOWN_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = threshold
        self._cooldown_s = cooldown_s
        self._clock = clock
        self._failures = 0
        self._open_until: float | None = None
        self._probe_in_flight = False
        self._lock = threading.Lock()

    def before_call(self) -> bool:
        """Return true for the single half-open probe, or reject while open."""
        with self._lock:
            if self._open_until is None:
                return False
            if self._clock() < self._open_until or self._probe_in_flight:
                raise CircuitOpenError("circuit open")
            self._probe_in_flight = True
            return True

    def success(self, *, on_transition: Callable[[str], None] | None = None) -> None:
        with self._lock:
            was_open = self._open_until is not None
            self._failures = 0
            self._open_until = None
            self._probe_in_flight = False
        if was_open and on_transition is not None:
            on_transition("closed")

    def exhausted(self, *, on_transition: Callable[[str], None] | None = None) -> None:
        with self._lock:
            if self._probe_in_flight or self._failures + 1 >= self._threshold:
                self._failures = 0
                self._open_until = self._clock() + self._cooldown_s
                self._probe_in_flight = False
                opened = True
            else:
                self._failures += 1
                opened = False
        if opened and on_transition is not None:
            on_transition("open")

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "open": self._open_until is not None,
                "open_until": self._open_until,
                "probe_in_flight": self._probe_in_flight,
                "consecutive_failures": self._failures,
            }


def retry_call(
    call: Callable[[], T],
    *,
    retryable: Callable[[Exception], bool],
    retry_after: Callable[[Exception], float | None] | None = None,
    max_retries: int = MAX_RETRIES,
    deadline_s: float = CALL_DEADLINE_S,
    deadline: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    rng: Callable[[], float] = random.SystemRandom().random,
    on_success: Callable[[], None] | None = None,
    on_nonretryable: Callable[[Exception], None] | None = None,
    on_exhausted: Callable[[Exception], None] | None = None,
) -> T:
    if max_retries < 0 or deadline_s <= 0:
        raise ValueError("retry limits must be positive")
    deadline_at = clock() + deadline_s if deadline is None else deadline
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        if clock() >= deadline_at:
            if last_error is not None:
                if on_exhausted is not None:
                    on_exhausted(last_error)
                raise last_error
            raise RetryDeadlineExceeded("retry deadline exceeded")
        try:
            result = call()
        except Exception as error:
            if not retryable(error):
                if on_nonretryable is not None:
                    on_nonretryable(error)
                raise
            last_error = error
            if attempt >= max_retries:
                if on_exhausted is not None:
                    on_exhausted(error)
                raise
            ceiling = min(BASE_DELAY_S * 2**attempt, MAX_DELAY_S)
            delay = max(0.0, min(1.0, rng())) * ceiling
            server_delay = retry_after(error) if retry_after is not None else None
            if server_delay is not None:
                delay = max(delay, min(MAX_RETRY_AFTER_S, max(0.0, server_delay)))
            remaining = deadline_at - clock()
            if remaining <= 0:
                if on_exhausted is not None:
                    on_exhausted(error)
                raise
            sleep(min(delay, remaining))
        else:
            if clock() > deadline_at:
                error = RetryDeadlineExceeded("retry deadline exceeded")
                if on_exhausted is not None:
                    on_exhausted(error)
                raise error
            if on_success is not None:
                on_success()
            return result

    raise RuntimeError("retry loop ended without a result")
