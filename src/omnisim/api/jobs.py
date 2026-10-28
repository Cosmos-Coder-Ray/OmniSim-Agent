from __future__ import annotations

import multiprocessing
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Protocol

from omnisim.api.paths import ApiPaths


@dataclass(frozen=True, slots=True)
class JobConfig:
    campaign_id: str
    request: dict[str, object]
    paths: ApiPaths
    cancel_flag: Path
    timeout_s: int


@dataclass(frozen=True, slots=True)
class JobExit:
    status: str
    exitcode: int | None
    forced: bool


class JobTarget(Protocol):
    def __call__(self, config: JobConfig) -> None: ...


class JobRunner(Protocol):
    def start(self) -> None: ...

    def submit(self, config: JobConfig) -> tuple[bool, int | None]: ...

    def queue_position(self, campaign_id: str) -> int | None: ...

    def queue_snapshot(self) -> dict[str, int]: ...

    def cancel(self, campaign_id: str) -> bool: ...

    def shutdown(self, grace_s: float = 5.0) -> None: ...


@dataclass
class _ManagedJob:
    config: JobConfig
    process: BaseProcess
    started: float
    cancel_at: float | None = None
    reason: str | None = None
    forced: bool = False


class ProcessJobRunner:
    def __init__(
        self,
        target: JobTarget,
        *,
        max_concurrent: int,
        max_queue: int,
        on_started: Callable[[JobConfig], None],
        on_finished: Callable[[JobConfig, JobExit], None],
        cancel_grace_s: float = 1.0,
    ) -> None:
        self._target = target
        self._max_concurrent = max_concurrent
        self._max_queue = max_queue
        self._on_started = on_started
        self._on_finished = on_finished
        self._cancel_grace_s = cancel_grace_s
        self._context = multiprocessing.get_context("spawn")
        self._condition = threading.Condition()
        self._queue: deque[JobConfig] = deque()
        self._active: dict[str, _ManagedJob] = {}
        self._closing = False
        self._close_deadline: float | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name="omnisim-job-supervisor")
            self._thread.start()

    def submit(self, config: JobConfig) -> tuple[bool, int | None]:
        with self._condition:
            if self._closing or len(self._queue) >= self._max_queue:
                return False, None
            position = (
                len(self._queue) + 1
                if len(self._active) >= self._max_concurrent or self._queue
                else None
            )
            self._queue.append(config)
            self._condition.notify_all()
            return True, position

    def queue_position(self, campaign_id: str) -> int | None:
        with self._condition:
            for index, config in enumerate(self._queue, 1):
                if config.campaign_id == campaign_id:
                    return index
        return None

    def queue_snapshot(self) -> dict[str, int]:
        with self._condition:
            return {
                "queued": len(self._queue),
                "running": len(self._active),
                "max_concurrent": self._max_concurrent,
                "max_queue": self._max_queue,
            }

    def cancel(self, campaign_id: str) -> bool:
        cancelled: JobConfig | None = None
        with self._condition:
            for config in self._queue:
                if config.campaign_id == campaign_id:
                    self._queue.remove(config)
                    cancelled = config
                    break
            managed = self._active.get(campaign_id)
            if managed is not None:
                self._request_cancel(managed, "user")
                self._condition.notify_all()
                return True
        if cancelled is not None:
            self._on_finished(cancelled, JobExit("cancelled", None, False))
            return True
        return False

    def shutdown(self, grace_s: float = 5.0) -> None:
        with self._condition:
            self._closing = True
            self._close_deadline = time.monotonic() + max(0.0, grace_s)
            queued = list(self._queue)
            self._queue.clear()
            for managed in self._active.values():
                self._request_cancel(managed, "shutdown")
            self._condition.notify_all()
            thread = self._thread
        for config in queued:
            self._on_finished(config, JobExit("cancelled", None, False))
        if thread is not None:
            thread.join(max(1.0, grace_s + self._cancel_grace_s + 2.0))

    def _request_cancel(self, managed: _ManagedJob, reason: str) -> None:
        if managed.cancel_at is not None:
            return
        managed.reason = reason
        managed.cancel_at = time.monotonic()
        managed.config.cancel_flag.parent.mkdir(parents=True, exist_ok=True)
        managed.config.cancel_flag.touch()

    def _finish(self, campaign_id: str, managed: _ManagedJob) -> None:
        process = managed.process
        process.join(timeout=0)
        exitcode = process.exitcode
        if managed.forced:
            status = "error" if managed.reason == "error" else "interrupted"
        elif managed.reason == "error":
            status = "error"
        elif managed.reason in {"shutdown", "user"}:
            status = "cancelled"
        else:
            status = "completed" if exitcode == 0 else "interrupted"
        del self._active[campaign_id]
        process.close()
        self._on_finished(managed.config, JobExit(status, exitcode, managed.forced))

    def _run(self) -> None:
        while True:
            with self._condition:
                while (
                    not self._closing and len(self._active) < self._max_concurrent and self._queue
                ):
                    config = self._queue.popleft()
                    process = self._context.Process(target=self._target, args=(config,))
                    process.daemon = False
                    try:
                        process.start()
                    except (OSError, ValueError):
                        self._on_finished(config, JobExit("error", None, False))
                        continue
                    managed = _ManagedJob(config, process, time.monotonic())
                    self._active[config.campaign_id] = managed
                    self._on_started(config)

                now = time.monotonic()
                for campaign_id, managed in list(self._active.items()):
                    managed.process.join(timeout=0)
                    if managed.process.exitcode is not None:
                        self._finish(campaign_id, managed)
                    elif (
                        managed.cancel_at is None
                        and now - managed.started >= managed.config.timeout_s
                    ):
                        self._request_cancel(managed, "error")
                    elif (
                        managed.cancel_at is not None
                        and now - managed.cancel_at >= self._cancel_grace_s
                    ):
                        if (
                            self._closing
                            and self._close_deadline is not None
                            and now < self._close_deadline
                        ):
                            continue
                        managed.process.terminate()
                        managed.process.join(timeout=1)
                        managed.forced = True
                        self._finish(campaign_id, managed)

                if self._closing and not self._active:
                    return
                self._condition.wait(0.05)
