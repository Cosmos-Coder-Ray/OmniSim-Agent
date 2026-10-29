from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import pytest

from omnisim.api.jobs import JobConfig, JobExit, ProcessJobRunner
from omnisim.api.paths import ApiPaths


def _exit_abruptly(config: JobConfig) -> None:
    del config
    os._exit(9)


def _child_pid() -> int:
    return os.getpid()


def _nested_pool_job(config: JobConfig) -> None:
    config.paths.data.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
        child_pid = pool.submit(_child_pid).result(timeout=20)
        (config.paths.data / "child.pid").write_text(str(child_pid), encoding="ascii")
        while not config.cancel_flag.exists():
            time.sleep(0.02)
    while True:
        time.sleep(0.02)


@pytest.mark.slow
def test_shutdown_cancels_nested_pool_job_then_terminates_it(tmp_path: Path) -> None:
    paths = ApiPaths.from_root(tmp_path)
    exits: list[JobExit] = []
    runner = ProcessJobRunner(
        _nested_pool_job,
        max_concurrent=1,
        max_queue=1,
        on_started=lambda _: None,
        on_finished=lambda _, result: exits.append(result),
        cancel_grace_s=0.2,
    )
    config = JobConfig(
        campaign_id="campaign_a",
        request={},
        paths=paths,
        cancel_flag=paths.control / "campaign_a.cancel",
        timeout_s=30,
    )
    runner.start()
    assert runner.submit(config) == (True, None)
    marker = paths.data / "child.pid"
    deadline = time.monotonic() + 30
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert marker.exists()

    runner.shutdown(grace_s=0.1)

    assert len(exits) == 1
    assert exits[0].status == "interrupted"
    assert exits[0].forced is True
    assert exits[0].exitcode is not None


def test_unexpected_job_process_exit_is_interrupted(tmp_path: Path) -> None:
    paths = ApiPaths.from_root(tmp_path)
    finished = threading.Event()
    exits: list[JobExit] = []

    def on_finished(config: JobConfig, result: JobExit) -> None:
        exits.append(result)
        finished.set()

    runner = ProcessJobRunner(
        _exit_abruptly,
        max_concurrent=1,
        max_queue=1,
        on_started=lambda _: None,
        on_finished=on_finished,
        cancel_grace_s=0.2,
    )
    config = JobConfig(
        campaign_id="campaign_b",
        request={},
        paths=paths,
        cancel_flag=paths.control / "campaign_b.cancel",
        timeout_s=30,
    )
    runner.start()
    assert runner.submit(config) == (True, None)
    try:
        assert finished.wait(timeout=10)
    finally:
        runner.shutdown(grace_s=0.1)

    assert len(exits) == 1
    assert exits[0].status == "interrupted"
    assert exits[0].exitcode == 9
