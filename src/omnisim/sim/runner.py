from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from multiprocessing.pool import Pool
from multiprocessing.process import BaseProcess
from pathlib import Path
from queue import Empty, Queue

import pybullet as p
import pybullet_data

from omnisim.audit_log import configure_logging
from omnisim.core.shield import EpisodeBlocked, Shield
from omnisim.schemas import (
    EpisodeResult,
    FailureEvent,
    Replay,
    ReplayFrame,
    ScenarioParams,
)
from omnisim.sim.detectors import DetectorConfig, detect, violation_masks
from omnisim.sim.limits import UrdfLimits
from omnisim.sim.panda_env import PandaSim
from omnisim.sim.telemetry import EpisodeTelemetry

_PARENT_TIMEOUT_GRACE_S = 1.0
_worker_start_queue = None


@dataclass(frozen=True, slots=True)
class EpisodeError:
    episode_id: str
    params: ScenarioParams
    error_type: str
    message: str


def _round_floats(value: object) -> object:
    if isinstance(value, float):
        rounded = round(value, 6)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    return value


def _write_replay(
    telemetry: EpisodeTelemetry,
    limits: UrdfLimits,
    cfg: DetectorConfig,
    episode_id: str,
    events: list[FailureEvent],
    run_dir: Path,
) -> Path:
    downsampled = telemetry.downsample()
    masks = violation_masks(telemetry, limits, cfg)
    stride = 60 // downsampled["fps"]
    frames: list[ReplayFrame] = []
    for index, data in enumerate(downsampled["frames"]):
        start = index * stride
        end = min(start + stride, len(telemetry.t))
        data["flags"] = [kind for kind, mask in masks.items() if mask[start:end].any()]
        frames.append(ReplayFrame.model_validate(data))

    replay = Replay(
        episode_id=episode_id,
        fps=downsampled["fps"],
        frames=frames,
        events=events,
        source="live",
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{episode_id}.json"
    payload = json.dumps(
        _round_floats(replay.model_dump(mode="json")),
        separators=(",", ":"),
        allow_nan=False,
    )
    path.write_text(payload, encoding="utf-8")
    return path


def run_episode(
    params: ScenarioParams,
    limits: UrdfLimits,
    shield: Shield | None = None,
    *,
    episode_id: str,
    run_dir: Path | None = None,
    cfg: DetectorConfig = DetectorConfig(),
    timeout_s: float = 30.0,
) -> EpisodeResult | EpisodeBlocked:
    if not isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be positive")

    original_params = params.model_copy(deep=True)
    if shield is None:
        sim_params = original_params
    else:
        decision = shield.decide(original_params.model_copy(deep=True))
        if decision.rejected:
            return EpisodeBlocked(episode_id, original_params, decision.reasons)
        sim_params = decision.params
    deadline = time.monotonic() + timeout_s
    with PandaSim(sim_params, limits.effort, limits.velocity) as sim:
        telemetry = sim.run(deadline)

    events, risk, metrics = detect(telemetry, limits, cfg)
    trajectory_path = None
    if run_dir is not None:
        path = _write_replay(telemetry, limits, cfg, episode_id, events, run_dir)
        trajectory_path = path.as_posix()

    return EpisodeResult(
        episode_id=episode_id,
        params=original_params,
        failures=events,
        risk=risk,
        max_tau_ratio=metrics["max_tau_ratio"],
        max_vel_ratio=metrics["max_vel_ratio"],
        final_ee_error_m=metrics["final_ee_error_m"],
        min_manipulability=metrics["min_manipulability"],
        trajectory_path=trajectory_path,
    )


def _worker_init(start_queue=None) -> None:
    global _worker_start_queue
    _worker_start_queue = start_queue
    configure_logging()
    p.getAPIVersion()
    pybullet_data.getDataPath()


def _error_result(episode_id: str, params: ScenarioParams, error: BaseException) -> EpisodeError:
    return EpisodeError(
        episode_id=episode_id,
        params=params.model_copy(deep=True),
        error_type=type(error).__name__,
        message=str(error) or type(error).__name__,
    )


def _run_worker(
    params: ScenarioParams,
    limits: UrdfLimits,
    shield: Shield | None,
    episode_id: str,
    run_dir: Path | None,
    cfg: DetectorConfig,
    timeout_s: float,
) -> EpisodeResult | EpisodeError | EpisodeBlocked:
    try:
        return run_episode(
            params,
            limits,
            shield,
            episode_id=episode_id,
            run_dir=run_dir,
            cfg=cfg,
            timeout_s=timeout_s,
        )
    except (OSError, TimeoutError, p.error) as error:
        return _error_result(episode_id, params, error)


def _invoke_worker(
    worker_fn: Callable,
    params: ScenarioParams,
    limits: UrdfLimits,
    shield: Shield | None,
    episode_id: str,
    run_dir: Path | None,
    cfg: DetectorConfig,
    timeout_s: float,
) -> EpisodeResult | EpisodeError | EpisodeBlocked:
    if _worker_start_queue is not None:
        _worker_start_queue.put((os.getpid(), episode_id))
    return worker_fn(params, limits, shield, episode_id, run_dir, cfg, timeout_s)


class EpisodeRunner:
    def __init__(self, worker_fn: Callable = _run_worker) -> None:
        self._worker_fn = worker_fn
        self._pool: Pool | None = None
        self._workers: int | None = None
        self._worker_starts = None
        self._episode_count = 0
        self._closed = False

    def __enter__(self) -> EpisodeRunner:
        if self._closed:
            raise RuntimeError("episode runner is closed")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close(terminate=exc_type is not None)

    def close(self, *, terminate: bool = False) -> None:
        if self._pool is not None:
            if terminate:
                self._pool.terminate()
            else:
                self._pool.close()
            self._pool.join()
            self._pool = None
            self._workers = None
        if self._worker_starts is not None:
            close = getattr(self._worker_starts, "close", None)
            if close is not None:
                close()
            self._worker_starts = None
        self._closed = True

    def _ensure_pool(self, workers: int) -> None:
        if self._pool is not None and self._workers == workers:
            return
        if self._pool is not None:
            self.close(terminate=True)
        self._closed = False
        self._worker_starts = mp.get_context("spawn").SimpleQueue()
        self._pool = mp.get_context("spawn").Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(self._worker_starts,),
        )
        self._workers = workers

    def run_batch(
        self,
        params_list: list[ScenarioParams],
        limits: UrdfLimits,
        *,
        workers: int,
        shield: Shield | None = None,
        run_dir: Path | None = None,
        cfg: DetectorConfig = DetectorConfig(),
        timeout_s: float = 30.0,
    ) -> list[EpisodeResult | EpisodeError | EpisodeBlocked]:
        if self._closed:
            raise RuntimeError("episode runner is closed")
        if not isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if not params_list:
            return []

        episode_ids = [f"ep_{self._episode_count + i:03d}" for i in range(len(params_list))]
        self._episode_count += len(params_list)
        if workers <= 1:
            serial_results: list[EpisodeResult | EpisodeError | EpisodeBlocked] = []
            for params, episode_id in zip(params_list, episode_ids, strict=True):
                try:
                    result = run_episode(
                        params,
                        limits,
                        shield,
                        episode_id=episode_id,
                        run_dir=run_dir,
                        cfg=cfg,
                        timeout_s=timeout_s,
                    )
                except (OSError, TimeoutError, p.error) as error:
                    result = _error_result(episode_id, params, error)
                serial_results.append(result)
            return serial_results

        self._ensure_pool(workers)
        assert self._pool is not None
        pool = self._pool
        completion_queue: Queue[tuple[int, int, EpisodeResult | EpisodeError | EpisodeBlocked]] = (
            Queue()
        )
        generation = 0
        results: list[EpisodeResult | EpisodeError | EpisodeBlocked | None] = [None] * len(
            params_list
        )
        in_flight: dict[int, float] = {}
        worker_assignments: dict[int, int] = {}
        worker_processes: dict[int, BaseProcess] = {}
        pending_retry: deque[int] = deque()
        next_index = 0

        def submit(index: int) -> None:
            params = params_list[index]
            episode_id = episode_ids[index]
            task_generation = generation
            in_flight[index] = time.monotonic()

            def on_success(result: EpisodeResult | EpisodeError | EpisodeBlocked) -> None:
                completion_queue.put((task_generation, index, result))

            def on_error(error: BaseException) -> None:
                completion_queue.put(
                    (task_generation, index, _error_result(episode_id, params, error))
                )

            try:
                pool.apply_async(
                    _invoke_worker,
                    (self._worker_fn, params, limits, shield, episode_id, run_dir, cfg, timeout_s),
                    callback=on_success,
                    error_callback=on_error,
                )
            except (OSError, RuntimeError, ValueError) as error:
                in_flight.pop(index)
                results[index] = _error_result(episode_id, params, error)

        def fill_slots() -> None:
            nonlocal next_index
            while len(in_flight) < workers:
                if pending_retry:
                    index = pending_retry.popleft()
                elif next_index < len(params_list):
                    index = next_index
                    next_index += 1
                else:
                    return
                submit(index)

        def drain_worker_starts() -> None:
            if self._worker_starts is None:
                return
            while not self._worker_starts.empty():
                pid, episode_id = self._worker_starts.get()
                index = episode_ids_by_name.get(episode_id)
                if index is None:
                    continue
                worker_assignments[pid] = index

        def accept_completion(
            index: int, result: EpisodeResult | EpisodeError | EpisodeBlocked
        ) -> None:
            in_flight.pop(index, None)
            results[index] = result
            for pid, assigned in list(worker_assignments.items()):
                if assigned == index:
                    worker_assignments.pop(pid, None)

        def refresh_workers() -> list[int]:
            for process in getattr(pool, "_pool", ()):
                if process.pid is not None:
                    worker_processes[process.pid] = process
            return [
                pid for pid, process in worker_processes.items() if process.exitcode is not None
            ]

        def restart_after_crash(crashed: list[int]) -> None:
            nonlocal pool, generation, worker_processes
            crashed_indexes = {
                worker_assignments[pid]
                for pid in crashed
                if pid in worker_assignments and worker_assignments[pid] in in_flight
            }
            if not crashed_indexes:
                return
            retry_indexes = sorted(set(pending_retry) | (set(in_flight) - crashed_indexes))
            for index in crashed_indexes:
                params = params_list[index]
                results[index] = EpisodeError(
                    episode_ids[index],
                    params.model_copy(deep=True),
                    "worker_crashed",
                    "worker exited unexpectedly",
                )
                in_flight.pop(index, None)
            self.close(terminate=True)
            generation += 1
            self._closed = False
            self._ensure_pool(workers)
            assert self._pool is not None
            pool = self._pool
            worker_processes = {}
            worker_assignments.clear()
            pending_retry.clear()
            pending_retry.extend(retry_indexes)
            fill_slots()

        episode_ids_by_name = {episode_id: index for index, episode_id in enumerate(episode_ids)}
        fill_slots()

        while in_flight:
            drain_worker_starts()
            while True:
                try:
                    completed_generation, completed_index, completed_result = (
                        completion_queue.get_nowait()
                    )
                except Empty:
                    break
                if completed_generation == generation:
                    accept_completion(completed_index, completed_result)
            if not in_flight:
                fill_slots()
                continue
            dead_workers = refresh_workers()
            if dead_workers:
                restart_after_crash(dead_workers)
                continue
            deadline = min(in_flight.values()) + timeout_s + _PARENT_TIMEOUT_GRACE_S
            wait_s = max(0.0, deadline - time.monotonic())
            try:
                completed_generation, index, result = completion_queue.get(
                    timeout=min(wait_s, 0.05)
                )
            except Empty:
                if time.monotonic() < deadline:
                    continue
                self.close(terminate=True)
                for unfinished, params in enumerate(params_list):
                    if results[unfinished] is None:
                        results[unfinished] = EpisodeError(
                            episode_id=episode_ids[unfinished],
                            params=params.model_copy(deep=True),
                            error_type="WorkerTimeout",
                            message="worker did not return before parent timeout",
                        )
                self._closed = False
                self._ensure_pool(workers)
                pool = self._pool
                worker_processes = {}
                worker_assignments.clear()
                pending_retry.clear()
                break

            if completed_generation == generation:
                accept_completion(index, result)
            fill_slots()

        if any(result is None for result in results):
            raise RuntimeError("batch ended without a result for every episode")
        completed_results: list[EpisodeResult | EpisodeError | EpisodeBlocked] = []
        for result in results:
            if result is None:
                raise RuntimeError("batch ended without a result for every episode")
            completed_results.append(result)
        return completed_results
