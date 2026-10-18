from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import dataclass
from math import isfinite
from multiprocessing.pool import Pool
from pathlib import Path
from queue import Empty, Queue

import pybullet as p
import pybullet_data

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


def _worker_init() -> None:
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


class EpisodeRunner:
    def __init__(self) -> None:
        self._pool: Pool | None = None
        self._workers: int | None = None
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
        self._closed = True

    def _ensure_pool(self, workers: int) -> None:
        if self._pool is not None and self._workers == workers:
            return
        if self._pool is not None:
            self.close(terminate=True)
        self._closed = False
        self._pool = mp.get_context("spawn").Pool(
            processes=workers,
            initializer=_worker_init,
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
        completion_queue: Queue[tuple[int, EpisodeResult | EpisodeError | EpisodeBlocked]] = Queue()
        results: list[EpisodeResult | EpisodeError | EpisodeBlocked | None] = [None] * len(
            params_list
        )
        in_flight: dict[int, float] = {}
        next_index = 0

        def submit(index: int) -> None:
            params = params_list[index]
            episode_id = episode_ids[index]
            in_flight[index] = time.monotonic()

            def on_success(result: EpisodeResult | EpisodeError | EpisodeBlocked) -> None:
                completion_queue.put((index, result))

            def on_error(error: BaseException) -> None:
                completion_queue.put((index, _error_result(episode_id, params, error)))

            try:
                pool.apply_async(
                    _run_worker,
                    (params, limits, shield, episode_id, run_dir, cfg, timeout_s),
                    callback=on_success,
                    error_callback=on_error,
                )
            except (OSError, RuntimeError, ValueError) as error:
                in_flight.pop(index)
                results[index] = _error_result(episode_id, params, error)

        while next_index < len(params_list) and len(in_flight) < workers:
            submit(next_index)
            next_index += 1

        while in_flight:
            deadline = min(in_flight.values()) + timeout_s + _PARENT_TIMEOUT_GRACE_S
            wait_s = max(0.0, deadline - time.monotonic())
            try:
                index, result = completion_queue.get(timeout=wait_s)
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
                break

            in_flight.pop(index, None)
            results[index] = result
            while next_index < len(params_list) and len(in_flight) < workers:
                submit(next_index)
                next_index += 1

        if any(result is None for result in results):
            raise RuntimeError("batch ended without a result for every episode")
        completed_results: list[EpisodeResult | EpisodeError | EpisodeBlocked] = []
        for result in results:
            if result is None:
                raise RuntimeError("batch ended without a result for every episode")
            completed_results.append(result)
        return completed_results
