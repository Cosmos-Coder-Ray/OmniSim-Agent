from __future__ import annotations

import json
import time
from math import isfinite
from pathlib import Path
from typing import Protocol

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


class Shield(Protocol):
    def apply(self, params: ScenarioParams) -> ScenarioParams: ...


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
) -> EpisodeResult:
    if not isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be positive")

    original_params = params.model_copy(deep=True)
    sim_params = (
        shield.apply(params.model_copy(deep=True)) if shield is not None else original_params
    )
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
