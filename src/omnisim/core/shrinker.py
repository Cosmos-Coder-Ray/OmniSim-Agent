from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from omnisim.agents.baselines import scenario_bounds
from omnisim.core.clustering import effective_params
from omnisim.schemas import EpisodeResult, FailureKind, ScenarioParams

_PARAMS = (
    "payload_kg",
    "link_mass_scale",
    "joint_damping",
    "latency_ticks",
    "sensor_noise_rad",
    "torque_limit_scale",
    "speed_scale",
    "kp_scale",
    "push_force_n",
)
_MAX_STEPS_PER_PARAM = 4
_NORMALIZED_TOLERANCE = 0.01

ClusterKey = tuple[FailureKind, int | None]
Evaluation = Callable[[ScenarioParams], EpisodeResult | None]


@dataclass(frozen=True, slots=True)
class ShrinkResult:
    minimal: ScenarioParams
    sims_used: int
    converged: bool
    trace: list[tuple[str, float | int, bool]]


def _persists(result: EpisodeResult | None, key: ClusterKey) -> bool:
    if result is None:
        return False
    kind, joint = key
    return any(
        event.kind == kind and (joint is None or event.joint == joint) for event in result.failures
    )


def _normalized_distance(name: str, value: float | int, nominal: ScenarioParams) -> float:
    low, high = scenario_bounds()[name]
    return abs(float(value) - float(getattr(nominal, name))) / (high - low)


def _within_tolerance(name: str, preserved: float | int, rejected: float | int) -> bool:
    low, high = scenario_bounds()[name]
    distance = abs(float(preserved) - float(rejected)) / (high - low)
    if name == "latency_ticks":
        return abs(int(preserved) - int(rejected)) <= 1
    return distance <= _NORMALIZED_TOLERANCE


def _midpoint(name: str, left: float | int, right: float | int) -> float | int:
    value = (float(left) + float(right)) / 2
    return int(round(value)) if name == "latency_ticks" else value


def shrink(
    params: ScenarioParams,
    cluster_key: ClusterKey,
    evaluate: Evaluation,
    cap: int = 25,
) -> ShrinkResult:
    if cap <= 0:
        raise ValueError("cap must be positive")

    nominal = ScenarioParams()
    minimal = effective_params(params)
    ordered = sorted(
        _PARAMS,
        key=lambda name: (_normalized_distance(name, getattr(minimal, name), nominal), name),
    )
    trace: list[tuple[str, float | int, bool]] = []
    sims_used = 0
    non_nominal: list[str] = []
    intervals: dict[str, tuple[float | int, float | int]] = {}

    for name in ordered:
        original = getattr(minimal, name)
        target = getattr(nominal, name)
        if original == target:
            continue
        if sims_used >= cap:
            return ShrinkResult(minimal, sims_used, False, trace)
        candidate = minimal.model_copy(update={name: target})
        result = evaluate(candidate)
        sims_used += 1
        persisted = _persists(result, cluster_key)
        trace.append((name, target, persisted))
        if persisted:
            minimal = effective_params(candidate)
        else:
            non_nominal.append(name)
            intervals[name] = (original, target)

    steps = {name: 0 for name in non_nominal}
    while sims_used < cap:
        progressed = False
        for name in non_nominal:
            if sims_used >= cap:
                break
            if steps[name] >= _MAX_STEPS_PER_PARAM:
                continue
            preserved, rejected = intervals[name]
            if _within_tolerance(name, preserved, rejected):
                continue
            value = _midpoint(name, preserved, rejected)
            if value in {preserved, rejected}:
                continue
            candidate = minimal.model_copy(update={name: value})
            result = evaluate(candidate)
            sims_used += 1
            persisted = _persists(result, cluster_key)
            trace.append((name, value, persisted))
            steps[name] += 1
            progressed = True
            if persisted:
                minimal = effective_params(candidate)
                intervals[name] = (value, rejected)
            else:
                intervals[name] = (preserved, value)
        if not progressed:
            break

    converged = all(
        _within_tolerance(name, *intervals[name]) or steps[name] >= _MAX_STEPS_PER_PARAM
        for name in non_nominal
    )
    if any(
        steps[name] >= _MAX_STEPS_PER_PARAM and not _within_tolerance(name, *intervals[name])
        for name in non_nominal
    ):
        converged = False
    if sims_used >= cap:
        converged = False
    return ShrinkResult(minimal, sims_used, converged, trace)
