from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from math import inf, isfinite
from pathlib import Path
from time import monotonic
from typing import Any, Protocol

import numpy as np

from omnisim.agents.baselines import PARAM_NAMES, scenario_bounds
from omnisim.core.shield import EpisodeBlocked, SafetyShield, Shield
from omnisim.schemas import EpisodeResult, FailureKind, SafeEnvelope, ScenarioParams
from omnisim.sim.detectors import DetectorConfig
from omnisim.sim.limits import UrdfLimits
from omnisim.sim.policy import planned_duration_s
from omnisim.sim.runner import EpisodeError, EpisodeRunner

Clock = Callable[[], float]
BatchResults = Sequence[EpisodeResult | EpisodeError | EpisodeBlocked]


class VerificationRunner(Protocol):
    def run_batch(
        self,
        params_list: list[ScenarioParams],
        limits: UrdfLimits,
        *,
        workers: int,
        shield: Shield | None = None,
        run_dir: Path | None = None,
        cfg: DetectorConfig = DetectorConfig(),
    ) -> BatchResults: ...

    def close(self) -> None: ...


TUNING_SCENARIOS = 100
CONFIRMATION_SCENARIOS = 200
MAX_REFINEMENTS = 3


@dataclass(frozen=True, slots=True)
class VerificationCriteria:
    unsafe_rate_max: float = 0.05
    blocked_rate_max: float = 0.40
    completion_rate_min: float = 0.90
    provisional: bool = True


@dataclass(frozen=True, slots=True)
class VerificationBaseline:
    scenarios: tuple[ScenarioParams, ...]
    results: tuple[EpisodeResult | EpisodeError, ...]


@dataclass(frozen=True, slots=True)
class VerificationResult:
    n_scenarios: int
    seed: int
    scenario_seeds: tuple[int, ...]
    fail_rate_before: float
    fail_rate_after: float
    blocked_count: int
    blocked_rate: float
    executed_count: int
    completion_count: int
    completion_rate: float
    mean_relative_cycle_time_increase: float | None
    per_kind: dict[str, dict[str, int | float]]
    overlap_count: int
    overlap_rate: float
    overlap_epsilon: float
    infrastructure_errors: int
    passed: bool
    wall_time_s: float
    failing_scenarios: tuple[dict[str, object], ...]
    scenario_results: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


EnvelopeDraft = Callable[[int, SafeEnvelope | None, VerificationResult | None], SafeEnvelope]


@dataclass(frozen=True, slots=True)
class RefinementResult:
    envelope: SafeEnvelope
    tuning_results: tuple[VerificationResult, ...]
    confirmation: VerificationResult
    tuning_seed: int
    confirmation_seed: int
    showcases: tuple[dict[str, object], ...] = ()


def _digest_seed(domain: str, seed: int, index: int = 0) -> int:
    payload = f"omnisim-verify-v1\0{domain}\0{seed}\0{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def verification_seeds(campaign_seed: int) -> tuple[int, int]:
    reserved = {campaign_seed}
    roots: list[int] = []
    for domain in ("tuning-root", "confirmation-root"):
        candidate = 100 + _digest_seed(domain, campaign_seed) % 900
        while candidate in reserved:
            candidate = 100 + (candidate - 99) % 900
        roots.append(candidate)
        reserved.add(candidate)
    return roots[0], roots[1]


def _scenario_seed(seed: int, index: int, used: set[int]) -> int:
    candidate = _digest_seed("scenario", seed, index)
    while candidate in used:
        candidate = (candidate + 1) % (2**32)
    used.add(candidate)
    return candidate


def _validate_bounds(
    bounds: Mapping[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    if set(bounds) != set(PARAM_NAMES):
        raise ValueError("verification bounds must cover every ScenarioParams field except seed")
    schema_bounds = scenario_bounds()
    validated = {name: (float(low), float(high)) for name, (low, high) in bounds.items()}
    for name, (low, high) in validated.items():
        schema_low, schema_high = schema_bounds[name]
        if (
            not isfinite(low)
            or not isfinite(high)
            or low >= high
            or low < schema_low
            or high > schema_high
        ):
            raise ValueError(f"verification bounds are outside ScenarioParams limits: {name}")
    return validated


def sample_scenarios(
    n: int,
    seed: int,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    *,
    reserved_seeds: Collection[int] = (),
) -> list[ScenarioParams]:
    if n <= 0:
        raise ValueError("n must be positive")
    active_bounds = _validate_bounds(bounds if bounds is not None else scenario_bounds())
    rng = np.random.default_rng(seed)
    used = set(reserved_seeds)
    scenarios: list[ScenarioParams] = []
    for index in range(n):
        values: dict[str, float | int] = {}
        for name in PARAM_NAMES:
            low, high = active_bounds[name]
            value = float(rng.uniform(low, high))
            values[name] = int(round(value)) if name == "latency_ticks" else value
        values["seed"] = _scenario_seed(seed, index, used)
        scenarios.append(ScenarioParams.model_validate(values))
    return scenarios


def _run_batch(
    runner: VerificationRunner,
    scenarios: list[ScenarioParams],
    limits: UrdfLimits,
    *,
    workers: int,
    cfg: DetectorConfig,
    run_dir: Path | None,
) -> list[EpisodeResult | EpisodeError]:
    results = runner.run_batch(
        scenarios,
        limits,
        workers=workers,
        shield=None,
        run_dir=run_dir,
        cfg=cfg,
    )
    if len(results) != len(scenarios):
        raise RuntimeError("verification runner returned a different number of results")
    if any(isinstance(result, EpisodeBlocked) for result in results):
        raise RuntimeError("verification OFF/accepted run unexpectedly returned EpisodeBlocked")
    return [result for result in results if isinstance(result, (EpisodeResult, EpisodeError))]


def run_off_baseline(
    scenarios: Sequence[ScenarioParams],
    limits: UrdfLimits,
    *,
    workers: int,
    cfg: DetectorConfig,
    runner: VerificationRunner,
    run_dir: Path | None = None,
) -> VerificationBaseline:
    scenario_list = list(scenarios)
    results = _run_batch(
        runner,
        scenario_list,
        limits,
        workers=workers,
        cfg=cfg,
        run_dir=run_dir,
    )
    return VerificationBaseline(tuple(scenario_list), tuple(results))


def _bounds_from_envelope(envelope: SafeEnvelope) -> dict[str, tuple[float, float]]:
    raw = envelope.domain_randomization_bounds
    if not raw:
        return scenario_bounds()
    bounds: dict[str, tuple[float, float]] = {}
    for name, value in raw.items():
        if (
            name not in PARAM_NAMES
            or not isinstance(value, (list, tuple))
            or len(value) != 2
            or isinstance(value[0], bool)
            or isinstance(value[1], bool)
            or not isinstance(value[0], (int, float))
            or not isinstance(value[1], (int, float))
        ):
            raise ValueError(f"invalid verification bounds for {name}")
        bounds[name] = (float(value[0]), float(value[1]))
    return _validate_bounds(bounds)


def _per_kind(
    off_results: Sequence[EpisodeResult | EpisodeError],
    on_results: Sequence[EpisodeResult | EpisodeError | None],
    n: int,
) -> dict[str, dict[str, int | float]]:
    kinds = tuple(FailureKind.__args__)
    output: dict[str, dict[str, int | float]] = {}
    for kind in kinds:
        off = sum(
            isinstance(result, EpisodeResult)
            and any(event.kind == kind for event in result.failures)
            for result in off_results
        )
        on = sum(
            isinstance(result, EpisodeResult)
            and any(event.kind == kind for event in result.failures)
            for result in on_results
        )
        output[kind] = {"off_count": off, "off_rate": off / n, "on_count": on, "on_rate": on / n}
    return output


def _overlap_count(
    scenarios: Sequence[ScenarioParams],
    discovery: Sequence[EpisodeResult],
    bounds: Mapping[str, tuple[float, float]],
    epsilon: float,
) -> int:
    if not discovery:
        return 0
    for name, limits in bounds.items():
        if limits[1] <= limits[0]:
            raise ValueError(f"invalid normalization bounds for {name}")
    overlaps = 0
    for scenario in scenarios:
        for episode in discovery:
            distance = max(
                abs(float(getattr(scenario, name)) - float(getattr(episode.params, name)))
                / (bounds[name][1] - bounds[name][0])
                for name in PARAM_NAMES
            )
            if distance <= epsilon:
                overlaps += 1
                break
    return overlaps


def verify(
    envelope: SafeEnvelope,
    limits: UrdfLimits,
    *,
    n: int,
    seed: int,
    workers: int,
    cfg: DetectorConfig,
    runner: VerificationRunner | None = None,
    clock: Clock = monotonic,
    scenarios: Sequence[ScenarioParams] | None = None,
    off_baseline: VerificationBaseline | None = None,
    discovery_episodes: Sequence[EpisodeResult] = (),
    criteria: VerificationCriteria = VerificationCriteria(),
    run_dir: Path | None = None,
) -> VerificationResult:
    if n <= 0 or workers <= 0:
        raise ValueError("n and workers must be positive")
    started = clock()
    own_runner = runner is None
    active_runner = runner or EpisodeRunner()
    try:
        bounds = _bounds_from_envelope(envelope)
        scenario_list = (
            list(scenarios) if scenarios is not None else sample_scenarios(n, seed, bounds)
        )
        if len(scenario_list) != n:
            raise ValueError("scenario count must match n")
        baseline = off_baseline or run_off_baseline(
            scenario_list,
            limits,
            workers=workers,
            cfg=cfg,
            runner=active_runner,
            run_dir=run_dir,
        )
        if baseline.scenarios != tuple(scenario_list) or len(baseline.results) != n:
            raise ValueError("OFF baseline does not match verification scenarios")

        shield = SafetyShield(envelope)
        decisions = [shield.decide(params) for params in scenario_list]
        blocked_count = sum(decision.rejected for decision in decisions)
        accepted_indices = [i for i, decision in enumerate(decisions) if not decision.rejected]
        accepted_params = [decisions[i].params for i in accepted_indices]
        accepted_results = (
            _run_batch(
                active_runner,
                accepted_params,
                limits,
                workers=workers,
                cfg=cfg,
                run_dir=run_dir,
            )
            if accepted_params
            else []
        )
        on_results: list[EpisodeResult | EpisodeError | None] = [None] * n
        for index, result in zip(accepted_indices, accepted_results, strict=True):
            if isinstance(result, EpisodeResult):
                result = result.model_copy(update={"params": scenario_list[index]})
            on_results[index] = result

        off_failed = sum(
            isinstance(result, EpisodeResult) and bool(result.failures)
            for result in baseline.results
        )
        on_failed = sum(
            isinstance(result, EpisodeResult) and bool(result.failures) for result in on_results
        )
        executed = [result for result in on_results if isinstance(result, EpisodeResult)]
        completion_count = sum(
            not result.failures and result.final_ee_error_m <= cfg.tracking_error_m
            for result in executed
        )
        cycle_increases: list[float] = []
        for index, on_result in enumerate(on_results):
            off_result = baseline.results[index]
            if not isinstance(off_result, EpisodeResult) or not isinstance(
                on_result, EpisodeResult
            ):
                continue
            off_duration = planned_duration_s(scenario_list[index].speed_scale)
            on_duration = planned_duration_s(decisions[index].params.speed_scale)
            cycle_increases.append(on_duration / off_duration - 1.0)

        infrastructure_errors = sum(isinstance(result, EpisodeError) for result in baseline.results)
        infrastructure_errors += sum(isinstance(result, EpisodeError) for result in on_results)
        fail_rate_before = off_failed / n
        fail_rate_after = on_failed / n
        blocked_rate = blocked_count / n
        completion_rate = completion_count / n
        passed = (
            infrastructure_errors == 0
            and fail_rate_after <= criteria.unsafe_rate_max
            and blocked_rate <= criteria.blocked_rate_max
            and completion_rate >= criteria.completion_rate_min
        )
        failing = tuple(
            {
                "params": scenario_list[index].model_dump(mode="json"),
                "kinds": sorted({event.kind for event in result.failures}),
            }
            for index, result in enumerate(on_results)
            if isinstance(result, EpisodeResult) and result.failures
        )
        scenario_results: list[dict[str, Any]] = []
        for index in range(n):
            off_result = baseline.results[index]
            on_result = on_results[index]
            scenario_results.append(
                {
                    "index": index,
                    "params": scenario_list[index].model_dump(mode="json"),
                    "off": {
                        "infrastructure_error": isinstance(off_result, EpisodeError),
                        "result": off_result.model_dump(mode="json")
                        if isinstance(off_result, EpisodeResult)
                        else None,
                        "kinds": sorted({event.kind for event in off_result.failures})
                        if isinstance(off_result, EpisodeResult)
                        else [],
                    },
                    "on": {
                        "blocked": on_result is None,
                        "infrastructure_error": isinstance(on_result, EpisodeError),
                        "result": on_result.model_dump(mode="json")
                        if isinstance(on_result, EpisodeResult)
                        else None,
                        "kinds": sorted({event.kind for event in on_result.failures})
                        if isinstance(on_result, EpisodeResult)
                        else [],
                        "final_error_m": on_result.final_ee_error_m
                        if isinstance(on_result, EpisodeResult)
                        else None,
                        "complete": isinstance(on_result, EpisodeResult)
                        and not on_result.failures
                        and on_result.final_ee_error_m <= cfg.tracking_error_m,
                    },
                }
            )
        return VerificationResult(
            n_scenarios=n,
            seed=seed,
            scenario_seeds=tuple(params.seed for params in scenario_list),
            fail_rate_before=fail_rate_before,
            fail_rate_after=fail_rate_after,
            blocked_count=blocked_count,
            blocked_rate=blocked_rate,
            executed_count=len(executed),
            completion_count=completion_count,
            completion_rate=completion_rate,
            mean_relative_cycle_time_increase=float(np.mean(cycle_increases))
            if cycle_increases
            else None,
            per_kind=_per_kind(baseline.results, on_results, n),
            overlap_count=_overlap_count(
                scenario_list,
                discovery_episodes,
                bounds,
                epsilon=0.01,
            ),
            overlap_rate=_overlap_count(
                scenario_list,
                discovery_episodes,
                bounds,
                epsilon=0.01,
            )
            / n,
            overlap_epsilon=0.01,
            infrastructure_errors=infrastructure_errors,
            passed=passed,
            wall_time_s=max(0.0, clock() - started),
            failing_scenarios=failing,
            scenario_results=tuple(scenario_results),
        )
    finally:
        if own_runner:
            active_runner.close()


def refine_and_verify(
    draft: EnvelopeDraft,
    *,
    campaign_seed: int,
    limits: UrdfLimits,
    workers: int,
    cfg: DetectorConfig,
    runner: VerificationRunner | None = None,
    clock: Clock = monotonic,
    domain_bounds: Mapping[str, tuple[float, float]] | None = None,
    discovery_episodes: Sequence[EpisodeResult] = (),
    criteria: VerificationCriteria = VerificationCriteria(),
    stage_callback: Callable[[str], None] | None = None,
    tuning_count: int = TUNING_SCENARIOS,
    confirmation_count: int = CONFIRMATION_SCENARIOS,
) -> RefinementResult:
    if workers <= 0 or tuning_count <= 0 or confirmation_count <= 0:
        raise ValueError("workers and verification sample counts must be positive")
    tuning_seed, confirmation_seed = verification_seeds(campaign_seed)
    bounds = _validate_bounds(domain_bounds if domain_bounds is not None else scenario_bounds())
    tuning_scenarios = sample_scenarios(
        tuning_count,
        tuning_seed,
        bounds,
        reserved_seeds=(campaign_seed,),
    )
    confirmation_scenarios = sample_scenarios(
        confirmation_count,
        confirmation_seed,
        bounds,
        reserved_seeds=(campaign_seed, *(params.seed for params in tuning_scenarios)),
    )
    own_runner = runner is None
    active_runner = runner or EpisodeRunner()
    try:
        tuning_off = run_off_baseline(
            tuning_scenarios,
            limits,
            workers=workers,
            cfg=cfg,
            runner=active_runner,
        )
        confirmation_off = run_off_baseline(
            confirmation_scenarios,
            limits,
            workers=workers,
            cfg=cfg,
            runner=active_runner,
        )
        tuning_results: list[VerificationResult] = []
        candidates: list[SafeEnvelope] = []
        previous_envelope: SafeEnvelope | None = None
        previous_result: VerificationResult | None = None
        if stage_callback is not None:
            stage_callback("mitigation")
        for iteration in range(MAX_REFINEMENTS):
            candidate = draft(iteration, previous_envelope, previous_result)
            if _bounds_from_envelope(candidate) != bounds:
                raise ValueError("draft envelope bounds must match verification domain bounds")
            candidates.append(candidate)
            if iteration == 0 and stage_callback is not None:
                stage_callback("verification")
            result = verify(
                candidate,
                limits,
                n=tuning_count,
                seed=tuning_seed,
                workers=workers,
                cfg=cfg,
                runner=active_runner,
                clock=clock,
                scenarios=tuning_scenarios,
                off_baseline=tuning_off,
                discovery_episodes=discovery_episodes,
                criteria=criteria,
            )
            tuning_results.append(result)
            previous_envelope = candidate
            previous_result = result
            if result.passed:
                break

        selected_index = min(
            range(len(tuning_results)),
            key=lambda index: (
                not tuning_results[index].passed,
                tuning_results[index].blocked_count,
                tuning_results[index].mean_relative_cycle_time_increase
                if tuning_results[index].mean_relative_cycle_time_increase is not None
                else inf,
            ),
        )
        if not candidates:
            raise RuntimeError("verification did not produce an envelope")
        selected = candidates[selected_index]
        confirmation = verify(
            selected,
            limits,
            n=confirmation_count,
            seed=confirmation_seed,
            workers=workers,
            cfg=cfg,
            runner=active_runner,
            clock=clock,
            scenarios=confirmation_scenarios,
            off_baseline=confirmation_off,
            discovery_episodes=discovery_episodes,
            criteria=criteria,
        )
        return RefinementResult(
            envelope=selected,
            tuning_results=tuple(tuning_results),
            confirmation=confirmation,
            tuning_seed=tuning_seed,
            confirmation_seed=confirmation_seed,
        )
    finally:
        if own_runner:
            active_runner.close()


def write_verification(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
