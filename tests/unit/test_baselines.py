from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import pytest

from omnisim.agents import baselines
from omnisim.agents.baselines import (
    CMAESStrategy,
    ProposalOrigin,
    RandomStrategy,
    Trial,
    denormalize,
    normalize,
    scenario_bounds,
)
from omnisim.core.campaign import run_trials
from omnisim.schemas import EpisodeResult, ScenarioParams
from omnisim.sim.detectors import DetectorConfig
from omnisim.sim.limits import UrdfLimits, load_urdf_limits
from omnisim.sim.runner import EpisodeError, EpisodeRunner


def _result(params: ScenarioParams, risk: float) -> EpisodeResult:
    return EpisodeResult(
        episode_id="ep_000",
        params=params,
        failures=[],
        risk=risk,
        max_tau_ratio=0.0,
        max_vel_ratio=0.0,
        final_ee_error_m=0.0,
        min_manipulability=0.0,
    )


def test_bounds_match_scenario_field_constraints() -> None:
    assert scenario_bounds() == {
        "payload_kg": (0.0, 3.0),
        "link_mass_scale": (0.7, 1.5),
        "joint_damping": (0.0, 5.0),
        "latency_ticks": (0.0, 12.0),
        "sensor_noise_rad": (0.0, 0.02),
        "torque_limit_scale": (0.5, 1.0),
        "speed_scale": (0.5, 2.0),
        "kp_scale": (0.5, 2.0),
        "push_force_n": (0.0, 40.0),
        "push_time_s": (0.0, 5.0),
    }


def test_normalize_and_denormalize_round_latency() -> None:
    bounds = scenario_bounds()
    params = ScenarioParams(latency_ticks=7, payload_kg=1.5)

    restored = denormalize(normalize(params, bounds), bounds)

    assert restored == params
    assert denormalize([0.51] * 10, bounds).latency_ticks == 6


def test_random_proposals_are_deterministic_and_in_bounds() -> None:
    bounds = scenario_bounds()
    first = RandomStrategy(5)
    second = RandomStrategy(5)

    _, proposals = first.propose([], bounds, 4)
    _, repeated = second.propose([], bounds, 4)

    assert proposals == repeated
    assert first.last_origins == ["random"] * 4
    assert all(
        low <= getattr(params, name) <= high
        for params in proposals
        for name, (low, high) in bounds.items()
    )


def test_cmaes_same_seed_and_truncated_batch() -> None:
    bounds = scenario_bounds()
    first = CMAESStrategy(0)
    second = CMAESStrategy(0)

    _, proposals = first.propose([], bounds, 3)
    _, repeated = second.propose([], bounds, 3)

    assert proposals == repeated
    assert len(proposals) == 3
    assert first.last_origins == ["cma"] * 3
    assert all(
        bounds[name][0] <= getattr(params, name) <= bounds[name][1]
        for params in proposals
        for name in bounds
    )
    assert all(isinstance(params.latency_ticks, int) for params in proposals)


def test_cmaes_tells_negative_risk_and_zero_for_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeES:
        instances: list[FakeES] = []
        popsize = 4
        sigma = 0.3
        countiter = 0
        told: tuple[list[list[float]], list[float]]

        def __init__(self, x0: list[float], sigma0: float, options: dict[str, object]) -> None:
            assert x0 == [0.5] * 10
            assert sigma0 == 0.3
            assert options["seed"] != 0
            self.instances.append(self)

        def ask(self, *, number: int) -> list[list[float]]:
            return [[0.5] * 10 for _ in range(number)]

        def tell(self, solutions: list[list[float]], values: list[float]) -> None:
            self.told = (solutions, values)
            self.countiter += 1

    monkeypatch.setattr(baselines.cma, "CMAEvolutionStrategy", FakeES)
    strategy = CMAESStrategy(0)
    bounds = scenario_bounds()
    _, first = strategy.propose([], bounds, 2)
    errors = [Trial(first[0], None, "WorkerTimeout"), Trial(first[1], _result(first[1], 1.5), None)]

    strategy.propose(errors, bounds, 1)

    assert FakeES.instances[0].told[1] == [0.0, -1.5]


def test_run_trials_assigns_same_seed_sequence_and_reports_errors() -> None:
    class FakeStrategy:
        batch_size = 3
        last_origins: list[ProposalOrigin] = ["random"] * 3

        def propose(
            self,
            history: list[Trial],
            bounds: Mapping[str, tuple[float, float]],
            k: int,
        ) -> tuple[str, list[ScenarioParams]]:
            return "test", [ScenarioParams() for _ in range(k)]

    class FakeRunner:
        def run_batch(
            self,
            params_list: list[ScenarioParams],
            limits: UrdfLimits,
            *,
            workers: int,
            run_dir: None,
            cfg: DetectorConfig,
        ) -> list[EpisodeResult | EpisodeError]:
            assert workers == 1
            assert run_dir is None
            return [
                _result(params_list[0], 0.0),
                EpisodeError("ep_001", params_list[1], "WorkerTimeout", "timed out"),
                _result(params_list[2], 0.0),
            ]

    captured: list[tuple[int, str, list[str], list[Trial]]] = []
    history = run_trials(
        FakeStrategy(),
        cast(EpisodeRunner, FakeRunner()),
        load_urdf_limits(),
        benchmark_seed=123,
        budget=3,
        workers=1,
        on_batch=lambda *row: captured.append(row),
    )

    assert len(history) == 3
    assert history[1].result is None
    assert history[1].error == "WorkerTimeout: timed out"
    assert len(captured[0][2]) == 3
    assert [trial.params.seed for trial in history] == [
        trial.params.seed for trial in captured[0][3]
    ]
    assert history[0].params.seed == int(
        np.random.SeedSequence([123, 0]).generate_state(1, dtype=np.uint32)[0]
    )
