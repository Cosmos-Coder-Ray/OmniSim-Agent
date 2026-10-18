from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from omnisim.core.shield import Shield
from omnisim.core.verify import (
    CONFIRMATION_SCENARIOS,
    TUNING_SCENARIOS,
    BatchResults,
    refine_and_verify,
    sample_scenarios,
    verification_seeds,
    verify,
)
from omnisim.schemas import EpisodeResult, FailureEvent, SafeEnvelope, ScenarioParams
from omnisim.sim.detectors import DetectorConfig
from omnisim.sim.limits import UrdfLimits
from omnisim.sim.runner import EpisodeError


def _envelope(max_payload: float = 3.0) -> SafeEnvelope:
    return SafeEnvelope(
        robot="franka_panda",
        version="test",
        limits={
            "max_payload_kg": max_payload,
            "max_latency_ms": 200.0,
            "torque_utilization_cap": 0.95,
            "speed_scale_max_by_payload_kg": [[0.0, 2.0], [max_payload, 2.0]],
            "speed_scale_max_by_latency_ticks": [2.0] * 13,
        },
        controller={"kp_scale_max": 2.0},
        domain_randomization_bounds={},
        evidence=[],
        verification={},
    )


def _limits() -> UrdfLimits:
    ones = np.ones(7)
    return UrdfLimits(ones, ones, -ones, ones)


def _result(params: ScenarioParams, episode_id: str) -> EpisodeResult:
    failures = (
        [FailureEvent(kind="tracking_error", t=1.0, severity=1.1)] if params.seed == 11 else []
    )
    return EpisodeResult(
        episode_id=episode_id,
        params=params,
        failures=failures,
        risk=1.1 if failures else 0.2,
        max_tau_ratio=0.4,
        max_vel_ratio=0.3,
        final_ee_error_m=0.01,
        min_manipulability=0.2,
    )


class FakeRunner:
    def __init__(self, *, errors: bool = False) -> None:
        self.errors = errors
        self.calls: list[list[ScenarioParams]] = []

    def run_batch(
        self,
        params_list: list[ScenarioParams],
        limits: UrdfLimits,
        *,
        workers: int,
        shield: Shield | None = None,
        run_dir: Path | None = None,
        cfg=DetectorConfig(),
    ) -> BatchResults:
        self.calls.append(list(params_list))
        results = []
        for index, item in enumerate(params_list):
            episode_id = f"ep_{len(self.calls)}_{index}"
            if self.errors:
                results.append(EpisodeError(episode_id, item, "WorkerError", "failed"))
            else:
                results.append(_result(item, episode_id))
        return results

    def close(self) -> None:
        pass


def test_verification_seeds_are_reserved_and_scenarios_are_deterministic() -> None:
    tuning, confirmation = verification_seeds(321)
    assert 100 <= tuning <= 999
    assert 100 <= confirmation <= 999
    assert tuning != confirmation
    assert 321 not in (tuning, confirmation)
    first = sample_scenarios(16, tuning, reserved_seeds=(321,))
    second = sample_scenarios(16, tuning, reserved_seeds=(321,))
    other = sample_scenarios(16, confirmation, reserved_seeds=(321, *[p.seed for p in first]))
    assert first == second
    assert {item.seed for item in first}.isdisjoint(item.seed for item in other)
    assert 321 not in {item.seed for item in first + other}
    assert all(0 <= item.latency_ticks <= 12 for item in first)


def test_verify_accounts_for_blocked_and_per_kind_rates() -> None:
    runner = FakeRunner()
    scenarios = [
        ScenarioParams(seed=11, payload_kg=0.0),
        ScenarioParams(seed=12, payload_kg=0.5),
        ScenarioParams(seed=13, payload_kg=1.0),
        ScenarioParams(seed=14, payload_kg=2.0),
    ]
    result = verify(
        _envelope(max_payload=1.0),
        _limits(),
        n=len(scenarios),
        seed=200,
        workers=1,
        cfg=DetectorConfig(),
        runner=runner,
        scenarios=scenarios,
    )

    assert [len(call) for call in runner.calls] == [4, 3]
    assert result.blocked_count == 1
    assert result.fail_rate_before == pytest.approx(0.25)
    assert result.fail_rate_after == pytest.approx(0.25)
    assert result.completion_count == 2
    assert result.completion_rate == pytest.approx(0.5)
    assert result.per_kind["tracking_error"]["off_count"] == 1
    assert result.per_kind["tracking_error"]["on_count"] == 1
    assert result.scenario_seeds == (11, 12, 13, 14)


def test_blocked_all_verification_is_not_a_pass() -> None:
    runner = FakeRunner()
    scenarios = [ScenarioParams(seed=index, payload_kg=1.0) for index in range(3)]
    result = verify(
        _envelope(max_payload=0.0),
        _limits(),
        n=3,
        seed=200,
        workers=1,
        cfg=DetectorConfig(),
        runner=runner,
        scenarios=scenarios,
    )

    assert len(runner.calls) == 1
    assert result.blocked_count == 3
    assert result.fail_rate_after == 0.0
    assert result.executed_count == 0
    assert result.completion_rate == 0.0
    assert not result.passed


def test_verification_errors_fail_acceptance() -> None:
    runner = FakeRunner(errors=True)
    result = verify(
        _envelope(),
        _limits(),
        n=2,
        seed=200,
        workers=1,
        cfg=DetectorConfig(),
        runner=runner,
        scenarios=[ScenarioParams(seed=1), ScenarioParams(seed=2)],
    )

    assert result.infrastructure_errors == 4
    assert result.completion_rate == 0.0
    assert not result.passed


def test_refinement_runs_off_once_per_set_and_confirms_selected_draft() -> None:
    runner = FakeRunner()
    calls: list[int] = []

    def draft(iteration, previous, feedback):
        calls.append(iteration)
        return _envelope()

    result = refine_and_verify(
        draft,
        campaign_seed=17,
        limits=_limits(),
        workers=1,
        cfg=DetectorConfig(),
        runner=runner,
        clock=lambda: 1.0,
    )

    assert calls == [0]
    assert [len(batch) for batch in runner.calls] == [
        TUNING_SCENARIOS,
        CONFIRMATION_SCENARIOS,
        TUNING_SCENARIOS,
        CONFIRMATION_SCENARIOS,
    ]
    tuning_seeds = set(result.tuning_results[0].scenario_seeds)
    confirmation_seeds = set(result.confirmation.scenario_seeds)
    assert tuning_seeds.isdisjoint(confirmation_seeds)
    assert result.confirmation.n_scenarios == CONFIRMATION_SCENARIOS
