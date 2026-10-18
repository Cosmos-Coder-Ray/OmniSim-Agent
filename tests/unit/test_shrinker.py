from __future__ import annotations

from omnisim.agents.baselines import scenario_bounds
from omnisim.core.shrinker import _PARAMS, shrink
from omnisim.schemas import EpisodeResult, FailureEvent, ScenarioParams

_KEY = ("torque_saturation", 2)


def _result(params: ScenarioParams, persists: bool) -> EpisodeResult:
    failures = (
        [FailureEvent(kind="torque_saturation", t=1.0, joint=2, severity=1.0)] if persists else []
    )
    return EpisodeResult(
        episode_id="ep_test",
        params=params,
        failures=failures,
        risk=1.1 if persists else 0.1,
        max_tau_ratio=0.0,
        max_vel_ratio=0.0,
        final_ee_error_m=0.0,
        min_manipulability=0.0,
    )


def test_shrinker_converges_to_known_synthetic_boundary() -> None:
    original = ScenarioParams(payload_kg=0.05, seed=42)

    def evaluate(params: ScenarioParams) -> EpisodeResult:
        return _result(params, params.payload_kg >= 0.006)

    result = shrink(original, _KEY, evaluate)

    assert result.converged
    assert abs(result.minimal.payload_kg - 0.006) / scenario_bounds()["payload_kg"][1] <= 0.01
    assert result.minimal.seed == original.seed
    assert result.trace[0] == ("payload_kg", 0.0, False)
    assert result.sims_used == len(result.trace) == 2


def test_shrinker_respects_simulation_cap() -> None:
    original = ScenarioParams(payload_kg=3.0)

    result = shrink(original, _KEY, lambda params: _result(params, params.payload_kg >= 1.0), cap=3)

    assert result.sims_used == 3
    assert len(result.trace) == 3
    assert not result.converged


def test_latency_bisection_uses_integer_ticks() -> None:
    original = ScenarioParams(latency_ticks=12)
    tried: list[int] = []

    def evaluate(params: ScenarioParams) -> EpisodeResult:
        tried.append(params.latency_ticks)
        return _result(params, params.latency_ticks >= 5)

    result = shrink(original, _KEY, evaluate)

    assert result.converged
    assert result.minimal.latency_ticks == 5
    assert all(isinstance(value, int) for value in tried)


def test_push_time_resets_only_when_zero_force_preserves_failure() -> None:
    original = ScenarioParams(push_force_n=8.0, push_time_s=4.5)

    persists = shrink(original, _KEY, lambda params: _result(params, True))
    fails_at_zero = shrink(
        original,
        _KEY,
        lambda params: _result(params, params.push_force_n > 0.0),
    )

    assert persists.minimal.push_force_n == 0.0
    assert persists.minimal.push_time_s == ScenarioParams().push_time_s
    assert fails_at_zero.minimal.push_force_n > 0.0
    assert fails_at_zero.minimal.push_time_s == original.push_time_s


def test_all_nine_params_get_a_bisection_after_all_snaps() -> None:
    original = ScenarioParams(
        payload_kg=3.0,
        link_mass_scale=1.5,
        joint_damping=5.0,
        latency_ticks=12,
        sensor_noise_rad=0.02,
        torque_limit_scale=0.5,
        speed_scale=2.0,
        kp_scale=0.5,
        push_force_n=40.0,
        seed=77,
    )
    nominal = ScenarioParams()
    tried_seeds: list[int] = []

    def evaluate(params: ScenarioParams) -> EpisodeResult:
        tried_seeds.append(params.seed)
        persists = True
        for name in _PARAMS:
            value = getattr(params, name)
            start = getattr(original, name)
            base = getattr(nominal, name)
            threshold = (float(start) + float(base)) / 2
            persists &= value < threshold if start < base else value > threshold
        return _result(params, persists)

    result = shrink(original, _KEY, evaluate)

    snapped_names = [name for name, _, _ in result.trace[: len(_PARAMS)]]
    bisected_names = {name for name, _, _ in result.trace[len(_PARAMS) :]}
    assert snapped_names == sorted(
        _PARAMS,
        key=lambda name: (
            abs(float(getattr(original, name)) - float(getattr(nominal, name)))
            / (scenario_bounds()[name][1] - scenario_bounds()[name][0]),
            name,
        ),
    )
    assert bisected_names == set(_PARAMS)
    assert len(result.trace) == result.sims_used <= 25
    assert not result.converged
    assert set(tried_seeds) == {original.seed}
    for name in _PARAMS:
        low, high = sorted((getattr(nominal, name), getattr(original, name)))
        assert low <= getattr(result.minimal, name) <= high


def test_global_cluster_does_not_match_joint_specific_event() -> None:
    original = ScenarioParams(payload_kg=1.0, seed=17)

    result = shrink(
        original, ("torque_saturation", None), lambda params: _result(params, True), cap=1
    )

    assert result.minimal.payload_kg == original.payload_kg
    assert result.trace == [("payload_kg", 0.0, False)]
