from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from omnisim.core.shield import (
    MAX_SPEED_SCALE,
    MIN_SPEED_SCALE,
    SafetyShield,
    build_latency_speed_table,
    build_payload_speed_table,
)
from omnisim.schemas import SafeEnvelope, ScenarioParams
from omnisim.sim.constants import TICK_MS
from omnisim.sim.limits import UrdfLimits


def envelope(
    *,
    max_payload_kg: float = 3.0,
    max_latency_ticks: int = 12,
    speed_cap: float = 2.0,
    kp_cap: float = 2.0,
) -> SafeEnvelope:
    return SafeEnvelope(
        robot="franka_panda",
        version="test",
        limits={
            "max_payload_kg": max_payload_kg,
            "max_latency_ms": max_latency_ticks * TICK_MS,
            "torque_utilization_cap": 0.95,
            "speed_scale_max_by_payload_kg": (
                [[0.0, speed_cap]]
                if max_payload_kg == 0
                else [[0.0, speed_cap], [max_payload_kg, speed_cap]]
            ),
            "speed_scale_max_by_latency_ticks": [speed_cap] * 13,
        },
        controller={"kp_scale_max": kp_cap},
        domain_randomization_bounds={},
        evidence=[],
        verification={},
    )


@given(
    payload=st.floats(min_value=0.0, max_value=3.0, allow_nan=False, allow_infinity=False),
    latency=st.integers(min_value=0, max_value=12),
    speed=st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
    kp=st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_shield_only_reduces_controller_settings(payload, latency, speed, kp) -> None:
    params = ScenarioParams(
        payload_kg=payload,
        latency_ticks=latency,
        speed_scale=speed,
        kp_scale=kp,
        link_mass_scale=1.4,
        joint_damping=3.0,
        sensor_noise_rad=0.01,
        torque_limit_scale=0.7,
        push_force_n=20.0,
        push_time_s=4.0,
        seed=901,
    )
    decision = SafetyShield(envelope(speed_cap=1.5, kp_cap=1.25)).decide(params)

    assert not decision.rejected
    assert decision.params.payload_kg == params.payload_kg
    assert decision.params.latency_ticks == params.latency_ticks
    assert decision.params.link_mass_scale == params.link_mass_scale
    assert decision.params.joint_damping == params.joint_damping
    assert decision.params.sensor_noise_rad == params.sensor_noise_rad
    assert decision.params.torque_limit_scale == params.torque_limit_scale
    assert decision.params.push_force_n == params.push_force_n
    assert decision.params.push_time_s == params.push_time_s
    assert decision.params.seed == params.seed
    assert decision.params.speed_scale <= 1.5
    assert decision.params.kp_scale <= 1.25


@given(
    payload=st.floats(min_value=0.0, max_value=3.0, allow_nan=False, allow_infinity=False),
    latency=st.integers(min_value=0, max_value=12),
)
def test_shield_is_idempotent(payload, latency) -> None:
    shield = SafetyShield(envelope(speed_cap=1.3, kp_cap=1.1))
    params = ScenarioParams(payload_kg=payload, latency_ticks=latency)
    first = shield.decide(params)
    second = shield.decide(first.params)
    assert second == first


@given(
    payload=st.floats(min_value=0.0, max_value=3.0, allow_nan=False, allow_infinity=False),
    latency=st.integers(min_value=0, max_value=12),
)
def test_unobservable_parameters_do_not_change_decision(payload, latency) -> None:
    shield = SafetyShield(envelope(speed_cap=1.6, kp_cap=1.4))
    base = ScenarioParams(payload_kg=payload, latency_ticks=latency)
    changed = base.model_copy(
        update={
            "link_mass_scale": 1.5,
            "joint_damping": 5.0,
            "sensor_noise_rad": 0.02,
            "torque_limit_scale": 0.5,
            "push_force_n": 40.0,
            "push_time_s": 5.0,
            "seed": 917,
        }
    )
    first = shield.decide(base)
    second = shield.decide(changed)
    assert first.rejected == second.rejected
    assert first.reasons == second.reasons
    assert (
        first.params.model_copy(
            update={
                "link_mass_scale": changed.link_mass_scale,
                "joint_damping": changed.joint_damping,
                "sensor_noise_rad": changed.sensor_noise_rad,
                "torque_limit_scale": changed.torque_limit_scale,
                "push_force_n": changed.push_force_n,
                "push_time_s": changed.push_time_s,
                "seed": changed.seed,
            }
        )
        == second.params
    )


@given(
    max_payload=st.floats(min_value=0.0, max_value=3.0, allow_nan=False, allow_infinity=False),
    latency_cap=st.integers(min_value=0, max_value=11),
)
def test_physical_parameters_over_limits_are_rejected(max_payload, latency_cap) -> None:
    shield = SafetyShield(envelope(max_payload_kg=max_payload, max_latency_ticks=latency_cap))
    over_payload = ScenarioParams(payload_kg=min(3.0, max_payload + 0.001))
    over_latency = ScenarioParams(latency_ticks=latency_cap + 1)
    if over_payload.payload_kg > max_payload:
        assert shield.decide(over_payload).rejected
    assert shield.decide(over_latency).rejected


@given(
    speed=st.floats(min_value=MIN_SPEED_SCALE, max_value=MAX_SPEED_SCALE, allow_nan=False),
    kp=st.floats(min_value=0.5, max_value=2.0, allow_nan=False),
)
def test_stricter_envelope_never_allows_more(speed, kp) -> None:
    params = ScenarioParams(speed_scale=speed, kp_scale=kp)
    loose = SafetyShield(envelope(speed_cap=1.6, kp_cap=1.4)).decide(params)
    strict = SafetyShield(envelope(speed_cap=1.2, kp_cap=1.0)).decide(params)
    assert strict.rejected == loose.rejected
    assert strict.params.speed_scale <= loose.params.speed_scale
    assert strict.params.kp_scale <= loose.params.kp_scale


def test_latency_speed_caps_are_monotone() -> None:
    caps = build_latency_speed_table()
    assert len(caps) == 13
    assert all(left >= right for left, right in zip(caps, caps[1:]))


def test_payload_speed_table_is_monotone_and_rejects_infeasible_payloads() -> None:
    limits = UrdfLimits(
        effort=np.ones(7),
        velocity=np.ones(7),
        lower=-np.ones(7),
        upper=np.ones(7),
    )

    def predictor(payload_kg: float, speed_scale: float, _: UrdfLimits) -> float:
        return payload_kg + speed_scale

    table = build_payload_speed_table(0.6, 1.0, limits, predictor=predictor)
    caps = [row[1] for row in table]
    assert table[-1][0] == 0.6
    assert all(left >= right for left, right in zip(caps, caps[1:]))
    assert caps[-1] == 0.0
