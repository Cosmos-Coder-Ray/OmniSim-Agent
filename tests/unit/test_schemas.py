from datetime import UTC, datetime
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel, ValidationError

from omnisim.schemas import (
    BenchmarkResult,
    CampaignEvent,
    CampaignResult,
    EpisodeResult,
    Evidence,
    FailureEvent,
    HardwareSpec,
    Replay,
    ReplayFrame,
    SafeEnvelope,
    ScenarioParams,
)

SCENARIO_BOUNDS: tuple[tuple[str, int | float, int | float, bool], ...] = (
    ("payload_kg", 0.0, 3.0, False),
    ("link_mass_scale", 0.7, 1.5, False),
    ("joint_damping", 0.0, 5.0, False),
    ("latency_ticks", 0, 12, True),
    ("sensor_noise_rad", 0.0, 0.02, False),
    ("torque_limit_scale", 0.5, 1.0, False),
    ("speed_scale", 0.5, 2.0, False),
    ("kp_scale", 0.5, 2.0, False),
    ("push_force_n", 0.0, 40.0, False),
    ("push_time_s", 0.0, 5.0, False),
)


def sample_models() -> list[tuple[type[BaseModel], BaseModel]]:
    event = FailureEvent(kind="torque_saturation", t=1.0, severity=0.8)
    params = ScenarioParams(seed=17)
    episode = EpisodeResult(
        episode_id="ep_017",
        params=params,
        failures=[event],
        risk=1.2,
        max_tau_ratio=1.1,
        max_vel_ratio=0.7,
        final_ee_error_m=0.04,
        min_manipulability=0.02,
        trajectory_path="episode_ep_017.json",
    )
    evidence = Evidence(
        claim="joint torque limit",
        value=87.0,
        unit="N·m",
        source_url="https://mock.invalid/robot",
        quote="[mock] torque limit is 87 N·m",
        verified=True,
    )
    hardware = HardwareSpec(
        robot="Franka Emika Panda",
        joint_torque_limits_nm=[evidence],
        joint_velocity_limits=[evidence],
        joint_position_limits=[evidence],
        error_taxonomy=[evidence],
    )
    envelope = SafeEnvelope(
        robot=hardware.robot,
        version="mock-0",
        limits={"max_payload_kg": 2.0},
        controller={"kp_scale": [0.7, 1.2]},
        domain_randomization_bounds={"latency_ticks": [0, 6]},
        evidence=[evidence],
        verification={"n_scenarios": 4},
    )
    frame = ReplayFrame(
        t=0.0,
        links=[(0.0, 0.0, float(index)) for index in range(8)],
        ee=(0.5, 0.0, 0.4),
        tau_ratio=[0.2] * 7,
        vel_ratio=[0.3] * 7,
        flags=[],
    )
    replay = Replay(
        episode_id="ep_017",
        fps=30,
        frames=[frame],
        events=[event],
        source="mock",
    )
    campaign = CampaignResult(
        robot=hardware.robot,
        budget=2,
        strategy="nemotron",
        seed=17,
        episodes=[episode],
        events=[
            CampaignEvent(
                seq=1,
                ts=datetime(2026, 10, 18, tzinfo=UTC),
                type="campaign_started",
                data={
                    "run_id": "schema-test",
                    "robot": hardware.robot,
                    "strategy": "random",
                    "seed": 17,
                    "budget": 1,
                    "started_at": datetime(2026, 10, 18, tzinfo=UTC),
                    "baseline_risk": None,
                    "baseline_max_tau_ratio": None,
                    "baseline_max_vel_ratio": None,
                    "baseline_final_ee_error_m": None,
                    "baseline_error": None,
                },
            )
        ],
        source="mock",
    )
    benchmark = BenchmarkResult.model_validate(
        {
            "budget": 1,
            "seed_count": 1,
            "strategies": [
                {
                    "strategy": "nemotron",
                    "sims_to_first_failure": {
                        "mean": 1.0,
                        "std": 0.0,
                        "per_seed": [1],
                        "seeds_with_failure": 1,
                        "mean_is_lower_bound": False,
                    },
                    "cumulative_failures": [1.0],
                    "unique_failure_kinds": {
                        "kinds": ["torque_saturation"],
                        "mean": 1.0,
                        "std": 0.0,
                    },
                    "best_risk": {"mean": 1.2, "std": 0.0},
                }
            ],
            "source": "mock",
        }
    )
    return [
        (ScenarioParams, params),
        (FailureEvent, event),
        (EpisodeResult, episode),
        (Evidence, evidence),
        (HardwareSpec, hardware),
        (SafeEnvelope, envelope),
        (ReplayFrame, frame),
        (Replay, replay),
        (CampaignResult, campaign),
        (BenchmarkResult, benchmark),
    ]


@pytest.mark.parametrize(("field", "minimum", "maximum", "is_int"), SCENARIO_BOUNDS)
def test_scenario_params_accepts_bounds_and_rejects_values_outside(
    field: str, minimum: int | float, maximum: int | float, is_int: bool
) -> None:
    for value in (minimum, maximum):
        ScenarioParams.model_validate({field: value})

    outside = (minimum - 1, maximum + 1) if is_int else (minimum - 0.001, maximum + 0.001)
    for value in outside:
        with pytest.raises(ValidationError):
            ScenarioParams.model_validate({field: value})


def test_scenario_params_defaults() -> None:
    assert ScenarioParams().model_dump() == {
        "payload_kg": 0.0,
        "link_mass_scale": 1.0,
        "joint_damping": 0.0,
        "latency_ticks": 0,
        "sensor_noise_rad": 0.0,
        "torque_limit_scale": 1.0,
        "speed_scale": 1.0,
        "kp_scale": 1.0,
        "push_force_n": 0.0,
        "push_time_s": 2.0,
        "seed": 0,
    }
    assert FailureEvent(kind="collision", t=0.0, severity=0.0).detail == {}


def test_failure_event_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        FailureEvent.model_validate({"kind": "unknown", "t": 0.0, "severity": 1.0})


def test_failure_event_detail_is_not_shared() -> None:
    first = FailureEvent(kind="collision", t=0.0, severity=0.0)
    second = FailureEvent(kind="collision", t=0.0, severity=0.0)

    first.detail["contact"] = "table"

    assert second.detail == {}


@given(data=st.data())
def test_scenario_params_property_bounds(data: st.DataObject) -> None:
    values: dict[str, int | float] = {}
    for field, minimum, maximum, is_int in SCENARIO_BOUNDS:
        if is_int:
            values[field] = data.draw(st.integers(min_value=int(minimum), max_value=int(maximum)))
        else:
            values[field] = data.draw(
                st.floats(
                    min_value=float(minimum),
                    max_value=float(maximum),
                    allow_nan=False,
                    allow_infinity=False,
                )
            )

    ScenarioParams.model_validate(values)

    for field, minimum, maximum, is_int in SCENARIO_BOUNDS:
        delta = 1 if is_int else 0.001
        for value in (minimum - delta, maximum + delta):
            invalid = dict(values)
            invalid[field] = value
            with pytest.raises(ValidationError):
                ScenarioParams.model_validate(invalid)


@pytest.mark.parametrize(("model_type", "model"), sample_models())
def test_models_round_trip_json(model_type: type[BaseModel], model: BaseModel) -> None:
    restored = model_type.model_validate_json(model.model_dump_json())
    assert restored == model


def test_replay_frame_is_nested_in_replay() -> None:
    frame = sample_models()[6][1]
    replay = sample_models()[7][1]

    assert isinstance(frame, ReplayFrame)
    assert isinstance(replay, Replay)
    assert replay.frames == [frame]


@pytest.mark.parametrize("invalid", ["curve", "per_seed", "duplicate_strategy"])
def test_benchmark_result_rejects_inconsistent_metrics(invalid: str) -> None:
    benchmark = sample_models()[-1][1]
    assert isinstance(benchmark, BenchmarkResult)
    payload: dict[str, Any] = benchmark.model_dump()

    if invalid == "curve":
        payload["strategies"][0]["cumulative_failures"] = []
    elif invalid == "per_seed":
        payload["strategies"][0]["sims_to_first_failure"]["per_seed"] = []
    else:
        payload["strategies"].append(payload["strategies"][0])

    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(payload)
