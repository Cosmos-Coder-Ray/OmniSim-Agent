import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnisim.sim.detectors import DetectorConfig, detect, violation_masks
from omnisim.sim.limits import UrdfLimits
from omnisim.sim.telemetry import EpisodeTelemetry


def make_telemetry(ticks: int = 8) -> EpisodeTelemetry:
    return EpisodeTelemetry(
        t=np.arange(1, ticks + 1, dtype=np.float64) / 60,
        q=np.zeros((ticks, 7)),
        q_meas=np.zeros((ticks, 7)),
        dq=np.zeros((ticks, 7)),
        tau=np.zeros((ticks, 7)),
        q_ref=np.zeros((ticks, 7)),
        ee_target=np.zeros((ticks, 3)),
        ee=np.zeros((ticks, 3)),
        tracking_error=np.zeros((ticks, 3)),
        tau_ratio=np.zeros((ticks, 7)),
        vel_ratio=np.zeros((ticks, 7)),
        link_positions=np.zeros((ticks, 8, 3)),
        jac_cond=np.ones(ticks),
        manipulability=np.ones(ticks),
        n_contacts=np.zeros(ticks),
        joint_margin_rad=np.ones((ticks, 7)),
    )


def make_limits() -> UrdfLimits:
    return UrdfLimits(
        effort=np.ones(7),
        velocity=np.ones(7),
        lower=-np.ones(7),
        upper=np.ones(7),
    )


def test_nominal_telemetry_has_no_events() -> None:
    events, risk, metrics = detect(make_telemetry(), make_limits(), DetectorConfig())

    assert events == []
    assert risk < 1.0
    assert metrics["risk_torque_saturation"] == 0.0


@pytest.mark.parametrize(("ticks", "fires"), [(4, False), (5, True)])
def test_torque_saturation_requires_five_consecutive_ticks(ticks: int, fires: bool) -> None:
    telemetry = make_telemetry(ticks)
    telemetry.tau_ratio[:] = 0.95

    masks = violation_masks(telemetry, make_limits(), DetectorConfig())
    events, risk, _ = detect(telemetry, make_limits(), DetectorConfig())

    assert bool(masks["torque_saturation"].any()) is fires
    assert bool(events) is fires
    assert (risk >= 1.0) is fires
    if fires:
        assert events[0].kind == "torque_saturation"
        assert events[0].joint == 0
        assert events[0].t == 0.0
        assert events[0].severity == 1.0


@pytest.mark.parametrize(("ratio", "fires"), [(0.949, False), (0.95, True), (0.951, True)])
def test_torque_saturation_boundary(ratio: float, fires: bool) -> None:
    telemetry = make_telemetry(5)
    telemetry.tau_ratio[:, 0] = ratio

    events, risk, _ = detect(telemetry, make_limits(), DetectorConfig())

    assert any(event.kind == "torque_saturation" for event in events) is fires
    assert (risk >= 1.0) is fires


def test_torque_gap_resets_sustained_count() -> None:
    telemetry = make_telemetry(9)
    telemetry.tau_ratio[:, 0] = 0.95
    telemetry.tau_ratio[4, 0] = 0.949

    masks = violation_masks(telemetry, make_limits(), DetectorConfig())

    assert not masks["torque_saturation"].any()

    telemetry.tau_ratio[4, 0] = 0.95
    masks = violation_masks(telemetry, make_limits(), DetectorConfig())
    assert masks["torque_saturation"].all()


@pytest.mark.parametrize(("ratio", "fires"), [(0.999, False), (1.0, True), (1.001, True)])
def test_velocity_limit_uses_inclusive_boundary(ratio: float, fires: bool) -> None:
    telemetry = make_telemetry(1)
    telemetry.vel_ratio[0, 2] = ratio

    events, risk, _ = detect(telemetry, make_limits(), DetectorConfig())

    assert any(event.kind == "velocity_limit" for event in events) is fires
    assert (risk >= 1.0) is fires


@pytest.mark.parametrize(("distance", "fires"), [(0.020001, False), (0.02, True), (0.019999, True)])
def test_joint_limit_uses_inclusive_margin(distance: float, fires: bool) -> None:
    telemetry = make_telemetry(1)
    telemetry.q[0, 3] = 1.0 - distance

    events, risk, _ = detect(telemetry, make_limits(), DetectorConfig())

    assert any(event.kind == "joint_limit" for event in events) is fires
    assert (risk >= 1.0) is fires


def test_tracking_starts_at_settle_time() -> None:
    telemetry = make_telemetry(31)
    telemetry.ee[28, 0] = 0.1

    events, _, _ = detect(telemetry, make_limits(), DetectorConfig())
    assert not any(event.kind == "tracking_error" for event in events)

    telemetry.ee[29, 0] = 0.05
    events, risk, _ = detect(telemetry, make_limits(), DetectorConfig())
    event = next(event for event in events if event.kind == "tracking_error")
    assert event.t == 29 / 60
    assert event.joint is None
    assert event.severity == 1.0
    assert risk >= 1.0


@pytest.mark.parametrize(("condition", "fires"), [(999.9, False), (1000.0, True), (1000.1, True)])
def test_singularity_uses_high_condition_number(condition: float, fires: bool) -> None:
    telemetry = make_telemetry(1)
    telemetry.jac_cond[0] = condition

    events, risk, _ = detect(telemetry, make_limits(), DetectorConfig())

    assert any(event.kind == "singularity" for event in events) is fires
    assert (risk >= 1.0) is fires


def test_collision_fires_on_first_contact_tick() -> None:
    telemetry = make_telemetry(3)
    telemetry.n_contacts[1] = 1
    telemetry.n_contacts[2] = 1

    events, risk, _ = detect(telemetry, make_limits(), DetectorConfig())

    event = next(event for event in events if event.kind == "collision")
    assert event.t == 1 / 60
    assert event.severity == 1.0
    assert risk >= 1.0


@pytest.mark.parametrize(("error", "fires"), [(0.049999, False), (0.05, True), (0.050001, True)])
def test_tracking_error_boundary(error: float, fires: bool) -> None:
    telemetry = make_telemetry(31)
    telemetry.ee[29, 0] = error

    events, risk, _ = detect(telemetry, make_limits(), DetectorConfig())

    assert any(event.kind == "tracking_error" for event in events) is fires
    assert (risk >= 1.0) is fires


@pytest.mark.parametrize("ratio", [9.999, 10.0, 10.001])
def test_divergence_velocity_guard(ratio: float) -> None:
    telemetry = make_telemetry(1)
    telemetry.vel_ratio[0, 0] = ratio

    events, risk, _ = detect(telemetry, make_limits(), DetectorConfig())

    assert any(event.kind == "instability" for event in events) is (ratio >= 10.0)
    assert risk >= 1.0


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_non_finite_samples_fire_instability_without_poisoning_metrics(invalid: float) -> None:
    telemetry = make_telemetry(3)
    telemetry.q[0, 0] = invalid
    telemetry.vel_ratio[1, 0] = 1.0

    events, risk, metrics = detect(telemetry, make_limits(), DetectorConfig())

    assert any(event.kind == "instability" and event.t == 0.0 for event in events)
    assert any(event.kind == "velocity_limit" for event in events)
    assert np.isfinite(risk)
    assert all(np.isfinite(value) for value in metrics.values())
    assert all(np.isfinite(event.severity) for event in events)
    assert all(np.isfinite(value) for event in events for value in event.detail.values())


def test_empty_and_short_telemetry_return_finite_results() -> None:
    telemetry = make_telemetry(0)
    telemetry.tau_ratio = np.ones((5, 7))
    telemetry.vel_ratio = np.ones((1, 7))

    events, risk, metrics = detect(telemetry, make_limits(), DetectorConfig())

    assert any(event.kind == "torque_saturation" for event in events)
    assert any(event.kind == "velocity_limit" for event in events)
    assert np.isfinite(risk)
    assert all(np.isfinite(value) for value in metrics.values())


@given(
    tau=st.lists(st.floats(min_value=0, max_value=12), max_size=20),
    velocity=st.lists(st.floats(min_value=0, max_value=12), max_size=20),
)
def test_risk_is_finite_and_matches_events(tau: list[float], velocity: list[float]) -> None:
    ticks = max(len(tau), len(velocity))
    telemetry = make_telemetry(ticks)
    telemetry.tau_ratio[:, 0] = 0.0
    telemetry.vel_ratio[:, 0] = 0.0
    telemetry.tau_ratio[: len(tau), 0] = tau
    telemetry.vel_ratio[: len(velocity), 0] = velocity

    events, risk, _ = detect(telemetry, make_limits(), DetectorConfig())

    assert np.isfinite(risk)
    assert 0.0 <= risk <= DetectorConfig().risk_cap
    assert (risk >= 1.0) is bool(events)


@given(
    tau=st.lists(st.floats(min_value=0, max_value=2), min_size=1, max_size=20),
    scale=st.floats(min_value=1, max_value=4),
)
def test_scaling_torque_ratio_never_lowers_risk(tau: list[float], scale: float) -> None:
    before = make_telemetry(len(tau))
    before.tau_ratio[:, 0] = tau
    after = make_telemetry(len(tau))
    after.tau_ratio[:, 0] = np.asarray(tau) * scale

    _, risk_before, _ = detect(before, make_limits(), DetectorConfig())
    _, risk_after, _ = detect(after, make_limits(), DetectorConfig())

    assert risk_after >= risk_before
