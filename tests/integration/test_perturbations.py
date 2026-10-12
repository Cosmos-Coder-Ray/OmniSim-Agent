from collections.abc import Callable

import numpy as np
import pybullet as p
import pytest

from omnisim.schemas import ScenarioParams
from omnisim.sim.limits import UrdfLimits
from omnisim.sim.panda_env import PandaSim
from omnisim.sim.telemetry import EpisodeTelemetry

PERTURBATIONS = (
    ("payload_kg", 3.0, {}),
    ("link_mass_scale", 1.5, {}),
    ("joint_damping", 5.0, {}),
    ("latency_ticks", 12, {}),
    ("sensor_noise_rad", 0.02, {}),
    ("torque_limit_scale", 0.5, {}),
    ("speed_scale", 2.0, {}),
    ("kp_scale", 2.0, {}),
    ("push_force_n", 40.0, {}),
    ("push_time_s", 3.0, {"push_force_n": 40.0}),
    ("seed", 18, {"sensor_noise_rad": 0.02}),
)


@pytest.mark.parametrize(("field", "value", "background"), PERTURBATIONS)
def test_perturbation_changes_telemetry(
    simulate: Callable[[ScenarioParams], EpisodeTelemetry],
    field: str,
    value: float | int,
    background: dict[str, float],
) -> None:
    baseline = simulate(ScenarioParams.model_validate({"seed": 17, **background}))
    changed = simulate(ScenarioParams.model_validate({"seed": 17, **background, field: value}))
    measured_fields = ("q_meas", "q", "dq", "tau", "ee", "tau_ratio", "vel_ratio")

    assert any(
        not np.array_equal(getattr(baseline, name), getattr(changed, name))
        for name in measured_fields
    )


def test_close_is_idempotent_and_disconnects_client(urdf_limits: UrdfLimits) -> None:
    sim = PandaSim(ScenarioParams(seed=17), urdf_limits.effort, urdf_limits.velocity)
    client_id = sim.client_id
    sim.close()
    sim.close()

    assert client_id is not None
    assert not p.isConnected(physicsClientId=client_id)
