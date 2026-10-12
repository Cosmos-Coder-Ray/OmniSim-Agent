import numpy as np
import pybullet as p
from numpy.typing import NDArray

from omnisim.schemas import ScenarioParams
from omnisim.sim.constants import HAND_LINK

PUSH_DURATION_S = 0.1
PAYLOAD_RADIUS_M = 0.05


def apply_scenario_dynamics(client_id: int, robot: int, params: ScenarioParams) -> None:
    for link in range(9):
        dynamics = p.getDynamicsInfo(robot, link, physicsClientId=client_id)
        mass = dynamics[0] * params.link_mass_scale
        inertia = np.asarray(dynamics[2], dtype=np.float64) * params.link_mass_scale
        if link == HAND_LINK and params.payload_kg:
            payload_inertia = 0.4 * params.payload_kg * PAYLOAD_RADIUS_M**2
            mass += params.payload_kg
            inertia += payload_inertia
        p.changeDynamics(
            robot,
            link,
            mass=mass,
            localInertiaDiagonal=inertia.tolist(),
            physicsClientId=client_id,
        )

    if params.joint_damping:
        for joint in range(7):
            p.changeDynamics(
                robot,
                joint,
                jointDamping=params.joint_damping,
                physicsClientId=client_id,
            )


def measured_position(
    q: NDArray[np.float64], params: ScenarioParams, rng: np.random.Generator
) -> NDArray[np.float64]:
    if params.sensor_noise_rad == 0:
        return q.copy()
    return q + rng.normal(0.0, params.sensor_noise_rad, size=q.shape)


def apply_push(client_id: int, robot: int, force_n: float) -> None:
    hand_state = p.getLinkState(
        robot,
        HAND_LINK,
        computeForwardKinematics=1,
        physicsClientId=client_id,
    )
    p.applyExternalForce(
        robot,
        HAND_LINK,
        (0.0, force_n, 0.0),
        hand_state[0],
        p.WORLD_FRAME,
        physicsClientId=client_id,
    )
