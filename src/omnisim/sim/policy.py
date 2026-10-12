from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from omnisim.schemas import ScenarioParams
from omnisim.sim.constants import (
    BASE_MOTION_SECONDS,
    CONTROL_HZ,
    EE_LINK,
    EPISODE_TICKS,
    HOME_Q,
    MOVABLE_JOINTS,
)

PICK_XYZ = (0.4, -0.15, 0.2)
APPROACH_Z = 0.28
PLACE_Y = 0.15


@dataclass(frozen=True, slots=True)
class ReferenceTrajectory:
    q_ref: NDArray[np.float64]
    ee_target: NDArray[np.float64]


def min_jerk(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def min_jerk_segment(
    start: NDArray[np.float64], end: NDArray[np.float64], u: float
) -> NDArray[np.float64]:
    return start + min_jerk(u) * (end - start)


def motion_duration(speed_scale: float) -> float:
    return BASE_MOTION_SECONDS / speed_scale


class CommandDelay:
    def __init__(self, latency_ticks: int) -> None:
        home = np.asarray(HOME_Q, dtype=np.float64)
        self._commands = [home.copy() for _ in range(latency_ticks + 1)]
        self._next = 0

    def push(self, command: NDArray[np.float64]) -> NDArray[np.float64]:
        self._commands[self._next] = command.copy()
        self._next = (self._next + 1) % len(self._commands)
        return self._commands[self._next].copy()


def plan_reference(params: ScenarioParams) -> ReferenceTrajectory:
    import pybullet as p
    import pybullet_data

    client_id = p.connect(p.DIRECT)
    try:
        p.setPhysicsEngineParameter(
            deterministicOverlappingPairs=1,
            physicsClientId=client_id,
        )
        robot = p.loadURDF(
            str(Path(pybullet_data.getDataPath()) / "franka_panda" / "panda.urdf"),
            useFixedBase=True,
            physicsClientId=client_id,
        )
        lower = [
            p.getJointInfo(robot, joint, physicsClientId=client_id)[8] for joint in MOVABLE_JOINTS
        ]
        upper = [
            p.getJointInfo(robot, joint, physicsClientId=client_id)[9] for joint in MOVABLE_JOINTS
        ]
        ranges = [high - low for low, high in zip(lower, upper, strict=True)]
        rest = [*HOME_Q, 0.04, 0.04]
        for joint, position in zip(range(7), HOME_Q, strict=True):
            p.resetJointState(robot, joint, position, physicsClientId=client_id)
        p.resetJointState(robot, 9, 0.04, physicsClientId=client_id)
        p.resetJointState(robot, 10, 0.04, physicsClientId=client_id)

        orientation = p.getQuaternionFromEuler([np.pi, 0.0, 0.0])
        waypoints = (
            (PICK_XYZ[0], PICK_XYZ[1], APPROACH_Z),
            PICK_XYZ,
            (PICK_XYZ[0], PICK_XYZ[1], APPROACH_Z),
            (PICK_XYZ[0], PLACE_Y, PICK_XYZ[2]),
        )
        joint_waypoints = [np.asarray(HOME_Q, dtype=np.float64)]
        for target in waypoints:
            solution = p.calculateInverseKinematics(
                robot,
                EE_LINK,
                target,
                orientation,
                lowerLimits=lower,
                upperLimits=upper,
                jointRanges=ranges,
                restPoses=rest,
                maxNumIterations=100,
                residualThreshold=1e-5,
                physicsClientId=client_id,
            )
            joint_waypoints.append(np.asarray(solution[:7], dtype=np.float64))

        duration = motion_duration(params.speed_scale)
        segment_duration = duration / (len(joint_waypoints) - 1)
        q_ref = np.empty((EPISODE_TICKS, 7), dtype=np.float64)
        for tick in range(EPISODE_TICKS):
            t = (tick + 1) / CONTROL_HZ
            if t >= duration:
                q_ref[tick] = joint_waypoints[-1]
                continue
            segment = min(int(t / segment_duration), len(joint_waypoints) - 2)
            u = (t - segment * segment_duration) / segment_duration
            q_ref[tick] = min_jerk_segment(
                joint_waypoints[segment], joint_waypoints[segment + 1], u
            )

        ee_target = np.empty((EPISODE_TICKS, 3), dtype=np.float64)
        for tick, position in enumerate(q_ref):
            for joint, value in zip(range(7), position, strict=True):
                p.resetJointState(robot, joint, float(value), physicsClientId=client_id)
            state = p.getLinkState(
                robot,
                EE_LINK,
                computeForwardKinematics=1,
                physicsClientId=client_id,
            )
            ee_target[tick] = state[4]
        return ReferenceTrajectory(q_ref=q_ref, ee_target=ee_target)
    finally:
        p.disconnect(physicsClientId=client_id)
