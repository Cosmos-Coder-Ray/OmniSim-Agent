from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
from numpy.typing import NDArray

from omnisim.schemas import ScenarioParams
from omnisim.sim.constants import (
    ARM_JOINTS,
    CONTROL_HZ,
    CONTROL_SUBSTEPS,
    CORRECTION_GAIN,
    EE_LINK,
    EPISODE_TICKS,
    HAND_LINK,
    HOME_Q,
    MOVABLE_JOINTS,
    PHYSICS_HZ,
)
from omnisim.sim.perturbations import (
    PUSH_DURATION_S,
    apply_push,
    apply_scenario_dynamics,
    measured_position,
)
from omnisim.sim.policy import CommandDelay, plan_reference
from omnisim.sim.telemetry import EpisodeTelemetry


class PandaSim:
    def __init__(
        self,
        params: ScenarioParams,
        torque_limits: NDArray[np.float64],
        vel_limits: NDArray[np.float64],
    ) -> None:
        self.params = params
        self.torque_limits = np.asarray(torque_limits, dtype=np.float64)
        self.vel_limits = np.asarray(vel_limits, dtype=np.float64)
        if self.torque_limits.shape != (7,) or self.vel_limits.shape != (7,):
            raise ValueError("Panda torque and velocity limits must each contain seven values")
        if np.any(self.torque_limits <= 0) or np.any(self.vel_limits <= 0):
            raise ValueError("Panda torque and velocity limits must be positive")

        self.effective_torque_limits = self.torque_limits * params.torque_limit_scale
        self.rng = np.random.default_rng(params.seed)
        self.client_id: int | None = p.connect(p.DIRECT)
        self.closed = False
        try:
            self._load_world()
        except (ValueError, p.error):
            self.close()
            raise

    def __enter__(self) -> PandaSim:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self.closed:
            return
        if self.client_id is not None:
            p.disconnect(physicsClientId=self.client_id)
        self.closed = True

    def _load_world(self) -> None:
        client_id = self._client()
        p.setPhysicsEngineParameter(
            deterministicOverlappingPairs=1,
            physicsClientId=client_id,
        )
        p.setTimeStep(1.0 / PHYSICS_HZ, physicsClientId=client_id)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=client_id)
        p.loadURDF(
            str(Path(pybullet_data.getDataPath()) / "plane.urdf"),
            physicsClientId=client_id,
        )
        self.robot = p.loadURDF(
            str(Path(pybullet_data.getDataPath()) / "franka_panda" / "panda.urdf"),
            useFixedBase=True,
            physicsClientId=client_id,
        )
        names = {
            HAND_LINK: p.getJointInfo(self.robot, HAND_LINK, physicsClientId=client_id)[
                12
            ].decode(),
            EE_LINK: p.getJointInfo(self.robot, EE_LINK, physicsClientId=client_id)[12].decode(),
        }
        if names != {HAND_LINK: "panda_hand", EE_LINK: "panda_grasptarget"}:
            raise ValueError(f"unexpected Panda hand or end-effector links: {names}")

        self.joint_lower = np.asarray(
            [
                p.getJointInfo(self.robot, joint, physicsClientId=client_id)[8]
                for joint in ARM_JOINTS
            ],
            dtype=np.float64,
        )
        self.joint_upper = np.asarray(
            [
                p.getJointInfo(self.robot, joint, physicsClientId=client_id)[9]
                for joint in ARM_JOINTS
            ],
            dtype=np.float64,
        )
        for joint, position in zip(ARM_JOINTS, HOME_Q, strict=True):
            p.resetJointState(self.robot, joint, position, physicsClientId=client_id)
        for joint in (9, 10):
            p.resetJointState(self.robot, joint, 0.04, physicsClientId=client_id)

        p.setJointMotorControlArray(
            self.robot,
            jointIndices=list(ARM_JOINTS),
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0] * 7,
            physicsClientId=client_id,
        )
        p.setJointMotorControlArray(
            self.robot,
            jointIndices=[9, 10],
            controlMode=p.POSITION_CONTROL,
            targetPositions=[0.04, 0.04],
            targetVelocities=[0.0, 0.0],
            forces=[20.0, 20.0],
            physicsClientId=client_id,
        )
        apply_scenario_dynamics(client_id, self.robot, self.params)

    def run(self, deadline: float | None = None) -> EpisodeTelemetry:
        client_id = self._client()
        reference = plan_reference(self.params)
        telemetry = EpisodeTelemetry.allocate()
        command_delay = CommandDelay(self.params.latency_ticks)
        position_gains = [0.05 * self.params.kp_scale] * 7

        for tick in range(EPISODE_TICKS):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("episode timed out")
            before = p.getJointStates(self.robot, list(ARM_JOINTS), physicsClientId=client_id)
            q_before = np.asarray([state[0] for state in before], dtype=np.float64)
            q_meas = measured_position(q_before, self.params, self.rng)
            q_ref = reference.q_ref[tick]
            q_des = q_ref + CORRECTION_GAIN * (q_ref - q_meas)
            command = command_delay.push(q_des)
            p.setJointMotorControlArray(
                self.robot,
                jointIndices=list(ARM_JOINTS),
                controlMode=p.POSITION_CONTROL,
                targetPositions=command.tolist(),
                targetVelocities=[0.0] * 7,
                forces=self.effective_torque_limits.tolist(),
                positionGains=position_gains,
                physicsClientId=client_id,
            )

            for substep in range(CONTROL_SUBSTEPS):
                sim_time = (tick * CONTROL_SUBSTEPS + substep) / PHYSICS_HZ
                if (
                    self.params.push_force_n > 0
                    and self.params.push_time_s
                    <= sim_time
                    < self.params.push_time_s + PUSH_DURATION_S
                ):
                    apply_push(client_id, self.robot, self.params.push_force_n)
                p.stepSimulation(physicsClientId=client_id)

            after = p.getJointStates(self.robot, list(ARM_JOINTS), physicsClientId=client_id)
            q = np.asarray([state[0] for state in after], dtype=np.float64)
            dq = np.asarray([state[1] for state in after], dtype=np.float64)
            tau = np.asarray([state[3] for state in after], dtype=np.float64)
            ee_state = p.getLinkState(
                self.robot,
                EE_LINK,
                computeForwardKinematics=1,
                physicsClientId=client_id,
            )
            ee = np.asarray(ee_state[4], dtype=np.float64)
            links = [p.getBasePositionAndOrientation(self.robot, physicsClientId=client_id)[0]]
            links.extend(
                p.getLinkState(
                    self.robot,
                    link,
                    computeForwardKinematics=1,
                    physicsClientId=client_id,
                )[4]
                for link in range(7)
            )

            all_states = p.getJointStates(
                self.robot, list(MOVABLE_JOINTS), physicsClientId=client_id
            )
            jacobian = p.calculateJacobian(
                self.robot,
                EE_LINK,
                [0.0, 0.0, 0.0],
                [state[0] for state in all_states],
                [state[1] for state in all_states],
                [0.0] * len(MOVABLE_JOINTS),
                physicsClientId=client_id,
            )[0]
            arm_jacobian = np.asarray(jacobian, dtype=np.float64)[:, :7]
            contact_links = {
                contact[3]
                for contact in p.getContactPoints(bodyA=self.robot, physicsClientId=client_id)
                if contact[3] >= 0
            }

            telemetry.t[tick] = (tick + 1) / CONTROL_HZ
            telemetry.q[tick] = q
            telemetry.q_meas[tick] = q_meas
            telemetry.dq[tick] = dq
            telemetry.tau[tick] = tau
            telemetry.q_ref[tick] = q_ref
            telemetry.ee_target[tick] = reference.ee_target[tick]
            telemetry.ee[tick] = ee
            telemetry.tracking_error[tick] = reference.ee_target[tick] - ee
            telemetry.tau_ratio[tick] = np.abs(tau) / self.effective_torque_limits
            telemetry.vel_ratio[tick] = np.abs(dq) / self.vel_limits
            telemetry.link_positions[tick] = np.asarray(links, dtype=np.float64)
            telemetry.jac_cond[tick] = np.linalg.cond(arm_jacobian)
            gram = arm_jacobian @ arm_jacobian.T
            telemetry.manipulability[tick] = np.sqrt(max(float(np.linalg.det(gram)), 0.0))
            telemetry.n_contacts[tick] = float(len(contact_links))
            telemetry.joint_margin_rad[tick] = np.minimum(
                q - self.joint_lower, self.joint_upper - q
            )

        return telemetry

    def _client(self) -> int:
        if self.closed or self.client_id is None:
            raise RuntimeError("PandaSim is closed")
        return self.client_id
