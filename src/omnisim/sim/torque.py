from __future__ import annotations

from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

from omnisim.schemas import ScenarioParams
from omnisim.sim.constants import CONTROL_HZ
from omnisim.sim.limits import UrdfLimits
from omnisim.sim.perturbations import apply_scenario_dynamics
from omnisim.sim.policy import ReferenceTrajectory, plan_reference


class InverseDynamicsPredictor:
    def __init__(self) -> None:
        self.client_id = p.connect(p.DIRECT)
        if self.client_id < 0:
            raise RuntimeError("could not connect to PyBullet")
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1, physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        self._bodies: dict[float, int] = {}
        self._references: dict[float, ReferenceTrajectory] = {}

    def __enter__(self) -> InverseDynamicsPredictor:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        p.disconnect(physicsClientId=self.client_id)

    def __call__(self, payload_kg: float, speed_scale: float, limits: UrdfLimits) -> float:
        body = self._body(payload_kg)
        reference = self._reference(speed_scale)
        q_ref = reference.q_ref
        dt = 1.0 / CONTROL_HZ
        dq_ref = np.gradient(q_ref, dt, axis=0, edge_order=2)
        ddq_ref = np.gradient(dq_ref, dt, axis=0, edge_order=2)
        peak = 0.0
        for q, dq, ddq in zip(q_ref, dq_ref, ddq_ref, strict=True):
            tau = p.calculateInverseDynamics(
                body,
                [*q.tolist(), 0.04, 0.04],
                [*dq.tolist(), 0.0, 0.0],
                [*ddq.tolist(), 0.0, 0.0],
                physicsClientId=self.client_id,
            )[:7]
            peak = max(peak, float(np.max(np.abs(tau) / limits.effort)))
        return peak

    def _body(self, payload_kg: float) -> int:
        if payload_kg not in self._bodies:
            robot = p.loadURDF(
                str(Path(pybullet_data.getDataPath()) / "franka_panda" / "panda.urdf"),
                useFixedBase=True,
                physicsClientId=self.client_id,
            )
            for joint in (9, 10):
                p.resetJointState(robot, joint, 0.04, physicsClientId=self.client_id)
            apply_scenario_dynamics(
                self.client_id,
                robot,
                ScenarioParams(payload_kg=payload_kg),
            )
            self._bodies[payload_kg] = robot
        return self._bodies[payload_kg]

    def _reference(self, speed_scale: float) -> ReferenceTrajectory:
        if speed_scale not in self._references:
            self._references[speed_scale] = plan_reference(ScenarioParams(speed_scale=speed_scale))
        return self._references[speed_scale]
