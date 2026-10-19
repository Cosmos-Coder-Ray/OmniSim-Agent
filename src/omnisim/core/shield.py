from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from omnisim.schemas import SafeEnvelope, ScenarioParams
from omnisim.sim.constants import BASE_MOTION_SECONDS, TICK_MS
from omnisim.sim.limits import UrdfLimits

MIN_SPEED_SCALE = 0.5
MAX_SPEED_SCALE = 2.0
PAYLOAD_STEP_KG = 0.25
SPEED_STEP = 0.05
TORQUE_HEADROOM = 0.9
MAX_LATENCY_TICKS = 12


@dataclass(frozen=True, slots=True)
class ShieldDecision:
    params: ScenarioParams
    rejected: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EpisodeBlocked:
    episode_id: str
    params: ScenarioParams
    reasons: tuple[str, ...]


class Shield(Protocol):
    def decide(self, params: ScenarioParams) -> ShieldDecision: ...


def _number(mapping: dict, key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"envelope is missing numeric {key}")
    return float(value)


def _payload_speed_cap(table: list, payload_kg: float) -> float:
    for item in table:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or isinstance(item[0], bool)
            or isinstance(item[1], bool)
            or not isinstance(item[0], (int, float))
            or not isinstance(item[1], (int, float))
        ):
            raise ValueError("invalid payload speed-cap table")
        if payload_kg <= float(item[0]):
            return float(item[1])
    raise ValueError("payload speed-cap table does not cover the payload")


class SafetyShield:
    def __init__(self, envelope: SafeEnvelope) -> None:
        self.envelope = envelope

    def decide(self, params: ScenarioParams) -> ShieldDecision:
        limits = self.envelope.limits
        controller = self.envelope.controller
        max_payload = _number(limits, "max_payload_kg")
        max_latency = _number(limits, "max_latency_ms")
        reasons: list[str] = []
        if params.payload_kg > max_payload:
            reasons.append("payload exceeds envelope limit")
        if params.latency_ticks * TICK_MS > max_latency:
            reasons.append("latency exceeds envelope limit")
        if reasons:
            return ShieldDecision(params.model_copy(deep=True), True, tuple(reasons))

        payload_table = limits.get("speed_scale_max_by_payload_kg")
        latency_table = limits.get("speed_scale_max_by_latency_ticks")
        if not isinstance(payload_table, list) or not isinstance(latency_table, list):
            raise ValueError("envelope is missing speed-cap tables")
        if len(latency_table) <= params.latency_ticks:
            raise ValueError("latency speed-cap table does not cover latency_ticks")

        payload_cap = _payload_speed_cap(payload_table, params.payload_kg)
        latency_cap = latency_table[params.latency_ticks]
        if isinstance(latency_cap, bool) or not isinstance(latency_cap, (int, float)):
            raise ValueError("invalid latency speed-cap table")
        speed_cap = min(payload_cap, float(latency_cap))
        if speed_cap < MIN_SPEED_SCALE:
            reasons.append("predicted torque exceeds cap at minimum speed")
        if reasons:
            return ShieldDecision(params.model_copy(deep=True), True, tuple(reasons))

        kp_cap = _number(controller, "kp_scale_max")
        adjusted = params.model_copy(
            update={
                "speed_scale": min(params.speed_scale, speed_cap),
                "kp_scale": min(params.kp_scale, kp_cap),
            },
            deep=True,
        )
        return ShieldDecision(adjusted, False, ())


def build_payload_speed_table(
    max_payload_kg: float,
    torque_utilization_cap: float,
    limits: UrdfLimits,
    *,
    predictor: Callable[[float, float, UrdfLimits], float],
) -> list[list[float]]:
    payloads = [
        index * PAYLOAD_STEP_KG for index in range(int(max_payload_kg // PAYLOAD_STEP_KG) + 1)
    ]
    if not payloads or payloads[-1] < max_payload_kg:
        payloads.append(max_payload_kg)
    payloads = sorted(set(round(value, 6) for value in payloads))

    speeds = [round(MIN_SPEED_SCALE + i * SPEED_STEP, 2) for i in range(31)]
    table: list[list[float]] = []
    previous_cap = MAX_SPEED_SCALE
    for payload in payloads:
        safe_speeds = [
            speed
            for speed in speeds
            if predictor(payload, speed, limits) <= torque_utilization_cap * TORQUE_HEADROOM
        ]
        cap = max(safe_speeds, default=0.0)
        previous_cap = min(previous_cap, cap)
        table.append([payload, previous_cap])
    return table


def build_latency_speed_table() -> list[float]:
    table: list[float] = []
    previous_cap = MAX_SPEED_SCALE
    for ticks in range(MAX_LATENCY_TICKS + 1):
        latency_s = ticks * TICK_MS / 1000.0
        cap = max(
            MIN_SPEED_SCALE,
            MAX_SPEED_SCALE * BASE_MOTION_SECONDS / (BASE_MOTION_SECONDS + latency_s),
        )
        previous_cap = min(previous_cap, cap)
        table.append(round(previous_cap, 6))
    return table
