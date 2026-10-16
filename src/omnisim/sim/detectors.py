from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import NDArray

from omnisim.schemas import FailureEvent, FailureKind
from omnisim.sim.constants import CONTROL_HZ
from omnisim.sim.limits import UrdfLimits
from omnisim.sim.telemetry import EpisodeTelemetry

_MAX_FLOAT = float(np.finfo(np.float64).max)


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    tau_ratio_threshold: float = 0.95
    tau_min_ticks: int = 5
    vel_ratio_threshold: float = 1.0
    joint_margin_rad: float = 0.02
    tracking_error_m: float = 0.05
    settle_time_s: float = 0.5
    cond_threshold: float = 1e3
    risk_cap: float = 10.0
    divergence_vel_ratio: float = 10.0

    def __post_init__(self) -> None:
        if self.tau_ratio_threshold <= 0 or self.tau_min_ticks < 1:
            raise ValueError("torque threshold and tick count must be positive")
        if self.vel_ratio_threshold <= 0 or self.joint_margin_rad <= 0:
            raise ValueError("velocity threshold and joint margin must be positive")
        if self.tracking_error_m <= 0 or self.settle_time_s < 0:
            raise ValueError("tracking threshold must be positive and settle time non-negative")
        if self.cond_threshold <= 0 or self.risk_cap < 1:
            raise ValueError("condition threshold must be positive and risk cap at least one")
        if self.divergence_vel_ratio <= 0:
            raise ValueError("divergence velocity ratio must be positive")


@dataclass(slots=True)
class _Analysis:
    masks: dict[FailureKind, NDArray[np.bool_]]
    terms: dict[FailureKind, float]
    joint_terms: dict[FailureKind, NDArray[np.float64]]
    events: list[FailureEvent]
    metrics: dict[str, float]


def _scaled(value: float, threshold: float) -> float:
    if threshold < 1 and value > _MAX_FLOAT * threshold:
        return float(_MAX_FLOAT)
    return value / threshold


def _finite_extreme(values: NDArray[np.float64], *, maximum: bool) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.max(finite) if maximum else np.min(finite))


def _row_finite(values: NDArray[np.float64]) -> NDArray[np.bool_]:
    if values.ndim == 1:
        return np.isfinite(values)
    return np.all(np.isfinite(values), axis=tuple(range(1, values.ndim)))


def _analyze(telemetry: EpisodeTelemetry, limits: UrdfLimits, cfg: DetectorConfig) -> _Analysis:
    arrays = {
        item.name: np.asarray(getattr(telemetry, item.name), dtype=np.float64)
        for item in fields(telemetry)
    }
    tick_count = max((array.shape[0] for array in arrays.values() if array.ndim > 0), default=0)
    names: tuple[FailureKind, ...] = (
        "torque_saturation",
        "joint_limit",
        "velocity_limit",
        "tracking_error",
        "singularity",
        "collision",
        "instability",
    )
    masks: dict[FailureKind, NDArray[np.bool_]] = {
        name: np.zeros(tick_count, dtype=np.bool_) for name in names
    }
    joint_terms: dict[FailureKind, NDArray[np.float64]] = {
        name: np.zeros(7, dtype=np.float64)
        for name in ("torque_saturation", "joint_limit", "velocity_limit")
    }
    joint_masks = {
        name: np.zeros((tick_count, 7), dtype=np.bool_)
        for name in ("torque_saturation", "joint_limit", "velocity_limit")
    }
    derived_nonfinite = np.zeros(tick_count, dtype=np.bool_)

    nonfinite = np.zeros(tick_count, dtype=np.bool_)
    for array in arrays.values():
        if array.ndim == 0:
            continue
        bad = ~_row_finite(array)
        length = min(tick_count, len(bad))
        nonfinite[:length] |= bad[:length]

    tau_ratio = arrays["tau_ratio"]
    if tau_ratio.ndim == 2:
        for joint in range(min(7, tau_ratio.shape[1])):
            values = tau_ratio[:, joint]
            if len(values) >= cfg.tau_min_ticks:
                for start in range(len(values) - cfg.tau_min_ticks + 1):
                    window = values[start : start + cfg.tau_min_ticks]
                    if not np.all(np.isfinite(window)):
                        continue
                    term = _scaled(float(np.min(window)), cfg.tau_ratio_threshold)
                    joint_terms["torque_saturation"][joint] = max(
                        joint_terms["torque_saturation"][joint], term
                    )
                    if np.min(window) >= cfg.tau_ratio_threshold:
                        joint_masks["torque_saturation"][
                            start : start + cfg.tau_min_ticks, joint
                        ] = True

    vel_ratio = arrays["vel_ratio"]
    if vel_ratio.ndim == 2:
        for joint in range(min(7, vel_ratio.shape[1])):
            values = vel_ratio[:, joint]
            finite = np.isfinite(values)
            if finite.any():
                joint_terms["velocity_limit"][joint] = _scaled(
                    float(np.max(values[finite])), cfg.vel_ratio_threshold
                )
                length = min(tick_count, len(values))
                joint_masks["velocity_limit"][:length, joint] = finite[:length] & (
                    values[:length] >= cfg.vel_ratio_threshold
                )
                instability_length = min(tick_count, len(values))
                masks["instability"][:instability_length] |= finite[:instability_length] & (
                    values[:instability_length] >= cfg.divergence_vel_ratio
                )

    q = arrays["q"]
    if q.ndim == 2:
        for joint in range(min(7, q.shape[1])):
            values = q[:, joint]
            margins = np.minimum(values - limits.lower[joint], limits.upper[joint] - values)
            length = min(tick_count, len(margins))
            derived_nonfinite[:length] |= ~np.isfinite(margins[:length])
            finite = np.isfinite(margins)
            if finite.any():
                min_margin = float(np.min(margins[finite]))
                joint_terms["joint_limit"][joint] = _scaled(
                    cfg.joint_margin_rad, max(min_margin, 1e-6)
                )
                joint_masks["joint_limit"][:length, joint] = finite[:length] & (
                    (margins[:length] <= cfg.joint_margin_rad)
                    | np.isclose(margins[:length], cfg.joint_margin_rad, rtol=1e-12, atol=0.0)
                )

    ee = arrays["ee"]
    ee_target = arrays["ee_target"]
    tracking_errors = np.zeros(tick_count, dtype=np.float64)
    tracking_valid = np.zeros(tick_count, dtype=np.bool_)
    if ee.ndim == 2 and ee_target.ndim == 2:
        length = min(tick_count, len(ee), len(ee_target))
        pair = ee[:length] - ee_target[:length]
        derived_nonfinite[:length] |= ~_row_finite(pair)
        row_finite = _row_finite(ee[:length]) & _row_finite(ee_target[:length])
        row_finite &= _row_finite(pair)
        if pair.shape[1] >= 3:
            norms = np.hypot(np.hypot(pair[:, 0], pair[:, 1]), pair[:, 2])
            derived_nonfinite[:length] |= row_finite & ~np.isfinite(norms)
            times = arrays["t"]
            for tick in range(length):
                if tick < len(times):
                    if not np.isfinite(times[tick]):
                        continue
                    time_s = times[tick]
                else:
                    time_s = tick / CONTROL_HZ
                if row_finite[tick] and time_s >= cfg.settle_time_s:
                    tracking_valid[tick] = True
                    tracking_errors[tick] = norms[tick]
                    if norms[tick] >= cfg.tracking_error_m:
                        masks["tracking_error"][tick] = True

    jac_cond = arrays["jac_cond"]
    jac_finite = np.isfinite(jac_cond)
    if jac_cond.ndim == 1:
        length = min(tick_count, len(jac_cond))
        masks["singularity"][:length] = jac_finite[:length] & (
            jac_cond[:length] >= cfg.cond_threshold
        )

    contacts = arrays["n_contacts"]
    if contacts.ndim == 1:
        length = min(tick_count, len(contacts))
        masks["collision"][:length] = np.isfinite(contacts[:length]) & (contacts[:length] > 0)

    masks["torque_saturation"] = np.any(joint_masks["torque_saturation"], axis=1)
    masks["velocity_limit"] = np.any(joint_masks["velocity_limit"], axis=1)
    masks["joint_limit"] = np.any(joint_masks["joint_limit"], axis=1)
    masks["instability"] |= nonfinite | derived_nonfinite

    terms: dict[FailureKind, float] = {
        "torque_saturation": float(np.max(joint_terms["torque_saturation"])),
        "joint_limit": float(np.max(joint_terms["joint_limit"])),
        "velocity_limit": float(np.max(joint_terms["velocity_limit"])),
        "tracking_error": _scaled(
            _finite_extreme(tracking_errors[tracking_valid], maximum=True), cfg.tracking_error_m
        ),
        "singularity": _scaled(_finite_extreme(jac_cond, maximum=True), cfg.cond_threshold),
        "collision": float(masks["collision"].any()),
        "instability": float(masks["instability"].any()),
    }
    for kind, mask in masks.items():
        if mask.any():
            terms[kind] = max(terms[kind], 1.0)

    events: list[FailureEvent] = []
    for joint in range(7):
        if joint_masks["torque_saturation"][:, joint].any():
            tick = int(np.flatnonzero(joint_masks["torque_saturation"][:, joint])[0])
            peak_ratio = joint_terms["torque_saturation"][joint] * cfg.tau_ratio_threshold
            events.append(
                FailureEvent(
                    kind="torque_saturation",
                    t=tick / CONTROL_HZ,
                    joint=joint,
                    severity=max(1.0, float(joint_terms["torque_saturation"][joint])),
                    detail={"peak_tau_ratio": float(peak_ratio)},
                )
            )
        if joint_masks["joint_limit"][:, joint].any():
            tick = int(np.flatnonzero(joint_masks["joint_limit"][:, joint])[0])
            events.append(
                FailureEvent(
                    kind="joint_limit",
                    t=tick / CONTROL_HZ,
                    joint=joint,
                    severity=max(1.0, float(joint_terms["joint_limit"][joint])),
                    detail={
                        "min_joint_margin_rad": _finite_extreme(
                            np.minimum(
                                q[:, joint] - limits.lower[joint], limits.upper[joint] - q[:, joint]
                            ),
                            maximum=False,
                        )
                    },
                )
            )
        if joint_masks["velocity_limit"][:, joint].any():
            tick = int(np.flatnonzero(joint_masks["velocity_limit"][:, joint])[0])
            values = vel_ratio[:, joint]
            events.append(
                FailureEvent(
                    kind="velocity_limit",
                    t=tick / CONTROL_HZ,
                    joint=joint,
                    severity=max(1.0, float(joint_terms["velocity_limit"][joint])),
                    detail={"max_vel_ratio": _finite_extreme(values, maximum=True)},
                )
            )

    if masks["tracking_error"].any():
        tick = int(np.flatnonzero(masks["tracking_error"])[0])
        peak_error = _finite_extreme(tracking_errors[tracking_valid], maximum=True)
        events.append(
            FailureEvent(
                kind="tracking_error",
                t=tick / CONTROL_HZ,
                severity=max(1.0, _scaled(peak_error, cfg.tracking_error_m)),
                detail={"max_tracking_error_m": peak_error},
            )
        )
    if masks["singularity"].any():
        tick = int(np.flatnonzero(masks["singularity"])[0])
        peak_cond = _finite_extreme(jac_cond, maximum=True)
        events.append(
            FailureEvent(
                kind="singularity",
                t=tick / CONTROL_HZ,
                severity=max(1.0, _scaled(peak_cond, cfg.cond_threshold)),
                detail={"max_cond": peak_cond},
            )
        )
    for kind in ("collision", "instability"):
        if masks[kind].any():
            tick = int(np.flatnonzero(masks[kind])[0])
            detail = {"non_finite": float(nonfinite[tick] or derived_nonfinite[tick])}
            if kind == "instability" and vel_ratio.ndim == 2:
                detail["max_vel_ratio"] = _finite_extreme(vel_ratio[tick], maximum=True)
            events.append(FailureEvent(kind=kind, t=tick / CONTROL_HZ, severity=1.0, detail=detail))

    q_margin = np.minimum(q - limits.lower, limits.upper - q) if q.ndim == 2 else np.empty(0)
    final_errors = np.zeros(0, dtype=np.float64)
    if ee.ndim == 2 and ee_target.ndim == 2:
        length = min(len(ee), len(ee_target))
        pairs = ee[:length] - ee_target[:length]
        finite = _row_finite(ee[:length]) & _row_finite(ee_target[:length]) & _row_finite(pairs)
        if pairs.shape[1] >= 3:
            final_errors = np.hypot(np.hypot(pairs[finite, 0], pairs[finite, 1]), pairs[finite, 2])
    manipulability = arrays["manipulability"]
    metrics = {
        "max_tau_ratio": _finite_extreme(tau_ratio, maximum=True),
        "max_vel_ratio": _finite_extreme(vel_ratio, maximum=True),
        "final_ee_error_m": float(final_errors[-1]) if final_errors.size else 0.0,
        "min_manipulability": _finite_extreme(manipulability, maximum=False),
        "max_cond": _finite_extreme(jac_cond, maximum=True),
        "min_joint_margin_rad": _finite_extreme(q_margin, maximum=False),
        "max_tracking_error_m": _finite_extreme(tracking_errors[tracking_valid], maximum=True),
    }
    metrics.update({f"risk_{kind}": value for kind, value in terms.items()})
    return _Analysis(masks, terms, joint_terms, events, metrics)


def violation_masks(
    telemetry: EpisodeTelemetry, limits: UrdfLimits, cfg: DetectorConfig
) -> dict[FailureKind, NDArray[np.bool_]]:
    return _analyze(telemetry, limits, cfg).masks


def detect(
    telemetry: EpisodeTelemetry, limits: UrdfLimits, cfg: DetectorConfig
) -> tuple[list[FailureEvent], float, dict[str, float]]:
    analysis = _analyze(telemetry, limits, cfg)
    risk = min(max(analysis.terms.values(), default=0.0), cfg.risk_cap)
    return analysis.events, float(risk), analysis.metrics
