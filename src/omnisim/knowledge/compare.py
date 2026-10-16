import math
import re
import unicodedata
from typing import Literal, cast

from omnisim.knowledge.verifier import verified_only
from omnisim.schemas import Evidence, HardwareSpec, Mismatch
from omnisim.sim.limits import UrdfLimits, load_urdf_limits

_CLAIM = re.compile(r"^joint ([1-7]) (torque|velocity|lower|upper)$")


def _unit_factor(unit: str | None, field: str) -> float | None:
    if unit is None:
        return None
    value = unicodedata.normalize("NFKC", unit).casefold().replace("·", " ").replace("°", "deg")
    value = " ".join(value.replace("per", "/").split())
    if field in {"torque", "velocity"} and value in {
        "nm",
        "n m",
        "n*m",
        "newton metre",
        "newton meter",
    }:
        return 1.0 if field == "torque" else None
    if field == "velocity":
        if value in {"rad/s", "radian/s", "radians/s", "rad s-1", "rad s^-1"}:
            return 1.0
        if value in {"deg/s", "degree/s", "degrees/s"}:
            return math.pi / 180.0
    if field in {"lower", "upper"}:
        if value in {"rad", "radian", "radians"}:
            return 1.0
        if value in {"deg", "degree", "degrees"}:
            return math.pi / 180.0
    return None


def _urdf_value(limits: UrdfLimits, field: str, joint: int) -> float:
    if field == "torque":
        return float(limits.effort[joint])
    if field == "velocity":
        return float(limits.velocity[joint])
    if field == "lower":
        return float(limits.lower[joint])
    return float(limits.upper[joint])


def _more_permissive(field: str, urdf: float, doc: float) -> bool:
    if field in {"torque", "velocity", "upper"}:
        return urdf > doc
    return urdf < doc


def compare_with_urdf(
    spec: HardwareSpec,
    limits: UrdfLimits | None = None,
    *,
    relative_tolerance: float = 1e-3,
) -> tuple[list[Mismatch], list[Evidence]]:
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be non-negative")
    limits = load_urdf_limits() if limits is None else limits
    verified = verified_only(spec)
    items = (
        *verified.joint_torque_limits_nm,
        *verified.joint_velocity_limits,
        *verified.joint_position_limits,
    )
    mismatches: list[Mismatch] = []
    not_compared: list[Evidence] = []
    for ev in items:
        claim = _CLAIM.fullmatch(ev.claim)
        if claim is None or not isinstance(ev.value, (int, float)):
            not_compared.append(ev)
            continue
        joint = int(claim.group(1)) - 1
        field = cast(Literal["torque", "velocity", "lower", "upper"], claim.group(2))
        factor = _unit_factor(ev.unit, field)
        if factor is None:
            not_compared.append(ev)
            continue
        doc = float(ev.value) * factor
        urdf = _urdf_value(limits, field, joint)
        if not math.isfinite(doc) or not math.isfinite(urdf):
            not_compared.append(ev)
            continue
        scale = max(abs(doc), abs(urdf), 1e-12)
        if abs(doc - urdf) <= relative_tolerance * scale:
            continue
        mismatches.append(
            Mismatch(
                joint=joint,
                field=field,
                urdf=urdf,
                doc=doc,
                source_url=ev.source_url,
                sim_more_permissive=_more_permissive(field, urdf, doc),
            )
        )
    return mismatches, not_compared
