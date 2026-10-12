import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "urdf_limits.json"


@dataclass(frozen=True, slots=True)
class UrdfLimits:
    effort: NDArray[np.float64]
    velocity: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]


def load_urdf_limits(path: Path = DATA_PATH) -> UrdfLimits:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != 7 or [row["joint"] for row in rows] != list(range(7)):
        raise ValueError(f"expected seven Panda arm joint limits in {path}")

    limits = UrdfLimits(
        effort=np.asarray([row["effort"] for row in rows], dtype=np.float64),
        velocity=np.asarray([row["velocity"] for row in rows], dtype=np.float64),
        lower=np.asarray([row["lower"] for row in rows], dtype=np.float64),
        upper=np.asarray([row["upper"] for row in rows], dtype=np.float64),
    )
    if np.any(limits.effort <= 0) or np.any(limits.velocity <= 0):
        raise ValueError(f"Panda effort and velocity limits must be positive in {path}")
    if np.any(limits.lower >= limits.upper):
        raise ValueError(f"Panda position limits are invalid in {path}")
    return limits
