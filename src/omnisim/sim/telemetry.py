from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from omnisim.schemas import ReplayFrame
from omnisim.sim.constants import EPISODE_TICKS


class DownsampledFrames(TypedDict):
    fps: int
    frames: list[dict[str, object]]


@dataclass(slots=True)
class EpisodeTelemetry:
    t: NDArray[np.float64]
    q: NDArray[np.float64]
    q_meas: NDArray[np.float64]
    dq: NDArray[np.float64]
    tau: NDArray[np.float64]
    q_ref: NDArray[np.float64]
    ee_target: NDArray[np.float64]
    ee: NDArray[np.float64]
    tracking_error: NDArray[np.float64]
    tau_ratio: NDArray[np.float64]
    vel_ratio: NDArray[np.float64]
    link_positions: NDArray[np.float64]
    jac_cond: NDArray[np.float64]
    manipulability: NDArray[np.float64]
    n_contacts: NDArray[np.float64]
    joint_margin_rad: NDArray[np.float64]

    @classmethod
    def allocate(cls) -> EpisodeTelemetry:
        return cls(
            t=np.empty(EPISODE_TICKS, dtype=np.float64),
            q=np.empty((EPISODE_TICKS, 7), dtype=np.float64),
            q_meas=np.empty((EPISODE_TICKS, 7), dtype=np.float64),
            dq=np.empty((EPISODE_TICKS, 7), dtype=np.float64),
            tau=np.empty((EPISODE_TICKS, 7), dtype=np.float64),
            q_ref=np.empty((EPISODE_TICKS, 7), dtype=np.float64),
            ee_target=np.empty((EPISODE_TICKS, 3), dtype=np.float64),
            ee=np.empty((EPISODE_TICKS, 3), dtype=np.float64),
            tracking_error=np.empty((EPISODE_TICKS, 3), dtype=np.float64),
            tau_ratio=np.empty((EPISODE_TICKS, 7), dtype=np.float64),
            vel_ratio=np.empty((EPISODE_TICKS, 7), dtype=np.float64),
            link_positions=np.empty((EPISODE_TICKS, 8, 3), dtype=np.float64),
            jac_cond=np.empty(EPISODE_TICKS, dtype=np.float64),
            manipulability=np.empty(EPISODE_TICKS, dtype=np.float64),
            n_contacts=np.empty(EPISODE_TICKS, dtype=np.float64),
            joint_margin_rad=np.empty((EPISODE_TICKS, 7), dtype=np.float64),
        )

    def downsample(self, fps: int = 30) -> DownsampledFrames:
        if fps <= 0 or 60 % fps != 0:
            raise ValueError("fps must be a positive divisor of 60")

        stride = 60 // fps
        frames = []
        for index in range(0, len(self.t), stride):
            frame = ReplayFrame(
                t=float(self.t[index]),
                links=[tuple(point) for point in self.link_positions[index]],
                ee=tuple(self.ee[index]),
                tau_ratio=self.tau_ratio[index].tolist(),
                vel_ratio=self.vel_ratio[index].tolist(),
                flags=[],
            )
            frames.append(frame.model_dump(mode="json"))
        return {"fps": fps, "frames": frames}
