from typing import Final

HOME_Q: Final[tuple[float, ...]] = (0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.785)
CONTROL_HZ: Final = 60
TICK_MS: Final = 1000.0 / CONTROL_HZ
PHYSICS_HZ: Final = 240
CONTROL_SUBSTEPS: Final = PHYSICS_HZ // CONTROL_HZ
EPISODE_TICKS: Final = 360
BASE_MOTION_SECONDS: Final = 2.5
CORRECTION_GAIN: Final = 0.25
ARM_JOINTS: Final[tuple[int, ...]] = tuple(range(7))
MOVABLE_JOINTS: Final[tuple[int, ...]] = (*ARM_JOINTS, 9, 10)
HAND_LINK: Final = 8
EE_LINK: Final = 11
