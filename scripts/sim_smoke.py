import time

import numpy as np

from omnisim.schemas import ScenarioParams
from omnisim.sim.limits import load_urdf_limits
from omnisim.sim.panda_env import PandaSim
from omnisim.sim.telemetry import EpisodeTelemetry

CASES = (
    ("nominal", {}),
    ("payload3_speed2", {"payload_kg": 3.0, "speed_scale": 2.0}),
    ("latency12", {"latency_ticks": 12}),
    ("torque_scale05", {"torque_limit_scale": 0.5}),
    ("noise_push", {"sensor_noise_rad": 0.02, "push_force_n": 40.0}),
)


def has_nan(telemetry: EpisodeTelemetry) -> bool:
    return any(np.isnan(getattr(telemetry, name)).any() for name in telemetry.__dataclass_fields__)


def main() -> int:
    limits = load_urdf_limits()
    print(
        "label              max_tau_ratio  max_vel_ratio  final_ee_error_m  "
        "min_manipulability  min_link_z  min_joint_margin_rad  wall_s"
    )
    start = time.perf_counter()
    for label, changes in CASES:
        params = ScenarioParams.model_validate({"seed": 17, **changes})
        episode_start = time.perf_counter()
        with PandaSim(params, limits.effort, limits.velocity) as sim:
            telemetry = sim.run()
        elapsed = time.perf_counter() - episode_start
        if has_nan(telemetry):
            print(f"{label}: NaN in telemetry", flush=True)
            return 1

        print(
            f"{label:<18} {np.max(telemetry.tau_ratio):>13.4f} "
            f"{np.max(telemetry.vel_ratio):>14.4f} "
            f"{np.linalg.norm(telemetry.tracking_error[-1]):>17.4f} "
            f"{np.min(telemetry.manipulability):>19.6f} "
            f"{np.min(telemetry.link_positions[:, :, 2]):>11.4f} "
            f"{np.min(telemetry.joint_margin_rad):>22.4f} {elapsed:>7.3f}"
        )
    print(f"total time: {time.perf_counter() - start:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
