import time

from omnisim.core.shield import EpisodeBlocked
from omnisim.schemas import EpisodeResult, ScenarioParams
from omnisim.sim.limits import load_urdf_limits
from omnisim.sim.runner import EpisodeError, EpisodeRunner

CASES = (
    ("nominal", {}),
    ("payload3_speed2", {"payload_kg": 3.0, "speed_scale": 2.0}),
    ("latency12", {"latency_ticks": 12}),
    ("torque_scale05", {"torque_limit_scale": 0.5}),
    ("noise_push", {"sensor_noise_rad": 0.02, "push_force_n": 40.0}),
)


def main() -> int:
    limits = load_urdf_limits()
    print(
        "label              risk  failures                     max_tau_ratio  max_vel_ratio  "
        "final_ee_error_m  wall_s"
    )
    start = time.perf_counter()
    with EpisodeRunner() as runner:
        for label, changes in CASES:
            params = ScenarioParams.model_validate({"seed": 17, **changes})
            episode_start = time.perf_counter()
            result = runner.run_batch([params], limits, workers=1)[0]
            elapsed = time.perf_counter() - episode_start
            if isinstance(result, EpisodeError):
                print(f"{label}: {result.error_type}: {result.message}", flush=True)
                return 1
            if isinstance(result, EpisodeBlocked):
                raise RuntimeError("smoke episode unexpectedly returned EpisodeBlocked")

            print_result(label, result, elapsed)
    print(f"total time: {time.perf_counter() - start:.3f}s")
    return 0


def print_result(label: str, result: EpisodeResult, elapsed: float) -> None:
    failure_kinds = ",".join(sorted({event.kind for event in result.failures})) or "none"
    print(
        f"{label:<18} {result.risk:>4.2f}  {failure_kinds:<28} "
        f"{result.max_tau_ratio:>13.4f} {result.max_vel_ratio:>14.4f} "
        f"{result.final_ee_error_m:>17.4f} {elapsed:>7.3f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
