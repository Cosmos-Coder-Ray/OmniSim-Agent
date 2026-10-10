from pathlib import Path

import numpy as np

from omnisim.schemas import (
    BenchmarkResult,
    CampaignEvent,
    CampaignResult,
    EpisodeResult,
    Evidence,
    FailureEvent,
    FailureKind,
    HardwareSpec,
    Replay,
    ReplayFrame,
    SafeEnvelope,
    ScenarioParams,
)

ROOT = Path(__file__).resolve().parents[1]
MOCK_DIR = ROOT / "web" / "public" / "mock"
FAILURE_KINDS = (
    "torque_saturation",
    "joint_limit",
    "velocity_limit",
    "tracking_error",
    "singularity",
    "collision",
    "instability",
)


def mock_evidence(claim: str, value: float | str, unit: str | None, slug: str) -> Evidence:
    return Evidence(
        claim=claim,
        value=value,
        unit=unit,
        source_url=f"https://mock.invalid/franka/{slug}",
        quote=f"[mock] {claim} is listed as {value} {unit or ''}".rstrip(),
        verified=True,
    )


def build_hardware_spec() -> HardwareSpec:
    taxonomy = Evidence(
        claim="error taxonomy",
        value="joint_reflex",
        source_url="https://mock.invalid/franka/error-taxonomy",
        quote="[mock] joint_reflex is a protective stop category",
        verified=False,
    )
    return HardwareSpec(
        robot="Franka Emika Panda",
        joint_torque_limits_nm=[
            mock_evidence("joint torque limit", 87.0, "N·m", "joint-torque-limits")
        ],
        joint_velocity_limits=[
            mock_evidence("joint velocity limit", 2.175, "rad/s", "joint-velocity-limits")
        ],
        joint_position_limits=[
            mock_evidence("joint position range", "[-2.9, 2.9]", "rad", "joint-position-limits")
        ],
        error_taxonomy=[taxonomy],
    )


def build_envelope(hardware: HardwareSpec) -> SafeEnvelope:
    verified = [
        *hardware.joint_torque_limits_nm,
        *hardware.joint_velocity_limits,
        *hardware.joint_position_limits,
    ]
    return SafeEnvelope(
        robot=hardware.robot,
        version="mock-0",
        limits={
            "max_payload_kg": 2.0,
            "speed_scale_max": 1.2,
            "max_latency_ms": 100,
            "torque_utilization_cap": 0.85,
        },
        controller={"kp_scale": [0.7, 1.2], "kd_scale": [0.7, 1.2]},
        domain_randomization_bounds={
            "payload_kg": [0.0, 2.0],
            "latency_ticks": [0, 6],
            "speed_scale": [0.5, 1.2],
        },
        evidence=verified,
        verification={
            "n_scenarios": 20,
            "seed": 17017,
            "fail_rate_before": 0.2,
            "fail_rate_after": 0.0,
        },
    )


def build_replay(rng: np.random.Generator) -> Replay:
    events = [
        FailureEvent(
            kind="torque_saturation",
            t=2.0,
            joint=4,
            severity=1.08,
            detail={"limit_ratio": 1.08, "label": "[mock]"},
        ),
        FailureEvent(
            kind="tracking_error",
            t=4.0,
            severity=0.74,
            detail={"error_m": 0.074, "label": "[mock]"},
        ),
    ]
    flags_by_frame: dict[int, list[FailureKind]] = {
        60: [events[0].kind],
        120: [events[1].kind],
    }
    frames: list[ReplayFrame] = []
    for index in range(180):
        t = index / 30
        phase = index / 179 * np.pi
        jitter = rng.normal(0.0, 0.002, size=(8, 3))
        links = [
            (
                round(
                    0.02 + joint * 0.065 + 0.025 * np.sin(phase + joint * 0.35) + jitter[joint, 0],
                    4,
                ),
                round(0.02 * np.sin(phase * 0.7 + joint * 0.2) + jitter[joint, 1], 4),
                round(
                    0.1 + joint * 0.043 + 0.03 * np.sin(phase + joint * 0.25) + jitter[joint, 2], 4
                ),
            )
            for joint in range(8)
        ]
        tau_ratio = [round(float(value), 3) for value in rng.uniform(0.24, 0.82, size=7)]
        vel_ratio = [round(float(value), 3) for value in rng.uniform(0.2, 0.88, size=7)]
        if index == 60:
            tau_ratio[4] = 1.08
        frames.append(
            ReplayFrame(
                t=round(t, 4),
                links=links,
                ee=links[-1],
                tau_ratio=tau_ratio,
                vel_ratio=vel_ratio,
                flags=flags_by_frame.get(index, []),
            )
        )
    return Replay(
        episode_id="ep_017",
        fps=30,
        frames=frames,
        events=events,
        source="mock",
    )


def build_benchmark(rng: np.random.Generator) -> BenchmarkResult:
    budget = 40
    seed_count = 3
    first_failures: dict[str, list[int | None]] = {
        "nemotron": [9, 15, None],
        "random": [28, None, None],
        "cmaes": [18, 22, 31],
    }
    strategies = []
    for name, per_seed in first_failures.items():
        capped = [budget + 1 if value is None else value for value in per_seed]
        cumulative_runs = np.zeros((seed_count, budget), dtype=float)
        for seed_index, first in enumerate(per_seed):
            count = 0
            for sim_index in range(1, budget + 1):
                if first is not None and sim_index >= first:
                    count += int(rng.random() < 0.32)
                    if sim_index == first:
                        count = max(count, 1)
                cumulative_runs[seed_index, sim_index - 1] = count

        kinds_by_seed = [
            set(rng.choice(FAILURE_KINDS, size=int(rng.integers(0, 4)), replace=False).tolist())
            if first is not None
            else set()
            for first in per_seed
        ]
        kinds = [kind for kind in FAILURE_KINDS if any(kind in found for found in kinds_by_seed)]
        kind_counts = [len(found) for found in kinds_by_seed]
        risks = rng.uniform(0.62, 1.28, size=seed_count)
        strategies.append(
            {
                "strategy": name,
                "sims_to_first_failure": {
                    "mean": round(float(np.mean(capped)), 3),
                    "std": round(float(np.std(capped, ddof=0)), 3),
                    "per_seed": per_seed,
                    "seeds_with_failure": sum(value is not None for value in per_seed),
                    "mean_is_lower_bound": any(value is None for value in per_seed),
                },
                "cumulative_failures": [
                    round(float(value), 3) for value in np.mean(cumulative_runs, axis=0)
                ],
                "unique_failure_kinds": {
                    "kinds": kinds,
                    "mean": round(float(np.mean(kind_counts)), 3),
                    "std": round(float(np.std(kind_counts, ddof=0)), 3),
                },
                "best_risk": {
                    "mean": round(float(np.mean(risks)), 3),
                    "std": round(float(np.std(risks, ddof=0)), 3),
                },
            }
        )
    return BenchmarkResult.model_validate(
        {"budget": budget, "seed_count": seed_count, "strategies": strategies, "source": "mock"}
    )


def generate() -> None:
    rng = np.random.default_rng(17017)
    hardware = build_hardware_spec()
    envelope = build_envelope(hardware)
    replay = build_replay(rng)
    fail_params = ScenarioParams(
        payload_kg=1.1,
        link_mass_scale=1.12,
        joint_damping=0.45,
        latency_ticks=4,
        sensor_noise_rad=0.006,
        torque_limit_scale=0.72,
        speed_scale=0.82,
        kp_scale=0.9,
        push_force_n=8.0,
        push_time_s=2.0,
        seed=17017,
    )
    episodes = [
        EpisodeResult(
            episode_id="ep_016",
            params=ScenarioParams(seed=17016),
            failures=[],
            risk=0.18,
            max_tau_ratio=0.62,
            max_vel_ratio=0.57,
            final_ee_error_m=0.012,
            min_manipulability=0.08,
        ),
        EpisodeResult(
            episode_id="ep_017",
            params=fail_params,
            failures=replay.events,
            risk=1.24,
            max_tau_ratio=1.08,
            max_vel_ratio=0.88,
            final_ee_error_m=0.074,
            min_manipulability=0.018,
            trajectory_path="episode_ep_017.json",
        ),
    ]
    campaign = CampaignResult(
        robot=hardware.robot,
        budget=2,
        strategy="nemotron",
        seed=17017,
        episodes=episodes,
        events=[
            CampaignEvent(type="campaign_started", data={"campaign_id": "mock_campaign_01"}),
            CampaignEvent(type="hypothesis", data={"episode_id": "ep_017", "label": "[mock]"}),
            CampaignEvent(type="scenario_batch", data={"episode_ids": ["ep_016", "ep_017"]}),
            CampaignEvent(
                type="episode_done",
                data={"episode_id": "ep_016", "ticks": 360, "control_hz": 60, "duration_s": 6.0},
            ),
            CampaignEvent(
                type="episode_done",
                data={"episode_id": "ep_017", "ticks": 360, "control_hz": 60, "duration_s": 6.0},
            ),
            CampaignEvent(type="failure_found", data={"episode_id": "ep_017", "risk": 1.24}),
            CampaignEvent(type="cluster_updated", data={"failure_count": 2}),
            CampaignEvent(type="spec_ready", data={"robot": hardware.robot}),
            CampaignEvent(type="shrink_done", data={"episode_id": "ep_017", "label": "[mock]"}),
            CampaignEvent(type="diagnosis_ready", data={"label": "[mock]"}),
            CampaignEvent(type="campaign_done", data={"episode_count": 2}),
        ],
        source="mock",
    )
    benchmark = build_benchmark(rng)
    artifacts = {
        "campaign.json": campaign,
        "episode_ep_017.json": replay,
        "benchmark.json": benchmark,
        "envelope.json": envelope,
        "evidence.json": hardware,
    }
    MOCK_DIR.mkdir(parents=True, exist_ok=True)
    for filename, model in artifacts.items():
        MOCK_DIR.joinpath(filename).write_text(
            model.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    generate()
