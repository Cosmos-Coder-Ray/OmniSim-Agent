from pathlib import Path

import pytest
from pydantic import BaseModel

from omnisim.schemas import (
    BenchmarkResult,
    CampaignResult,
    HardwareSpec,
    Replay,
    SafeEnvelope,
    ScenarioParams,
)

MOCK_DIR = Path(__file__).resolve().parents[2] / "web" / "public" / "mock"
PARAM_BOUNDS: tuple[tuple[str, float, float], ...] = (
    ("payload_kg", 0.0, 3.0),
    ("link_mass_scale", 0.7, 1.5),
    ("joint_damping", 0.0, 5.0),
    ("latency_ticks", 0.0, 12.0),
    ("sensor_noise_rad", 0.0, 0.02),
    ("torque_limit_scale", 0.5, 1.0),
    ("speed_scale", 0.5, 2.0),
    ("kp_scale", 0.5, 2.0),
    ("push_force_n", 0.0, 40.0),
    ("push_time_s", 0.0, 5.0),
)


@pytest.mark.parametrize(
    ("filename", "model"),
    [
        ("campaign.json", CampaignResult),
        ("episode_ep_017.json", Replay),
        ("benchmark.json", BenchmarkResult),
        ("envelope.json", SafeEnvelope),
        ("evidence.json", HardwareSpec),
    ],
)
def test_mock_file_validates(filename: str, model: type[BaseModel]) -> None:
    model.model_validate_json((MOCK_DIR / filename).read_text(encoding="utf-8"))


def test_mock_files_are_consistent() -> None:
    campaign = CampaignResult.model_validate_json(
        (MOCK_DIR / "campaign.json").read_text(encoding="utf-8")
    )
    replay = Replay.model_validate_json(
        (MOCK_DIR / "episode_ep_017.json").read_text(encoding="utf-8")
    )
    benchmark = BenchmarkResult.model_validate_json(
        (MOCK_DIR / "benchmark.json").read_text(encoding="utf-8")
    )
    envelope = SafeEnvelope.model_validate_json(
        (MOCK_DIR / "envelope.json").read_text(encoding="utf-8")
    )
    hardware = HardwareSpec.model_validate_json(
        (MOCK_DIR / "evidence.json").read_text(encoding="utf-8")
    )

    failed = next(episode for episode in campaign.episodes if episode.episode_id == "ep_017")
    passing = [episode for episode in campaign.episodes if episode.episode_id != "ep_017"]
    assert passing
    assert failed.trajectory_path == "episode_ep_017.json"
    assert failed.risk >= 1.0
    assert failed.failures == replay.events
    assert replay.episode_id == "ep_017"
    assert replay.fps == 30
    assert len(replay.frames) == 180
    assert len(replay.frames) / replay.fps == 6
    assert all(episode.risk < 1.0 and not episode.failures for episode in passing)

    run_event = next(
        event
        for event in campaign.events
        if event.type == "episode_done" and event.data.get("episode_id") == "ep_017"
    )
    assert run_event.data["ticks"] == 360
    assert run_event.data["control_hz"] == 60
    for episode in campaign.episodes:
        ScenarioParams.model_validate(episode.params.model_dump())
        for name, minimum, maximum in PARAM_BOUNDS:
            value = getattr(episode.params, name)
            assert minimum <= value <= maximum

    assert campaign.source == replay.source == benchmark.source == "mock"
    assert envelope.version == "mock-0"
    assert envelope.robot == hardware.robot
    assert all(item.verified for item in envelope.evidence)
    assert any(not item.verified for item in hardware.error_taxonomy)
    assert hardware.error_taxonomy
    for item in [
        *hardware.joint_torque_limits_nm,
        *hardware.joint_velocity_limits,
        *hardware.joint_position_limits,
        *hardware.error_taxonomy,
    ]:
        assert item.source_url.startswith("https://mock.invalid/")
        assert item.quote.startswith("[mock]")
