from __future__ import annotations

from pathlib import Path

import pytest

from omnisim.core.demo_run import save_demo_run
from omnisim.schemas import CampaignResult, Replay

MOCK_DIR = Path(__file__).resolve().parents[2] / "web" / "public" / "mock"


def make_run(root: Path) -> None:
    run_dir = root / "source-run"
    trajectories = run_dir / "trajectories"
    trajectories.mkdir(parents=True)
    campaign = CampaignResult.model_validate_json(
        (MOCK_DIR / "campaign.json").read_text(encoding="utf-8")
    )
    episode = campaign.episodes[-1].model_copy(
        update={"trajectory_path": "trajectories/episode_ep_017.json"}
    )
    campaign = CampaignResult.model_validate(
        campaign.model_copy(update={"source": "live", "episodes": [episode]}).model_dump()
    )
    (run_dir / "campaign.json").write_text(campaign.model_dump_json(), encoding="utf-8")
    replay = Replay.model_validate_json(
        (MOCK_DIR / "episode_ep_017.json").read_text(encoding="utf-8")
    ).model_copy(update={"source": "live"})
    (trajectories / "episode_ep_017.json").write_text(replay.model_dump_json(), encoding="utf-8")
    (run_dir / "events.jsonl").write_text("{}\n", encoding="utf-8")


def test_demo_run_rewrites_and_validates_campaign_and_replays(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    make_run(runs_root)

    saved = save_demo_run("source-run", runs_root=runs_root, demo_root=tmp_path / "demo_runs")

    campaign = CampaignResult.model_validate_json(
        (saved.path / "campaign.json").read_text(encoding="utf-8")
    )
    replay = Replay.model_validate_json(
        (saved.path / "trajectories" / "episode_ep_017.json").read_text(encoding="utf-8")
    )
    assert campaign.source == replay.source == "recorded"
    assert campaign.episodes[0].trajectory_path == "trajectories/episode_ep_017.json"
    assert replay.episode_id == campaign.episodes[0].episode_id
    assert {name for name, _ in saved.files} == {
        "campaign.json",
        "events.jsonl",
        "trajectories/episode_ep_017.json",
    }


def test_demo_run_refuses_overwrite_without_force_and_replaces_with_force(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    demo_root = tmp_path / "demo_runs"
    make_run(runs_root)
    destination = demo_root / "hero"
    destination.mkdir(parents=True)
    marker = destination / "old.txt"
    marker.write_text("keep until force", encoding="utf-8")

    with pytest.raises(FileExistsError):
        save_demo_run("source-run", runs_root=runs_root, demo_root=demo_root)
    assert marker.read_text(encoding="utf-8") == "keep until force"

    saved = save_demo_run("source-run", force=True, runs_root=runs_root, demo_root=demo_root)

    assert not marker.exists()
    assert (
        CampaignResult.model_validate_json(
            (saved.path / "campaign.json").read_text(encoding="utf-8")
        ).source
        == "recorded"
    )


@pytest.mark.parametrize("value", ["..", "../outside", "nested/name"])
def test_demo_run_rejects_path_components(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError):
        save_demo_run(value, runs_root=tmp_path / "runs", demo_root=tmp_path / "demo_runs")
