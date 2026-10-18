from __future__ import annotations

from pathlib import Path

import pytest

from omnisim.core.orchestrator import run_campaign
from omnisim.schemas import HardwareSpec, Replay
from omnisim.sim.limits import load_urdf_limits
from omnisim.sim.runner import EpisodeRunner


def _spec() -> HardwareSpec:
    return HardwareSpec(
        robot="Franka Emika Panda",
        joint_torque_limits_nm=[],
        joint_velocity_limits=[],
        joint_position_limits=[],
        error_taxonomy=[],
    )


def _no_live_calls(*args, **kwargs):
    raise AssertionError("the random integration campaign must not make an LLM or spec call")


@pytest.mark.slow
def test_random_campaign_budget_8_runs_with_real_panda_sim(tmp_path: Path) -> None:
    outcome = run_campaign(
        "franka_panda",
        "random",
        seed=1000,
        budget=8,
        workers=2,
        run_id="campaign-integration",
        runs_root=tmp_path / "runs",
        runner=EpisodeRunner(),
        chat_json=_no_live_calls,
        spec=_spec(),
        build_spec_fn=_no_live_calls,
        compare_fn=lambda spec, limits: ([], []),
        mapping_fn=lambda kinds, spec, *, run_id=None: {},
        known_issues_fn=lambda robot, kind, *, run_id=None: [],
        max_diagnosed=0,
        limits=load_urdf_limits(),
    )

    assert outcome.result.budget == 8
    assert len(outcome.result.events) > 2
    assert outcome.result.events[-1].data["sims_used"] <= 8
    assert outcome.run_dir.joinpath("nominal.json").is_file()
    assert outcome.run_dir.joinpath("episodes.jsonl").is_file()
    for episode in outcome.result.episodes:
        assert episode.trajectory_path is not None
        assert not Path(episode.trajectory_path).is_absolute()
        replay_path = outcome.run_dir / episode.trajectory_path
        replay = Replay.model_validate_json(replay_path.read_text(encoding="utf-8"))
        assert replay.episode_id == episode.episode_id
        assert replay.source == "live"
