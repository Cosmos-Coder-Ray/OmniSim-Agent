from pathlib import Path

from omnisim.schemas import Replay, ScenarioParams
from omnisim.sim.limits import UrdfLimits
from omnisim.sim.runner import run_episode


class SlowShield:
    def apply(self, params: ScenarioParams) -> ScenarioParams:
        return params.model_copy(update={"speed_scale": 0.5})


def test_run_episode_without_run_dir_does_not_write_a_replay(
    urdf_limits: UrdfLimits,
) -> None:
    result = run_episode(
        ScenarioParams(seed=17),
        urdf_limits,
        episode_id="ep_000",
    )

    assert result.episode_id == "ep_000"
    assert result.trajectory_path is None
    assert result.params == ScenarioParams(seed=17)
    assert result.risk < 1.0


def test_run_episode_writes_live_replay_and_keeps_original_params(
    tmp_path: Path,
    urdf_limits: UrdfLimits,
) -> None:
    params = ScenarioParams(seed=17)
    baseline = run_episode(params, urdf_limits, episode_id="ep_000")
    result = run_episode(
        params,
        urdf_limits,
        SlowShield(),
        episode_id="ep_001",
        run_dir=tmp_path,
    )

    assert result.params == params
    assert result.max_tau_ratio != baseline.max_tau_ratio
    assert result.trajectory_path == (tmp_path / "ep_001.json").as_posix()
    replay_path = tmp_path / "ep_001.json"
    replay = Replay.model_validate_json(replay_path.read_text(encoding="utf-8"))
    assert replay.episode_id == "ep_001"
    assert replay.source == "live"
    assert replay.fps == 30
    assert len(replay.frames) == 180
    assert ": " not in replay_path.read_text(encoding="utf-8")
