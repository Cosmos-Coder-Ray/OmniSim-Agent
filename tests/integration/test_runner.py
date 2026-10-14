from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from omnisim.schemas import EpisodeResult, Replay, ScenarioParams
from omnisim.sim import runner as runner_module
from omnisim.sim.limits import UrdfLimits
from omnisim.sim.runner import EpisodeError, EpisodeRunner, run_episode


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


def test_run_batch_serial_and_spawn_results_match(
    tmp_path: Path,
    urdf_limits: UrdfLimits,
) -> None:
    params = [ScenarioParams(seed=seed) for seed in range(20)]
    run_dir = tmp_path / "runs"

    with EpisodeRunner() as serial_runner:
        serial = serial_runner.run_batch(params, urdf_limits, workers=1, run_dir=run_dir)
    serial_files = {path.name: path.read_bytes() for path in run_dir.glob("*.json")}

    with EpisodeRunner() as parallel_runner:
        parallel = parallel_runner.run_batch(params, urdf_limits, workers=4, run_dir=run_dir)

    assert parallel == serial
    assert [result.episode_id for result in parallel] == [f"ep_{i:03d}" for i in range(20)]
    assert {path.name: path.read_bytes() for path in run_dir.glob("*.json")} == serial_files


def test_run_batch_cooperative_timeout_returns_ordered_errors(
    urdf_limits: UrdfLimits,
) -> None:
    params = [ScenarioParams(seed=seed) for seed in range(4)]

    with EpisodeRunner() as runner:
        results = runner.run_batch(params, urdf_limits, workers=2, timeout_s=1e-9)

    assert [result.episode_id for result in results] == [f"ep_{i:03d}" for i in range(4)]
    errors = [result for result in results if isinstance(result, EpisodeError)]
    assert len(errors) == 4
    assert all(error.error_type in {"TimeoutError", "WorkerTimeout"} for error in errors)


def test_run_batch_returns_serial_infrastructure_errors(
    monkeypatch: pytest.MonkeyPatch,
    urdf_limits: UrdfLimits,
) -> None:
    def fail_episode(*args: object, **kwargs: object) -> EpisodeResult:
        raise OSError("storage unavailable")

    monkeypatch.setattr(runner_module, "run_episode", fail_episode)
    params = ScenarioParams(seed=17)

    with EpisodeRunner() as runner:
        results = runner.run_batch([params], urdf_limits, workers=1)

    assert len(results) == 1
    error = results[0]
    assert isinstance(error, EpisodeError)
    assert error.params == params
    assert error.error_type == "OSError"
    assert error.message == "storage unavailable"


def test_run_batch_keeps_successes_and_worker_errors_in_order(
    monkeypatch: pytest.MonkeyPatch,
    urdf_limits: UrdfLimits,
) -> None:
    class FakePool:
        def __init__(self, processes: int, initializer: object) -> None:
            assert processes == 2
            assert callable(initializer)

        def apply_async(
            self,
            _function: object,
            args: tuple[object, ...],
            *,
            callback: Callable[[EpisodeResult | EpisodeError], None],
            error_callback: Callable[[BaseException], None],
        ) -> None:
            params = cast(ScenarioParams, args[0])
            episode_id = cast(str, args[3])
            if episode_id == "ep_001":
                error_callback(RuntimeError("worker failed"))
                return
            callback(
                EpisodeResult(
                    episode_id=episode_id,
                    params=params,
                    failures=[],
                    risk=0.0,
                    max_tau_ratio=0.0,
                    max_vel_ratio=0.0,
                    final_ee_error_m=0.0,
                    min_manipulability=0.0,
                )
            )

        def close(self) -> None:
            pass

        def terminate(self) -> None:
            pass

        def join(self) -> None:
            pass

    class FakeContext:
        def Pool(self, *, processes: int, initializer: object) -> FakePool:
            return FakePool(processes, initializer)

    monkeypatch.setattr(runner_module.mp, "get_context", lambda method: FakeContext())
    params = [ScenarioParams(seed=seed) for seed in range(3)]

    with EpisodeRunner() as runner:
        results = runner.run_batch(params, urdf_limits, workers=2)

    assert [result.episode_id for result in results] == ["ep_000", "ep_001", "ep_002"]
    assert isinstance(results[0], EpisodeResult)
    assert isinstance(results[1], EpisodeError)
    assert results[1].error_type == "RuntimeError"
    assert results[2].params == params[2]
