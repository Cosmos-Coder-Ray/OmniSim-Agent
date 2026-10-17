from __future__ import annotations

from pathlib import Path

import pytest

from omnisim.agents.baselines import Trial
from omnisim.core.benchmark import (
    aggregate,
    resolve_benchmark_paths,
    write_completed_outputs,
)
from omnisim.schemas import EpisodeResult, FailureEvent, FailureKind, ScenarioParams


def _trial(
    *,
    risk: float = 0.2,
    failure: bool = False,
    error: bool = False,
    kind: FailureKind = "collision",
) -> Trial:
    params = ScenarioParams()
    result = None
    message = None
    if not error:
        result = EpisodeResult(
            episode_id="ep_0",
            params=params,
            failures=([FailureEvent(kind=kind, t=0.1, severity=1.0)] if failure else []),
            risk=1.0 if failure else risk,
            max_tau_ratio=0.0,
            max_vel_ratio=0.0,
            final_ee_error_m=0.0,
            min_manipulability=0.0,
        )
    else:
        message = "WorkerTimeout: timed out"
    return Trial(params, result, message)


def test_aggregate_censors_missing_failures_and_averages_seed_curves() -> None:
    result = aggregate(
        {
            "random": [
                [_trial(), _trial(failure=True), _trial(error=True)],
                [_trial(risk=0.2), _trial(risk=0.4), _trial(risk=0.6)],
            ]
        },
        budget=3,
    )
    metrics = result.strategies[0]

    assert metrics.sims_to_first_failure.per_seed == [2, None]
    assert metrics.sims_to_first_failure.mean == 3.0
    assert metrics.sims_to_first_failure.std == 1.0
    assert metrics.sims_to_first_failure.seeds_with_failure == 1
    assert metrics.sims_to_first_failure.mean_is_lower_bound
    assert metrics.cumulative_failures == [0.0, 0.5, 0.5]
    assert metrics.unique_failure_kinds.kinds == ["collision"]
    assert metrics.unique_failure_kinds.mean == 0.5
    assert metrics.best_risk.mean == 0.8
    assert len(metrics.cumulative_failures) == 3


def test_seed_ranges_select_dev_or_reporting_paths(tmp_path: Path) -> None:
    dev = resolve_benchmark_paths(tmp_path, seed_start=99, seeds=2)
    report = resolve_benchmark_paths(tmp_path, seed_start=100, seeds=1)
    negative_report = resolve_benchmark_paths(tmp_path, seed_start=-5, seeds=5)
    negative_dev = resolve_benchmark_paths(tmp_path, seed_start=-2, seeds=5)

    assert dev.is_dev
    assert dev.output == tmp_path / "data" / "dev" / "benchmark.json"
    assert dev.raw == tmp_path / "data" / "dev" / "benchmark_raw.jsonl"
    assert dev.cache == tmp_path / "data" / "dev" / "benchmark_llm_cache"
    assert not report.is_dev
    assert report.output == tmp_path / "data" / "benchmark.json"
    assert report.raw == tmp_path / "data" / "benchmark_raw.jsonl"
    assert report.cache == tmp_path / "data" / "benchmark_llm_cache"
    assert not negative_report.is_dev
    assert negative_dev.is_dev


@pytest.mark.parametrize(
    ("seed_start", "output", "message"),
    [
        (10, Path("data/benchmark.json"), "dev output"),
        (1000, Path("data/dev/custom.json"), "reporting output"),
        (1000, Path("data/benchmark_raw.jsonl"), "must not overwrite"),
    ],
)
def test_custom_output_must_stay_in_run_class(
    tmp_path: Path, seed_start: int, output: Path, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_benchmark_paths(
            tmp_path,
            seed_start=seed_start,
            seeds=1,
            output=output,
        )


def test_completed_outputs_replace_prior_content_from_scratch(tmp_path: Path) -> None:
    paths = resolve_benchmark_paths(tmp_path, seed_start=1000, seeds=1)
    result = aggregate({"random": [[_trial()]]}, budget=1)
    paths.output.parent.mkdir(parents=True, exist_ok=True)
    paths.raw.write_text("old raw\n", encoding="utf-8")
    paths.output.write_text("old result\n", encoding="utf-8")

    write_completed_outputs(paths, [{"type": "header"}, {"type": "summary"}], result)

    assert paths.raw.read_text(encoding="utf-8").splitlines() == [
        '{"type":"header"}',
        '{"type":"summary"}',
    ]
    assert '"budget": 1' in paths.output.read_text(encoding="utf-8")
    assert not list(paths.raw.parent.glob("*.tmp"))
    assert not list(paths.raw.parent.glob(".*.backup"))
