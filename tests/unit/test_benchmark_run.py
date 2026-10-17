from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pytest

from omnisim.core.benchmark import resolve_benchmark_paths
from omnisim.core.benchmark_run import BenchmarkOptions, _parser, run_benchmark
from omnisim.llm.budget import BudgetExceeded
from omnisim.sim.runner import EpisodeRunner


@pytest.mark.parametrize("seed_start", [1, 1000])
def test_abort_leaves_existing_benchmark_files_untouched(tmp_path: Path, seed_start: int) -> None:
    paths = resolve_benchmark_paths(tmp_path, seed_start=seed_start, seeds=1)
    paths.output.parent.mkdir(parents=True, exist_ok=True)
    paths.raw.write_text("keep raw\n", encoding="utf-8")
    paths.output.write_text("keep result\n", encoding="utf-8")

    class AbortedRunner:
        def __enter__(self) -> AbortedRunner:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def run_batch(self, *args: object, **kwargs: object) -> object:
            raise BudgetExceeded("test abort")

    options = BenchmarkOptions(
        budget=1,
        seeds=1,
        seed_start=seed_start,
        strategies=("random",),
        workers=1,
    )

    with pytest.raises(BudgetExceeded):
        run_benchmark(
            options,
            root=tmp_path,
            runner_factory=cast(type[EpisodeRunner], AbortedRunner),
        )

    assert paths.raw.read_text(encoding="utf-8") == "keep raw\n"
    assert paths.output.read_text(encoding="utf-8") == "keep result\n"


def test_cli_defaults_match_benchmark_policy() -> None:
    args = _parser().parse_args(["--budget", "1", "--seeds", "1"])

    assert args.seed_start == 1000
    assert args.strategies == ["nemotron", "random", "cmaes"]
    assert args.workers == min(4, os.cpu_count() or 1)
    assert not args.replay_only
    assert args.out is None
