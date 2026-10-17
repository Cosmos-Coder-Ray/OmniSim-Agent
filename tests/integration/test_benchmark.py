import json
from pathlib import Path

import pytest

from omnisim.core.benchmark_run import BenchmarkOptions, run_benchmark


@pytest.mark.slow
def test_offline_random_and_cmaes_benchmark(tmp_path: Path) -> None:
    report = run_benchmark(
        BenchmarkOptions(
            budget=8,
            seeds=1,
            seed_start=1000,
            strategies=("random", "cmaes"),
            workers=2,
        ),
        root=tmp_path,
    )

    assert [item.strategy for item in report.result.strategies] == ["random", "cmaes"]
    assert all(len(item.cumulative_failures) == 8 for item in report.result.strategies)
    assert report.paths.raw.is_file()
    assert report.paths.output.is_file()
    assert report.episode_errors == 0
    rows = [json.loads(line) for line in report.paths.raw.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["type"] == "header"
    assert rows[-1]["type"] == "summary"
    assert rows[0]["resolved_model_id"] is None
    assert rows[0]["temperature"] == 0.7
    assert rows[0]["max_tokens"] == 4096
    assert rows[0]["prompt_file_hash"]
    assert rows[0]["detector_config"]["risk_cap"] == 10.0
    assert rows[0]["pybullet_version"]
    assert len(rows) == 18
