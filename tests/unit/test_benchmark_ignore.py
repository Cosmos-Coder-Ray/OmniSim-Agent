from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _is_ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", path],
        cwd=ROOT,
        check=False,
    )
    return result.returncode == 0


def test_dev_paths_and_non_benchmark_llm_caches_are_ignored() -> None:
    for path in (
        "data/dev/benchmark.json",
        "data/dev/benchmark_raw.jsonl",
        "data/dev/benchmark_llm_cache/entry.json",
        "data/campaign_llm_cache/entry.json",
        "data/llm_cache/entry.json",
    ):
        assert _is_ignored(path), path


def test_reporting_benchmark_paths_remain_visible_to_git() -> None:
    for path in (
        "data/benchmark.json",
        "data/benchmark_raw.jsonl",
        "data/benchmark_llm_cache/entry.json",
    ):
        assert not _is_ignored(path), path
