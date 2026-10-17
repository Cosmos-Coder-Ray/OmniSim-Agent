from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from omnisim.agents.baselines import Trial
from omnisim.schemas import (
    BenchmarkResult,
    BenchmarkStrategy,
    FailureKind,
    FirstFailureSummary,
    MeanStd,
    StrategyName,
    UniqueFailureKindsSummary,
)


@dataclass(frozen=True, slots=True)
class BenchmarkPaths:
    output: Path
    raw: Path
    cache: Path
    is_dev: bool


def resolve_benchmark_paths(
    root: Path,
    *,
    seed_start: int,
    seeds: int,
    output: Path | None = None,
) -> BenchmarkPaths:
    if seeds <= 0:
        raise ValueError("seeds must be positive")
    root = root.resolve()
    data = (root / "data").resolve()
    dev_dir = data / "dev"
    is_dev = seed_start <= 99 and seed_start + seeds - 1 >= 0
    default_output = dev_dir / "benchmark.json" if is_dev else data / "benchmark.json"
    chosen = output if output is not None else default_output
    if not chosen.is_absolute():
        chosen = root / chosen
    chosen = chosen.resolve()
    if is_dev:
        if not chosen.is_relative_to(dev_dir):
            raise ValueError("dev output must stay under data/dev")
        raw = dev_dir / "benchmark_raw.jsonl"
        cache = dev_dir / "benchmark_llm_cache"
    else:
        if not chosen.is_relative_to(data) or chosen.is_relative_to(dev_dir):
            raise ValueError("reporting output must stay under data and outside data/dev")
        raw = data / "benchmark_raw.jsonl"
        cache = data / "benchmark_llm_cache"
    if chosen == raw:
        raise ValueError("result output must not overwrite the raw JSONL path")
    return BenchmarkPaths(chosen, raw, cache, is_dev)


def _failed(trial: Trial) -> bool:
    return trial.result is not None and (bool(trial.result.failures) or trial.result.risk >= 1.0)


def aggregate(runs: dict[StrategyName, list[list[Trial]]], *, budget: int) -> BenchmarkResult:
    strategies: list[BenchmarkStrategy] = []
    for name, seeds in runs.items():
        if not seeds:
            raise ValueError(f"strategy {name} has no seed runs")
        if any(len(trials) != budget for trials in seeds):
            raise ValueError("each seed run must contain exactly budget trials")

        first: list[int | None] = []
        per_seed_kinds: list[set[FailureKind]] = []
        best_risks: list[float] = []
        failure_curves: list[list[float]] = []
        for trials in seeds:
            first.append(
                next((index + 1 for index, trial in enumerate(trials) if _failed(trial)), None)
            )
            kinds: set[FailureKind] = {
                event.kind
                for trial in trials
                if trial.result is not None
                for event in trial.result.failures
            }
            per_seed_kinds.append(kinds)
            best_risks.append(max((trial.result.risk if trial.result else 0.0) for trial in trials))
            cumulative = 0
            curve: list[float] = []
            for trial in trials:
                cumulative += int(_failed(trial))
                curve.append(float(cumulative))
            failure_curves.append(curve)

        censored = [budget + 1 if value is None else value for value in first]
        all_kinds = sorted(set().union(*per_seed_kinds))
        strategies.append(
            BenchmarkStrategy(
                strategy=name,
                sims_to_first_failure=FirstFailureSummary(
                    mean=fmean(censored),
                    std=pstdev(censored),
                    per_seed=first,
                    seeds_with_failure=sum(value is not None for value in first),
                    mean_is_lower_bound=any(value is None for value in first),
                ),
                cumulative_failures=[
                    fmean(curve[index] for curve in failure_curves) for index in range(budget)
                ],
                unique_failure_kinds=UniqueFailureKindsSummary(
                    kinds=all_kinds,
                    mean=fmean(len(kinds) for kinds in per_seed_kinds),
                    std=pstdev(len(kinds) for kinds in per_seed_kinds),
                ),
                best_risk=MeanStd(mean=fmean(best_risks), std=pstdev(best_risks)),
            )
        )
    return BenchmarkResult(
        budget=budget,
        seed_count=len(next(iter(runs.values()))) if runs else 0,
        strategies=strategies,
        source="live",
    )


def _write_temp(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as stream:
            temp_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        return temp_path
    except BaseException:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def write_completed_outputs(
    paths: BenchmarkPaths,
    rows: list[dict[str, Any]],
    result: BenchmarkResult,
) -> None:
    if paths.output == paths.raw:
        raise ValueError("result output must not overwrite the raw JSONL path")
    raw_content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    )
    result_content = (
        json.dumps(result.model_dump(mode="json"), sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    temps: dict[Path, Path] = {}
    backups: dict[Path, Path | None] = {}
    replaced: list[Path] = []
    try:
        temps[paths.raw] = _write_temp(paths.raw, raw_content)
        temps[paths.output] = _write_temp(paths.output, result_content)
        for path in (paths.raw, paths.output):
            if path.exists():
                backup = path.with_name(f".{path.name}.backup")
                shutil.copy2(path, backup)
                backups[path] = backup
            else:
                backups[path] = None
        for path in (paths.raw, paths.output):
            os.replace(temps[path], path)
            replaced.append(path)
    except BaseException:
        for path in reversed(replaced):
            backup = backups.get(path)
            if backup is not None and backup.exists():
                os.replace(backup, path)
            elif path.exists():
                path.unlink()
        raise
    finally:
        for path in (*temps.values(), *(item for item in backups.values() if item is not None)):
            if path.exists():
                path.unlink()
