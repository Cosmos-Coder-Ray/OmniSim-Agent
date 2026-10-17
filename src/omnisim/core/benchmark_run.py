from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

from rich.console import Console
from rich.table import Table

from omnisim.agents.adversary import PROMPT_PATH, NemotronAdversary
from omnisim.agents.baselines import ProposalOrigin, Trial
from omnisim.agents.factory import make_strategy
from omnisim.core.benchmark import (
    BenchmarkPaths,
    aggregate,
    resolve_benchmark_paths,
    write_completed_outputs,
)
from omnisim.core.campaign import run_trials
from omnisim.llm.benchmark_cache import (
    BenchmarkLLMCache,
    ReplayCacheError,
    cache_size,
    prompt_file_hash,
)
from omnisim.llm.budget import BudgetExceeded
from omnisim.schemas import BenchmarkResult, StrategyName
from omnisim.sim.detectors import DetectorConfig
from omnisim.sim.limits import load_urdf_limits
from omnisim.sim.runner import EpisodeRunner

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STRATEGIES: tuple[StrategyName, ...] = ("nemotron", "random", "cmaes")
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 4096
MAX_CACHE_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    budget: int
    seeds: int
    seed_start: int = 1000
    strategies: tuple[StrategyName, ...] = DEFAULT_STRATEGIES
    workers: int = min(4, os.cpu_count() or 1)
    output: Path | None = None
    replay_only: bool = False
    llm_fraction: float = 1.0
    salt: str = "benchmark-v1"


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    result: BenchmarkResult
    paths: BenchmarkPaths
    fallback_batches: int
    episode_errors: int
    llm_calls: int
    cache_hits: int
    llm_tokens: int
    cache_bytes: int
    wall_seconds: float
    resolved_model_id: str | None


def _strategy_settings(
    options: BenchmarkOptions, cmaes_population_size: int | None
) -> dict[str, object]:
    return {
        "strategies": list(options.strategies),
        "workers": options.workers,
        "llm_fraction": options.llm_fraction,
        "llm_temperature": LLM_TEMPERATURE,
        "llm_max_tokens": LLM_MAX_TOKENS,
        "cmaes": {
            "x0": 0.5,
            "sigma0": 0.3,
            "bounds": [0.0, 1.0],
            "restarts": 0,
            "population_size": cmaes_population_size,
            "library_seed": "deterministic nonzero seed derived from strategy seed",
        },
    }


def _validate_options(options: BenchmarkOptions) -> None:
    if options.budget <= 0:
        raise ValueError("budget must be positive")
    if options.seeds <= 0:
        raise ValueError("seeds must be positive")
    if options.workers <= 0:
        raise ValueError("workers must be positive")
    if not options.strategies or len(set(options.strategies)) != len(options.strategies):
        raise ValueError("strategies must be non-empty and unique")
    if options.replay_only and "nemotron" not in options.strategies:
        raise ValueError("replay-only requires the nemotron strategy")
    if not 0.0 <= options.llm_fraction <= 1.0:
        raise ValueError("llm-fraction must be between zero and one")


def run_benchmark(
    options: BenchmarkOptions,
    *,
    root: Path = ROOT,
    runner_factory: Callable[[], EpisodeRunner] = EpisodeRunner,
) -> BenchmarkReport:
    _validate_options(options)
    paths = resolve_benchmark_paths(
        root,
        seed_start=options.seed_start,
        seeds=options.seeds,
        output=options.output,
    )
    cfg = DetectorConfig()
    prompt_hash = prompt_file_hash(PROMPT_PATH)
    cache = BenchmarkLLMCache(
        paths.cache,
        salt=options.salt,
        prompt_hash=prompt_hash,
        replay_only=options.replay_only,
    )
    limits = load_urdf_limits()
    seeds = list(range(options.seed_start, options.seed_start + options.seeds))
    runs: dict[StrategyName, list[list[Trial]]] = {}
    rows: list[dict[str, object]] = []
    fallback_batches = 0
    episode_errors = 0
    llm_calls = 0
    cache_hits = 0
    llm_tokens = 0
    model_ids: set[str] = set()
    cmaes_population_size: int | None = None
    started = time.perf_counter()

    with runner_factory() as runner:
        for strategy_name in options.strategies:
            runs[strategy_name] = []
            for benchmark_seed in seeds:
                strategy = make_strategy(
                    strategy_name,
                    seed=benchmark_seed,
                    llm_fraction=options.llm_fraction,
                    budget=options.budget,
                    chat_json=cache if strategy_name == "nemotron" else None,
                )
                if strategy_name == "cmaes":
                    cmaes_population_size = strategy.batch_size
                simulation_index = 0

                def record_batch(
                    batch_index: int,
                    hypothesis: str,
                    origins: list[ProposalOrigin],
                    trials: list[Trial],
                ) -> None:
                    nonlocal cache_hits, episode_errors, fallback_batches, llm_calls, llm_tokens
                    fallback = isinstance(strategy, NemotronAdversary) and strategy.last_fallback
                    if fallback:
                        fallback_batches += 1
                    if isinstance(strategy, NemotronAdversary):
                        if strategy.last_cache_hit:
                            cache_hits += 1
                        elif strategy.last_call_made:
                            llm_calls += 1
                            llm_tokens += strategy.last_usage.total
                        if strategy.resolved_model_id:
                            model_ids.add(strategy.resolved_model_id)
                    for offset, (origin, trial) in enumerate(zip(origins, trials, strict=True)):
                        nonlocal simulation_index
                        episode_errors += int(trial.error is not None)
                        rows.append(
                            {
                                "type": "simulation",
                                "strategy": strategy_name,
                                "benchmark_seed": benchmark_seed,
                                "simulation_index": simulation_index + offset,
                                "sim_seed": trial.params.seed,
                                "batch_index": batch_index,
                                "hypothesis": hypothesis,
                                "origin": origin,
                                "fallback_batch": fallback,
                                "params": trial.params.model_dump(mode="json"),
                                "result": (
                                    trial.result.model_dump(mode="json")
                                    if trial.result is not None
                                    else None
                                ),
                                "error": trial.error,
                            }
                        )
                    simulation_index += len(trials)

                history = run_trials(
                    strategy,
                    runner,
                    limits,
                    benchmark_seed=benchmark_seed,
                    budget=options.budget,
                    workers=options.workers,
                    cfg=cfg,
                    on_batch=record_batch,
                )
                runs[strategy_name].append(history)

    if len(model_ids) > 1:
        raise RuntimeError(f"Nemotron model changed during benchmark: {sorted(model_ids)}")
    result = aggregate(runs, budget=options.budget)
    wall_seconds = time.perf_counter() - started
    cache_bytes = cache_size(paths.cache)
    resolved_model_id = next(iter(model_ids), None)
    raw_header: dict[str, object] = {
        "type": "header",
        "format_version": 1,
        "resolved_model_id": resolved_model_id,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "prompt_file": "src/omnisim/llm/prompts/adversary_system.md",
        "prompt_file_hash": prompt_hash,
        "detector_config": asdict(cfg),
        "pybullet_version": version("pybullet"),
        "budget": options.budget,
        "seeds": seeds,
        "strategy_settings": _strategy_settings(options, cmaes_population_size),
        "salt": options.salt,
        "replay_only": options.replay_only,
        "dev_run": paths.is_dev,
    }
    counts = {
        "simulations": len(rows),
        "episode_errors": episode_errors,
        "fallback_batches": fallback_batches,
        "llm_calls": llm_calls,
        "cache_hits": cache_hits,
        "llm_tokens": llm_tokens,
    }
    final_rows = [raw_header, *rows, {"type": "summary", "counts": counts}]
    write_completed_outputs(paths, final_rows, result)
    return BenchmarkReport(
        result=result,
        paths=paths,
        fallback_batches=fallback_batches,
        episode_errors=episode_errors,
        llm_calls=llm_calls,
        cache_hits=cache_hits,
        llm_tokens=llm_tokens,
        cache_bytes=cache_bytes,
        wall_seconds=wall_seconds,
        resolved_model_id=resolved_model_id,
    )


def _token_estimate(options: BenchmarkOptions) -> int:
    prompt_tokens = (PROMPT_PATH.stat().st_size + 3) // 4
    calls = options.seeds * ((options.budget + 3) // 4)
    return calls * 2 * (prompt_tokens + LLM_MAX_TOKENS)


def print_summary(report: BenchmarkReport, console: Console | None = None) -> None:
    console = console or Console()
    table = Table(title="Benchmark results")
    table.add_column("strategy")
    table.add_column("first failure (mean ± std)")
    table.add_column("seeds with failure")
    table.add_column("failure kinds (mean ± std)")
    table.add_column("best risk (mean ± std)")
    for item in report.result.strategies:
        first = item.sims_to_first_failure
        table.add_row(
            item.strategy,
            f"{first.mean:.2f} ± {first.std:.2f}"
            + (" lower bound" if first.mean_is_lower_bound else ""),
            f"{first.seeds_with_failure}/{report.result.seed_count}",
            f"{item.unique_failure_kinds.mean:.2f} ± {item.unique_failure_kinds.std:.2f}",
            f"{item.best_risk.mean:.3f} ± {item.best_risk.std:.3f}",
        )
    console.print(table)
    for item in report.result.strategies:
        curve = ", ".join(f"{value:.2f}" for value in item.cumulative_failures)
        console.print(f"{item.strategy} mean cumulative failures: {curve}")
    console.print(
        f"wall time {report.wall_seconds:.1f}s; live LLM calls {report.llm_calls}; "
        f"cache hits {report.cache_hits}; live tokens {report.llm_tokens}"
    )
    if report.fallback_batches or report.episode_errors:
        console.print(
            "[bold red]WARNING: Nemotron fallback proposals: "
            f"{report.fallback_batches}; EpisodeError results: {report.episode_errors}[/bold red]"
        )
    console.print(f"benchmark cache size: {report.cache_bytes:,} bytes")
    if report.cache_bytes > MAX_CACHE_BYTES:
        console.print("[bold yellow]WARNING: benchmark cache exceeds 5 MiB[/bold yellow]")
    console.print(f"raw: {report.paths.raw}")
    console.print(f"result: {report.paths.output}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic adversary benchmarks")
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seeds", type=int, required=True)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=DEFAULT_STRATEGIES,
        default=list(DEFAULT_STRATEGIES),
    )
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--replay-only", action="store_true")
    parser.add_argument("--llm-fraction", type=float, default=1.0)
    parser.add_argument("--salt", default="benchmark-v1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    options = BenchmarkOptions(
        budget=args.budget,
        seeds=args.seeds,
        seed_start=args.seed_start,
        strategies=tuple(args.strategies),
        workers=args.workers,
        output=args.out,
        replay_only=args.replay_only,
        llm_fraction=args.llm_fraction,
        salt=args.salt,
    )
    try:
        _validate_options(options)
        paths = resolve_benchmark_paths(
            ROOT,
            seed_start=options.seed_start,
            seeds=options.seeds,
            output=options.output,
        )
        console = Console()
        run_type = "dev" if paths.is_dev else "reporting"
        console.print(
            f"{run_type} benchmark: seeds {options.seed_start}–"
            f"{options.seed_start + options.seeds - 1}; output {paths.output}"
        )
        if "nemotron" in options.strategies and options.llm_fraction:
            console.print(f"approximate maximum Nemotron tokens: {_token_estimate(options):,}")
        report = run_benchmark(options)
    except BudgetExceeded as exc:
        Console(stderr=True).print(
            f"benchmark stopped by token budget: {exc}. Check OMNISIM_DAILY_TOKEN_CAP."
        )
        return 2
    except ReplayCacheError as exc:
        Console(stderr=True).print(f"benchmark replay failed: {exc}")
        return 2
    except KeyboardInterrupt:
        Console(stderr=True).print("benchmark aborted; existing output files were left untouched")
        return 130
    except ValueError as exc:
        Console(stderr=True).print(f"benchmark configuration error: {exc}")
        return 2
    print_summary(report, console)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
