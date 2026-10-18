from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np

from omnisim.agents.baselines import ProposalOrigin, Strategy, Trial, scenario_bounds
from omnisim.schemas import EpisodeResult, ScenarioParams
from omnisim.sim.detectors import DetectorConfig
from omnisim.sim.limits import UrdfLimits
from omnisim.sim.runner import EpisodeError

BatchCallback = Callable[[int, str, list[ProposalOrigin], list[Trial]], None]
StopPredicate = Callable[[list[Trial]], bool]


class BatchRunner(Protocol):
    def run_batch(
        self,
        params_list: list[ScenarioParams],
        limits: UrdfLimits,
        *,
        workers: int,
        run_dir: Path | None = None,
        cfg: DetectorConfig = DetectorConfig(),
    ) -> list[EpisodeResult | EpisodeError]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TrialBatch:
    index: int
    hypothesis: str
    origins: list[ProposalOrigin]
    trials: list[Trial]
    duration_s: float


def _sim_seed(benchmark_seed: int, sim_index: int) -> int:
    return int(
        np.random.SeedSequence([benchmark_seed % (2**32), sim_index]).generate_state(
            1, dtype=np.uint32
        )[0]
    )


def iter_trial_batches(
    strategy: Strategy,
    runner: BatchRunner,
    limits: UrdfLimits,
    *,
    benchmark_seed: int,
    budget: int,
    workers: int,
    cfg: DetectorConfig = DetectorConfig(),
    on_batch: BatchCallback | None = None,
    stop_predicate: StopPredicate | None = None,
    run_dir: Path | None = None,
) -> Iterator[TrialBatch]:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if strategy.batch_size <= 0:
        raise ValueError("strategy batch size must be positive")

    history: list[Trial] = []
    batch_index = 0
    bounds = scenario_bounds()
    while len(history) < budget:
        k = min(strategy.batch_size, budget - len(history))
        hypothesis, proposals = strategy.propose(history, bounds, k)
        if len(proposals) != k or len(strategy.last_origins) != k:
            raise RuntimeError("strategy must return k proposals and origins")
        sim_start = len(history)
        params_list = [
            params.model_copy(update={"seed": _sim_seed(benchmark_seed, sim_start + i)})
            for i, params in enumerate(proposals)
        ]
        batch_started = perf_counter()
        results = runner.run_batch(params_list, limits, workers=workers, run_dir=run_dir, cfg=cfg)
        duration_s = perf_counter() - batch_started
        trials = [
            Trial(
                params=params,
                result=result if isinstance(result, EpisodeResult) else None,
                error=(
                    None
                    if isinstance(result, EpisodeResult)
                    else f"{result.error_type}: {result.message}"
                ),
            )
            for params, result in zip(params_list, results, strict=True)
        ]
        if on_batch is not None:
            on_batch(batch_index, hypothesis, strategy.last_origins.copy(), trials)
        history.extend(trials)
        yield TrialBatch(batch_index, hypothesis, strategy.last_origins.copy(), trials, duration_s)
        batch_index += 1
        if stop_predicate is not None and stop_predicate(history):
            break


def run_trials(
    strategy: Strategy,
    runner: BatchRunner,
    limits: UrdfLimits,
    *,
    benchmark_seed: int,
    budget: int,
    workers: int,
    cfg: DetectorConfig = DetectorConfig(),
    on_batch: BatchCallback | None = None,
) -> list[Trial]:
    return [
        trial
        for batch in iter_trial_batches(
            strategy,
            runner,
            limits,
            benchmark_seed=benchmark_seed,
            budget=budget,
            workers=workers,
            cfg=cfg,
            on_batch=on_batch,
        )
        for trial in batch.trials
    ]
