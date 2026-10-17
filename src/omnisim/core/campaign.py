from __future__ import annotations

from collections.abc import Callable

import numpy as np

from omnisim.agents.baselines import ProposalOrigin, Strategy, Trial, scenario_bounds
from omnisim.schemas import EpisodeResult
from omnisim.sim.detectors import DetectorConfig
from omnisim.sim.limits import UrdfLimits
from omnisim.sim.runner import EpisodeRunner

BatchCallback = Callable[[int, str, list[ProposalOrigin], list[Trial]], None]


def _sim_seed(benchmark_seed: int, sim_index: int) -> int:
    return int(
        np.random.SeedSequence([benchmark_seed % (2**32), sim_index]).generate_state(
            1, dtype=np.uint32
        )[0]
    )


def run_trials(
    strategy: Strategy,
    runner: EpisodeRunner,
    limits: UrdfLimits,
    *,
    benchmark_seed: int,
    budget: int,
    workers: int,
    cfg: DetectorConfig = DetectorConfig(),
    on_batch: BatchCallback | None = None,
) -> list[Trial]:
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
        results = runner.run_batch(params_list, limits, workers=workers, run_dir=None, cfg=cfg)
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
        batch_index += 1
    return history
