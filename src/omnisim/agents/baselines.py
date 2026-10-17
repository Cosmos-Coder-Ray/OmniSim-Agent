from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import cma
import numpy as np
from numpy.typing import NDArray

from omnisim.schemas import EpisodeResult, ScenarioParams

PARAM_NAMES = tuple(name for name in ScenarioParams.model_fields if name != "seed")
ProposalOrigin = Literal["llm", "exploit", "explore", "random", "cma"]
Bounds = dict[str, tuple[float, float]]


@dataclass(frozen=True, slots=True)
class Trial:
    params: ScenarioParams
    result: EpisodeResult | None
    error: str | None


class Strategy(Protocol):
    batch_size: int
    last_origins: list[ProposalOrigin]

    def propose(
        self, history: list[Trial], bounds: Mapping[str, tuple[float, float]], k: int
    ) -> tuple[str, list[ScenarioParams]]: ...


def scenario_bounds() -> Bounds:
    fields = ScenarioParams.model_json_schema()["properties"]
    return {
        name: (float(fields[name]["minimum"]), float(fields[name]["maximum"]))
        for name in PARAM_NAMES
    }


def normalize(
    params: ScenarioParams, bounds: Mapping[str, tuple[float, float]]
) -> NDArray[np.float64]:
    return np.asarray(
        [
            (float(getattr(params, name)) - bounds[name][0]) / (bounds[name][1] - bounds[name][0])
            for name in PARAM_NAMES
        ],
        dtype=np.float64,
    )


def denormalize(
    values: Sequence[float] | NDArray[np.float64],
    bounds: Mapping[str, tuple[float, float]],
) -> ScenarioParams:
    params: dict[str, float | int] = {}
    for name, value in zip(PARAM_NAMES, values, strict=True):
        low, high = bounds[name]
        raw = low + float(np.clip(value, 0.0, 1.0)) * (high - low)
        if name == "latency_ticks":
            params[name] = int(np.clip(round(raw), low, high))
        else:
            params[name] = float(np.clip(raw, low, high))
    return ScenarioParams.model_validate(params)


class RandomStrategy:
    batch_size = 4

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.last_origins: list[ProposalOrigin] = []

    def propose(
        self, history: list[Trial], bounds: Mapping[str, tuple[float, float]], k: int
    ) -> tuple[str, list[ScenarioParams]]:
        del history
        proposals = [denormalize(self.rng.random(len(PARAM_NAMES)), bounds) for _ in range(k)]
        self.last_origins = ["random"] * k
        return "uniform random search", proposals


class CMAESStrategy:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        cma_seed = int(self.rng.integers(0, 2**32 - 1, dtype=np.uint64)) + 1
        self._es = cma.CMAEvolutionStrategy(
            [0.5] * len(PARAM_NAMES),
            0.3,
            {
                "bounds": [0.0, 1.0],
                "randn": self._randn,
                "seed": cma_seed,
                "verbose": -9,
            },
        )
        self.batch_size = int(self._es.popsize)
        self._pending: list[list[float]] = []
        self.last_origins: list[ProposalOrigin] = []

    def _randn(self, *shape: int) -> NDArray[np.float64]:
        return self.rng.standard_normal(shape)

    def propose(
        self, history: list[Trial], bounds: Mapping[str, tuple[float, float]], k: int
    ) -> tuple[str, list[ScenarioParams]]:
        if self._pending:
            trials = history[-len(self._pending) :]
            if len(trials) != len(self._pending):
                raise ValueError("history does not contain the pending CMA-ES batch")
            objectives = [
                -trial.result.risk if trial.result is not None else 0.0 for trial in trials
            ]
            self._es.tell(self._pending, objectives)

        candidates = self._es.ask(number=k)
        self._pending = [[float(value) for value in candidate] for candidate in candidates]
        proposals = [denormalize(candidate, bounds) for candidate in self._pending]
        self.last_origins = ["cma"] * len(proposals)
        return f"CMA-ES generation {self._es.countiter + 1}, sigma {self._es.sigma:.2f}", proposals
