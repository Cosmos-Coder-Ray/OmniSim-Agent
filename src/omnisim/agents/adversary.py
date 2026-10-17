from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from omnisim.agents.baselines import (
    ProposalOrigin,
    Strategy,
    Trial,
    denormalize,
    normalize,
)
from omnisim.llm import nebius
from omnisim.llm.budget import BudgetExceeded, LLMError, Usage
from omnisim.schemas import ScenarioParams

PROMPT_PATH = Path(__file__).parents[1] / "llm" / "prompts" / "adversary_system.md"


class AdversaryResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hypothesis: str = ""
    scenarios: list[dict[str, Any]] = Field(default_factory=list)


ChatJSON = Callable[..., tuple[AdversaryResponse, Usage]]


class NemotronAdversary(Strategy):
    batch_size = 4

    def __init__(
        self,
        *,
        seed: int,
        llm_fraction: float = 1.0,
        epsilon: float = 0.001,
        total_budget: int | None = None,
        chat_json: ChatJSON | None = None,
        prompt_path: Path | None = None,
    ) -> None:
        if not 0.0 <= llm_fraction <= 1.0:
            raise ValueError("llm_fraction must be between zero and one")
        if not 0.0 < epsilon < 1.0:
            raise ValueError("epsilon must be between zero and one")
        self.rng = np.random.default_rng(seed)
        self.llm_fraction = llm_fraction
        self.epsilon = epsilon
        self.total_budget = total_budget
        self._chat_json = chat_json or nebius.chat_json
        self.prompt_path = prompt_path or PROMPT_PATH
        self.last_origins: list[ProposalOrigin] = []
        self.last_fallback = False
        self.last_usage = Usage()
        self.resolved_model_id: str | None = None

    def _request(
        self, history: list[Trial], bounds: Mapping[str, tuple[float, float]], k: int
    ) -> tuple[str, list[dict[str, Any]]]:
        prompt = self.prompt_path.read_text(encoding="utf-8")
        recent = [
            {
                "params": trial.params.model_dump(exclude={"seed"}),
                "risk": trial.result.risk,
                "failures": [
                    {"kind": event.kind, "joint": event.joint} for event in trial.result.failures
                ],
            }
            for trial in history[-12:]
            if trial.result is not None
        ]
        counts: Counter[tuple[str, int | None]] = Counter()
        best_risk: dict[tuple[str, int | None], float] = {}
        for trial in history:
            if trial.result is None:
                continue
            for event in trial.result.failures:
                key = (event.kind, event.joint)
                counts[key] += 1
                best_risk[key] = max(best_risk.get(key, 0.0), trial.result.risk)
        user = json.dumps(
            {
                "bounds": bounds,
                "requested": k,
                "remaining_budget": (
                    max(0, self.total_budget - len(history))
                    if self.total_budget is not None
                    else None
                ),
                "recent_successful_trials": recent,
                "failure_tally": [
                    {
                        "kind": kind,
                        "joint": joint,
                        "count": count,
                        "best_risk": best_risk[(kind, joint)],
                    }
                    for (kind, joint), count in sorted(counts.items())
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        response, usage = self._chat_json(
            "fast",
            prompt,
            user,
            AdversaryResponse,
            temperature=0.7,
            max_tokens=4096,
        )
        self.last_usage = usage
        return response.hypothesis[:400], response.scenarios

    @staticmethod
    def _sanitize(raw: object, bounds: Mapping[str, tuple[float, float]]) -> ScenarioParams | None:
        if not isinstance(raw, dict):
            return None
        params = ScenarioParams().model_dump(exclude={"seed"})
        for name in params:
            if name not in raw:
                continue
            value = raw[name]
            if isinstance(value, bool):
                return None
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(numeric):
                return None
            low, high = bounds[name]
            numeric = float(np.clip(numeric, low, high))
            params[name] = int(round(numeric)) if name == "latency_ticks" else numeric
        try:
            return ScenarioParams.model_validate(params)
        except ValueError:
            return None

    def _is_unique(
        self,
        candidate: ScenarioParams,
        seen: list[ScenarioParams],
        bounds: Mapping[str, tuple[float, float]],
    ) -> bool:
        vector = normalize(candidate, bounds)
        return all(
            np.max(np.abs(vector - normalize(other, bounds))) > self.epsilon for other in seen
        )

    def _random_unique(
        self,
        seen: list[ScenarioParams],
        bounds: Mapping[str, tuple[float, float]],
    ) -> ScenarioParams:
        for _ in range(10000):
            candidate = denormalize(self.rng.random(len(bounds)), bounds)
            if self._is_unique(candidate, seen, bounds):
                return candidate
        raise RuntimeError("could not find a unique scenario within 10000 attempts")

    def _exploit(
        self,
        successful: list[Trial],
        seen: list[ScenarioParams],
        bounds: Mapping[str, tuple[float, float]],
    ) -> ScenarioParams:
        best = max(successful, key=lambda trial: trial.result.risk)  # type: ignore[union-attr]
        center = normalize(best.params, bounds)
        for _ in range(10000):
            values = np.clip(center + self.rng.normal(0.0, 0.1, len(bounds)), 0.0, 1.0)
            candidate = denormalize(values, bounds)
            if self._is_unique(candidate, seen, bounds):
                return candidate
        return self._random_unique(seen, bounds)

    def propose(
        self, history: list[Trial], bounds: Mapping[str, tuple[float, float]], k: int
    ) -> tuple[str, list[ScenarioParams]]:
        if k < 0:
            raise ValueError("k must be non-negative")
        if k == 0:
            self.last_origins = []
            self.last_fallback = False
            self.last_usage = Usage()
            return "no proposals requested", []

        requested_llm = min(k, int(k * self.llm_fraction))
        hypothesis = ""
        raw_scenarios: list[dict[str, Any]] = []
        self.last_fallback = False
        self.last_usage = Usage()
        if requested_llm:
            try:
                hypothesis, raw_scenarios = self._request(history, bounds, requested_llm)
            except BudgetExceeded:
                raise
            except LLMError:
                hypothesis = "LLM unavailable; using deterministic fallback proposals."
                self.last_fallback = True

        seen = [trial.params for trial in history]
        proposals: list[ScenarioParams] = []
        origins: list[ProposalOrigin] = []
        for raw in raw_scenarios[:requested_llm]:
            candidate = self._sanitize(raw, bounds)
            if candidate is None or not self._is_unique(candidate, seen + proposals, bounds):
                self.last_fallback = True
                continue
            proposals.append(candidate)
            origins.append("llm")

        missing = k - len(proposals)
        successful = [trial for trial in history if trial.result is not None]
        exploit_count = (missing + 1) // 2 if successful else 0
        explore_count = missing - exploit_count
        for _ in range(exploit_count):
            candidate = self._exploit(successful, seen + proposals, bounds)
            proposals.append(candidate)
            origins.append("exploit")
        for _ in range(explore_count):
            candidate = self._random_unique(seen + proposals, bounds)
            proposals.append(candidate)
            origins.append("explore")

        if requested_llm and sum(origin == "llm" for origin in origins) < requested_llm:
            self.last_fallback = True
        self.last_origins = origins
        if len(proposals) != k:
            raise RuntimeError("adversary must return exactly k proposals")
        if not hypothesis:
            hypothesis = "Explore bounded perturbations and test which conditions raise risk."
        return hypothesis[:400], proposals
