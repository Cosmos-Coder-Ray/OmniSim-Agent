from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from omnisim.agents.adversary import AdversaryResponse, NemotronAdversary
from omnisim.agents.baselines import Trial, normalize, scenario_bounds
from omnisim.llm.budget import BudgetExceeded, LLMUnavailable, Usage
from omnisim.schemas import EpisodeResult, ScenarioParams


def _trial(params: ScenarioParams, risk: float = 0.5) -> Trial:
    return Trial(
        params,
        EpisodeResult(
            episode_id="ep_0",
            params=params,
            failures=[],
            risk=risk,
            max_tau_ratio=0.0,
            max_vel_ratio=0.0,
            final_ee_error_m=0.0,
            min_manipulability=0.0,
        ),
        None,
    )


def _chat(
    response: AdversaryResponse,
) -> Callable[..., tuple[AdversaryResponse, Usage]]:
    def call(*args: object, **kwargs: object) -> tuple[AdversaryResponse, Usage]:
        return response, Usage(3, 4)

    return call


def test_adversary_sanitizes_partial_unknown_and_seed_fields() -> None:
    adversary = NemotronAdversary(
        seed=2,
        chat_json=_chat(
            AdversaryResponse(
                hypothesis="test",
                scenarios=[{"payload_kg": 99, "seed": 55, "unknown": 4}],
            )
        ),
    )

    _, proposals = adversary.propose([], scenario_bounds(), 1)

    assert proposals[0].payload_kg == 3.0
    assert proposals[0].seed == 0
    assert adversary.last_origins == ["llm"]


def test_adversary_rejects_non_finite_candidates_and_fills_batch() -> None:
    adversary = NemotronAdversary(
        seed=7,
        chat_json=_chat(
            AdversaryResponse(
                hypothesis="invalid candidate",
                scenarios=[{"payload_kg": float("nan")}],
            )
        ),
    )

    _, proposals = adversary.propose([], scenario_bounds(), 4)

    assert len(proposals) == 4
    assert adversary.last_origins == ["explore"] * 4
    assert adversary.last_fallback
    assert len({tuple(normalize(item, scenario_bounds())) for item in proposals}) == 4


def test_adversary_deduplicates_and_splits_fallback_evenly() -> None:
    base = ScenarioParams(payload_kg=2.0)
    adversary = NemotronAdversary(
        seed=5,
        chat_json=_chat(
            AdversaryResponse(
                hypothesis="same candidate",
                scenarios=[{"payload_kg": 2.0}] * 4,
            )
        ),
    )

    _, proposals = adversary.propose([_trial(base)], scenario_bounds(), 4)

    assert len(proposals) == 4
    assert adversary.last_origins == ["exploit", "exploit", "explore", "explore"]
    assert adversary.last_fallback
    assert all(
        np.max(np.abs(normalize(item, scenario_bounds()) - normalize(base, scenario_bounds())))
        > adversary.epsilon
        for item in proposals
    )


def test_llm_fraction_zero_makes_no_llm_call() -> None:
    def fail(*args: object, **kwargs: object) -> tuple[AdversaryResponse, Usage]:
        raise AssertionError("llm call was made")

    adversary = NemotronAdversary(seed=11, llm_fraction=0.0, chat_json=fail)

    _, proposals = adversary.propose([], scenario_bounds(), 3)

    assert len(proposals) == 3
    assert adversary.last_origins == ["explore"] * 3


def test_adversary_falls_back_on_llm_error_and_caps_hypothesis() -> None:
    response = AdversaryResponse(hypothesis="x" * 600, scenarios=[{"push_force_n": 4}])
    adversary = NemotronAdversary(seed=3, chat_json=_chat(response))

    hypothesis, _ = adversary.propose([], scenario_bounds(), 1)

    assert len(hypothesis) <= 400
    failing = NemotronAdversary(
        seed=3,
        chat_json=lambda *args, **kwargs: (_ for _ in ()).throw(LLMUnavailable("offline")),
    )
    _, proposals = failing.propose([], scenario_bounds(), 2)
    assert len(proposals) == 2
    assert failing.last_fallback
    assert len(failing.last_origins) == 2


def test_budget_exceeded_propagates() -> None:
    adversary = NemotronAdversary(
        seed=3,
        chat_json=lambda *args, **kwargs: (_ for _ in ()).throw(BudgetExceeded("cap")),
    )

    with pytest.raises(BudgetExceeded):
        adversary.propose([], scenario_bounds(), 1)


@pytest.mark.parametrize("k", [1, 2, 4, 5])
def test_last_origins_matches_each_requested_batch(k: int) -> None:
    adversary = NemotronAdversary(seed=23, llm_fraction=0.0)

    _, proposals = adversary.propose([], scenario_bounds(), k)

    assert len(proposals) == k
    assert len(adversary.last_origins) == k
