from __future__ import annotations

from omnisim.agents.adversary import ChatJSON, NemotronAdversary
from omnisim.agents.baselines import CMAESStrategy, RandomStrategy, Strategy
from omnisim.schemas import StrategyName


def make_strategy(
    name: StrategyName,
    *,
    seed: int,
    llm_fraction: float = 1.0,
    budget: int | None = None,
    run_id: str | None = None,
    chat_json: ChatJSON | None = None,
) -> Strategy:
    strategy_seed = seed % (2**32)
    if name == "nemotron":
        return NemotronAdversary(
            seed=strategy_seed,
            llm_fraction=llm_fraction,
            total_budget=budget,
            run_id=run_id,
            chat_json=chat_json,
        )
    if name == "random":
        return RandomStrategy(strategy_seed)
    if name == "cmaes":
        return CMAESStrategy(strategy_seed)
    raise ValueError(f"unknown strategy: {name}")
