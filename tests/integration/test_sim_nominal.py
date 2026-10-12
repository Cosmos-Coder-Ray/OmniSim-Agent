from collections.abc import Callable
from dataclasses import fields

import numpy as np

from omnisim.schemas import ScenarioParams
from omnisim.sim.constants import EPISODE_TICKS
from omnisim.sim.telemetry import EpisodeTelemetry


def test_sim_nominal(
    simulate: Callable[[ScenarioParams], EpisodeTelemetry],
) -> None:
    telemetry = simulate(ScenarioParams(seed=17))

    assert len(telemetry.t) == EPISODE_TICKS
    assert telemetry.t[-1] == 6.0
    assert np.linalg.norm(telemetry.tracking_error[-1]) <= 0.05
    for field in fields(telemetry):
        assert not np.isnan(getattr(telemetry, field.name)).any()
