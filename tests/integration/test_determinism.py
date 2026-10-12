from collections.abc import Callable

import numpy as np

from omnisim.schemas import ScenarioParams
from omnisim.sim.telemetry import EpisodeTelemetry


def test_determinism(
    simulate: Callable[[ScenarioParams], EpisodeTelemetry],
) -> None:
    params = ScenarioParams(sensor_noise_rad=0.01, push_force_n=15.0, seed=41)
    first = simulate(params)
    second = simulate(params)

    for name in first.__dataclass_fields__:
        assert np.array_equal(getattr(first, name), getattr(second, name))
