from collections.abc import Callable

import pytest

from omnisim.schemas import ScenarioParams
from omnisim.sim.limits import UrdfLimits, load_urdf_limits
from omnisim.sim.panda_env import PandaSim
from omnisim.sim.telemetry import EpisodeTelemetry


@pytest.fixture(scope="session")
def urdf_limits() -> UrdfLimits:
    return load_urdf_limits()


@pytest.fixture
def simulate(
    urdf_limits: UrdfLimits,
) -> Callable[[ScenarioParams], EpisodeTelemetry]:
    def run(params: ScenarioParams) -> EpisodeTelemetry:
        with PandaSim(params, urdf_limits.effort, urdf_limits.velocity) as sim:
            return sim.run()

    return run
