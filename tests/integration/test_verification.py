from __future__ import annotations

import pytest

from omnisim.agents.baselines import scenario_bounds
from omnisim.core.verify import sample_scenarios, verify
from omnisim.schemas import SafeEnvelope
from omnisim.sim.detectors import DetectorConfig
from omnisim.sim.limits import load_urdf_limits
from omnisim.sim.runner import EpisodeRunner


@pytest.mark.slow
def test_real_sim_on_off_verification_accounts_for_eight_scenarios() -> None:
    scenarios = sample_scenarios(8, 712, scenario_bounds())
    envelope = SafeEnvelope(
        robot="franka_panda",
        version="test",
        limits={
            "max_payload_kg": 3.0,
            "max_latency_ms": 200.0,
            "torque_utilization_cap": 0.95,
            "speed_scale_max_by_payload_kg": [[0.0, 2.0], [3.0, 2.0]],
            "speed_scale_max_by_latency_ticks": [2.0] * 13,
        },
        controller={"kp_scale_max": 2.0},
        domain_randomization_bounds={},
        evidence=[],
        verification={},
    )
    with EpisodeRunner() as runner:
        result = verify(
            envelope,
            load_urdf_limits(),
            n=8,
            seed=712,
            workers=1,
            cfg=DetectorConfig(),
            runner=runner,
            scenarios=scenarios,
        )

    assert result.n_scenarios == 8
    assert result.executed_count == 8
    assert result.blocked_count == 0
    assert result.fail_rate_after == result.fail_rate_before
    assert result.completion_rate <= 1.0
    assert result.infrastructure_errors == 0
