from __future__ import annotations

import hashlib

from omnisim.agents.adversary import AdversaryResponse, NemotronAdversary
from omnisim.agents.baselines import Trial, scenario_bounds
from omnisim.llm.budget import Usage
from omnisim.schemas import EpisodeResult, FailureEvent, ScenarioParams


def test_adversary_request_matches_fixed_history_golden() -> None:
    captured: dict[str, object] = {}

    def chat_json(role, system, user, response_model, **kwargs):
        captured.update(role=role, system=system, user=user, kwargs=kwargs)
        return AdversaryResponse(
            hypothesis="[mock] test payload and latency", scenarios=[]
        ), Usage()

    params = ScenarioParams(payload_kg=1.0, latency_ticks=2, seed=8)
    result = EpisodeResult(
        episode_id="ep_000",
        params=params,
        failures=[
            FailureEvent(kind="tracking_error", t=1.5, joint=2, severity=1.1),
        ],
        risk=1.1,
        max_tau_ratio=0.7,
        max_vel_ratio=0.8,
        final_ee_error_m=0.06,
        min_manipulability=0.03,
    )
    history = [
        Trial(params=params, result=result, error=None),
        Trial(params=ScenarioParams(seed=9), result=None, error="WorkerTimeout: timed out"),
    ]
    strategy = NemotronAdversary(seed=17, total_budget=10, chat_json=chat_json)

    strategy._request(history, scenario_bounds(), 2)

    assert captured["role"] == "fast"
    assert hashlib.sha256(str(captured["system"]).encode()).hexdigest() == (
        "8e003f07ecc0192471c4e822eda72b1666e360e2e18e46250acf97b938a689d7"
    )
    assert captured["user"] == (
        '{"bounds":{"joint_damping":[0.0,5.0],"kp_scale":[0.5,2.0],'
        '"latency_ticks":[0.0,12.0],"link_mass_scale":[0.7,1.5],'
        '"payload_kg":[0.0,3.0],"push_force_n":[0.0,40.0],"push_time_s":[0.0,5.0],'
        '"sensor_noise_rad":[0.0,0.02],"speed_scale":[0.5,2.0],'
        '"torque_limit_scale":[0.5,1.0]},"failure_tally":[{"best_risk":1.1,"count":1,'
        '"joint":2,"kind":"tracking_error"}],"recent_successful_trials":[{"failures":'
        '[{"joint":2,"kind":"tracking_error"}],"params":{"joint_damping":0.0,'
        '"kp_scale":1.0,"latency_ticks":2,"link_mass_scale":1.0,"payload_kg":1.0,'
        '"push_force_n":0.0,"push_time_s":2.0,"sensor_noise_rad":0.0,"speed_scale":1.0,'
        '"torque_limit_scale":1.0},"risk":1.1}],"remaining_budget":8,"requested":2}'
    )
    assert captured["kwargs"] == {"temperature": 0.7, "max_tokens": 4096}
