from __future__ import annotations

import numpy as np
from pydantic import ValidationError

from omnisim.agents.mitigator import (
    EnvelopeDraft,
    _ConstraintDraft,
    load_domain_bounds,
    mitigate,
)
from omnisim.core.shrinker import ShrinkResult
from omnisim.llm.budget import LLMError, Usage
from omnisim.schemas import DiagnosisReport, Evidence, HardwareSpec
from omnisim.sim.detectors import DetectorConfig
from omnisim.sim.limits import UrdfLimits


def _spec() -> HardwareSpec:
    return HardwareSpec(
        robot="franka_panda",
        joint_torque_limits_nm=[
            Evidence(
                claim="joint 1 torque",
                value=87.0,
                unit="Nm",
                source_url="https://manufacturer.example/limits",
                quote="The rated torque is 87 Nm for joint 1.",
                verified=True,
            ),
            Evidence(
                claim="joint 2 torque",
                value=99.0,
                unit="Nm",
                source_url="https://manufacturer.example/unverified",
                quote="Unverified value.",
                verified=False,
            ),
        ],
        joint_velocity_limits=[],
        joint_position_limits=[],
        error_taxonomy=[],
    )


def _limits() -> UrdfLimits:
    ones = np.ones(7)
    return UrdfLimits(ones, ones, -ones, ones)


def _facts() -> dict[str, dict]:
    return {
        "cluster.1": {
            "representative": {
                "params": {
                    "payload_kg": 2.5,
                    "latency_ticks": 6,
                    "kp_scale": 1.6,
                    "link_mass_scale": 1.4,
                }
            }
        }
    }


def _draft() -> EnvelopeDraft:
    return EnvelopeDraft(
        constraints=[
            _ConstraintDraft(
                cluster_id="cluster.1",
                parameter="payload_kg",
                bound="max",
                value=2.5,
                basis="representative.params.payload_kg",
            ),
            _ConstraintDraft(
                cluster_id="cluster.1",
                parameter="latency_ticks",
                bound="max",
                value=6,
                basis="representative.params.latency_ticks",
            ),
            _ConstraintDraft(
                cluster_id="cluster.1",
                parameter="kp_scale",
                bound="max",
                value=1.6,
                basis="representative.params.kp_scale",
            ),
            _ConstraintDraft(
                cluster_id="cluster.1",
                parameter="link_mass_scale",
                bound="max",
                value=1.4,
                basis="representative.params.link_mass_scale",
            ),
        ],
        evidence_ids=["V1"],
    )


def _chat(draft: EnvelopeDraft):
    def chat_json(role, system, user, schema, **kwargs):
        assert role == "deep"
        assert schema is EnvelopeDraft
        assert kwargs["temperature"] == 0
        return draft, Usage()

    return chat_json


def _predictor(payload: float, speed: float, limits: UrdfLimits) -> float:
    return 0.1 * payload + 0.2 * speed


def test_mitigator_resolves_facts_filters_evidence_and_owns_shield_fields() -> None:
    report = DiagnosisReport(
        cluster_id="cluster.1",
        root_cause="the payload increased risk",
        mechanism="the tracked motion lagged",
        numeric_claims=[],
        real_error_mapping=None,
        evidence_refs=[],
        confidence="medium",
        recommended_constraints=[],
        corroborating_urls=[],
    )
    envelope = mitigate(
        robot="franka_panda",
        spec=_spec(),
        diagnoses={"cluster.1": report},
        shrinks={},
        facts_by_cluster=_facts(),
        limits=_limits(),
        domain_bounds={
            "payload_kg": (0.0, 2.0),
            "link_mass_scale": (0.7, 1.5),
            "joint_damping": (0.0, 5.0),
            "latency_ticks": (0.0, 12.0),
            "sensor_noise_rad": (0.0, 0.02),
            "torque_limit_scale": (0.5, 1.0),
            "speed_scale": (0.5, 2.0),
            "kp_scale": (0.5, 2.0),
            "push_force_n": (0.0, 40.0),
            "push_time_s": (0.0, 5.0),
        },
        chat_json=_chat(_draft()),
        predictor=_predictor,
    )

    assert envelope.limits["max_payload_kg"] == 2.0
    assert envelope.limits["max_latency_ms"] == 6 * 1000 / 60
    assert envelope.controller["kp_scale_max"] == 1.6
    assert envelope.limits["torque_utilization_cap"] == DetectorConfig().tau_ratio_threshold
    assert envelope.limits["speed_scale_max_by_payload_kg"][-1][0] == 2.0
    assert [item.claim for item in envelope.evidence] == ["joint 1 torque"]
    assert any("link_mass_scale" in item for item in envelope.verification["monitoring_advisories"])
    assert envelope.verification["mitigator_fallback"] is False


def test_mitigator_repairs_invalid_basis_then_falls_back_to_shrunk_values() -> None:
    calls = 0

    def invalid_chat(role, system, user, schema, **kwargs):
        nonlocal calls
        calls += 1
        return EnvelopeDraft(
            constraints=[
                _ConstraintDraft(
                    cluster_id="cluster.1",
                    parameter="payload_kg",
                    bound="max",
                    value=2.5,
                    basis="representative.params.push_force_n",
                )
            ]
        ), Usage()

    from omnisim.schemas import ScenarioParams

    shrinks = {
        "cluster.1": ShrinkResult(
            minimal=ScenarioParams(payload_kg=1.0, latency_ticks=6, kp_scale=1.5),
            sims_used=4,
            converged=True,
            trace=[],
        )
    }
    envelope = mitigate(
        robot="franka_panda",
        spec=_spec(),
        diagnoses={},
        shrinks=shrinks,
        facts_by_cluster=_facts(),
        limits=_limits(),
        chat_json=invalid_chat,
        predictor=_predictor,
    )

    assert calls == 2
    assert envelope.verification["mitigator_fallback"] is True
    assert envelope.limits["max_payload_kg"] == 0.9
    assert envelope.limits["max_latency_ms"] == 5.4 * 1000 / 60
    assert envelope.controller["kp_scale_max"] == 1.4
    assert envelope.evidence == []


def test_mitigator_falls_back_when_chat_fails() -> None:
    def fail(*args, **kwargs):
        raise LLMError("offline")

    envelope = mitigate(
        robot="franka_panda",
        spec=_spec(),
        diagnoses={},
        shrinks={},
        facts_by_cluster=_facts(),
        limits=_limits(),
        chat_json=fail,
        predictor=_predictor,
    )

    assert envelope.verification["mitigator_fallback"] is True
    assert envelope.limits["max_payload_kg"] == 2.7


def test_envelope_draft_rejects_code_owned_fields() -> None:
    try:
        EnvelopeDraft.model_validate({"limits": {"max_payload_kg": 3.0}})
    except ValidationError:
        pass
    else:
        raise AssertionError("LLM draft accepted a code-owned envelope field")


def test_domain_bounds_load_and_validate(tmp_path) -> None:
    path = tmp_path / "domain.json"
    path.write_text(
        '{"payload_kg":[0,2],"link_mass_scale":[0.7,1.5],"joint_damping":[0,5],'
        '"latency_ticks":[0,12],"sensor_noise_rad":[0,0.02],"torque_limit_scale":[0.5,1],'
        '"speed_scale":[0.5,2],"kp_scale":[0.5,2],"push_force_n":[0,40],'
        '"push_time_s":[0,5]}',
        encoding="utf-8",
    )
    assert load_domain_bounds(path)["payload_kg"] == (0.0, 2.0)
    path.write_text('{"payload_kg":[-1,2]}', encoding="utf-8")
    try:
        load_domain_bounds(path)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid domain bounds were accepted")
