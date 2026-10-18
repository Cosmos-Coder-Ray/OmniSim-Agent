from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from omnisim.agents.diagnostician import DiagnosisError, diagnose, measured_facts
from omnisim.core.clustering import ClusterSummary
from omnisim.llm.budget import Usage
from omnisim.schemas import (
    DiagnosisReport,
    EpisodeResult,
    ErrorMapping,
    Evidence,
    FailureEvent,
    HardwareSpec,
    ScenarioParams,
)


def inputs() -> tuple[ClusterSummary, EpisodeResult, ScenarioParams, HardwareSpec]:
    params = ScenarioParams(payload_kg=0.812345678, seed=17)
    episode = EpisodeResult(
        episode_id="episode",
        params=params,
        failures=[
            FailureEvent(
                kind="tracking_error",
                t=1.23456789,
                joint=1,
                severity=1.55555555,
            )
        ],
        risk=1.23456789,
        max_tau_ratio=0.987654321,
        max_vel_ratio=0.876543219,
        final_ee_error_m=0.065432198,
        min_manipulability=0.012345678,
    )
    cluster = ClusterSummary(
        cluster_id="tracking_error:j1",
        kind="tracking_error",
        joint=1,
        episode_ids=(episode.episode_id,),
        best_risk=episode.risk,
        max_severity=1.55555555,
    )
    verified = Evidence(
        claim="joint 2 torque",
        value=87.123,
        unit="Nm",
        source_url="https://manufacturer.example/spec",
        quote="Joint 2 torque is 87.123 Nm in the supplied manufacturer table.",
        verified=True,
    )
    unverified = Evidence(
        claim="joint 2 velocity",
        value=2.0,
        unit="rad/s",
        source_url="https://manufacturer.example/draft",
        quote="Unverified velocity value.",
        verified=False,
    )
    spec = HardwareSpec(
        robot="Panda",
        joint_torque_limits_nm=[verified],
        joint_velocity_limits=[],
        joint_position_limits=[],
        error_taxonomy=[unverified],
    )
    return cluster, episode, ScenarioParams(payload_kg=0.75, seed=17), spec


def draft(*, numeric_value: float = 1.234568, evidence_ids=None, root_cause=None) -> dict[str, Any]:
    return {
        "root_cause": root_cause or "payload perturbation increased tracking error at joint 2",
        "mechanism": "the tested motion deviated after the perturbation",
        "numeric_claims": [
            {
                "text": "representative risk",
                "basis": "representative.risk",
                "value": numeric_value,
            }
        ],
        "evidence_ids": ["V1"] if evidence_ids is None else evidence_ids,
        "confidence": "medium",
        "recommended_constraints": [
            {
                "parameter": "payload_kg",
                "bound": "max",
                "value": 0.75,
                "basis": "minimal.params.payload_kg",
            }
        ],
        "known_issue_ids": ["K1"],
    }


def fake_chat(outputs: list[dict[str, Any]]):
    calls: list[tuple[str, str, str, dict[str, object]]] = []

    def chat(role, system, user, response_model: type[BaseModel], **kwargs):
        calls.append((role, system, user, kwargs))
        return response_model.model_validate(outputs.pop(0)), Usage()

    return chat, calls


def test_facts_are_rounded_once_and_resolved_to_verified_code_data() -> None:
    cluster, representative, minimal, spec = inputs()
    facts = measured_facts(cluster, representative, minimal)
    chat, calls = fake_chat([draft()])
    mapping = ErrorMapping(kind="tracking_error", real_error_name="documented analogue")
    known_issues = [
        {
            "url": "https://manufacturer.example/issues/42",
            "title": "Panda version 2 issue",
            "content": "Joint 2 issue 42 at https://manufacturer.example/private",
        }
    ]

    report = diagnose(
        cluster,
        representative,
        minimal,
        spec,
        mapping=mapping,
        known_issues=known_issues,
        chat_json=chat,
    )

    prompt = json.loads(calls[0][2])
    assert prompt["measured_facts"] == facts
    assert prompt["measured_facts"]["representative"]["risk"] == 1.234568
    assert prompt["verified_evidence"] == [
        {
            "id": "V1",
            "claim": "joint 2 torque",
            "value": 87.123,
            "unit": "Nm",
            "quote": "Joint 2 torque is 87.123 Nm in the supplied manufacturer table.",
        }
    ]
    assert "manufacturer.example" not in calls[0][2]
    assert "42" not in prompt["known_issues"][0]["content"]
    assert report.real_error_mapping == mapping
    assert report.evidence_refs == spec.joint_torque_limits_nm
    assert report.corroborating_urls == [known_issues[0]["url"]]
    assert calls[0][0] == "deep"
    assert calls[0][3] == {
        "temperature": 0,
        "max_tokens": 4096,
        "retries": 0,
        "run_id": None,
    }


@pytest.mark.parametrize(
    ("bad_output", "reason"),
    [
        ({"numeric_value": 1.2347}, "numeric"),
        ({"evidence_ids": ["V2"]}, "evidence"),
        ({"root_cause": "risk increased by 4"}, "digits"),
    ],
)
def test_invalid_diagnosis_gets_one_repair_attempt(bad_output: dict[str, Any], reason: str) -> None:
    cluster, representative, minimal, spec = inputs()
    invalid = draft(**bad_output)
    chat, calls = fake_chat([invalid, draft()])

    report = diagnose(
        cluster,
        representative,
        minimal,
        spec,
        known_issues=[{"url": "https://manufacturer.example/issues/42", "content": "qualitative"}],
        chat_json=chat,
    )

    assert isinstance(report, DiagnosisReport)
    assert len(calls) == 2, reason
    assert (
        json.loads(calls[0][2])["measured_facts"]
        == json.loads(calls[1][2].split("\n\nRepair")[0])["measured_facts"]
    )


def test_invalid_diagnosis_after_repair_fails() -> None:
    cluster, representative, minimal, spec = inputs()
    chat, calls = fake_chat([draft(numeric_value=5.0), draft(numeric_value=5.0)])

    with pytest.raises(DiagnosisError, match="diagnosis validation failed"):
        diagnose(cluster, representative, minimal, spec, chat_json=chat)

    assert len(calls) == 2
