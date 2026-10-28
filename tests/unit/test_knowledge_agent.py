import numpy as np
import pytest
import requests
from tavily.errors import TavilyKeylessLimitError

from omnisim.knowledge import compare
from omnisim.knowledge import tavily_agent as agent
from omnisim.knowledge.errors import KnowledgeUnavailable
from omnisim.knowledge.tavily_agent import ErrorsDraft, LimitDraft, LimitsDraft
from omnisim.llm.budget import LLMUnavailable
from omnisim.resilience import CircuitBreaker
from omnisim.schemas import Evidence, HardwareSpec
from omnisim.sim.limits import UrdfLimits


class FakeTavily:
    def search(self, **kwargs):
        return {"results": [{"url": "https://docs.franka.de/panda", "score": 0.9}]}

    def extract(self, **kwargs):
        return {
            "results": [
                {
                    "url": url,
                    "raw_content": "Synthetic Panda manual: joint 1 torque limit is 90 Nm. "
                    "Fault code E17 means joint drive fault.",
                }
                for url in kwargs["urls"]
            ]
        }


def test_build_spec_verifies_claims_and_drops_unknown_ids(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(agent, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(agent, "CALL_LOG_PATH", tmp_path / "calls.jsonl")
    monkeypatch.setattr(agent, "_client", lambda: FakeTavily())
    calls = 0

    def fake_chat(role, system, user, schema, **kwargs):
        nonlocal calls
        calls += 1
        if schema is LimitsDraft:
            draft = LimitsDraft(
                items=[
                    LimitDraft(
                        page_id="P1",
                        quantity="torque",
                        joint=1,
                        value=90,
                        unit="Nm",
                        quote="joint 1 torque limit is 90 Nm",
                    ),
                    LimitDraft(
                        page_id="P9",
                        quantity="velocity",
                        joint=1,
                        value=2,
                        unit="rad/s",
                        quote="joint 1 velocity limit is 2 rad/s",
                    ),
                    LimitDraft(
                        page_id="P1",
                        quantity="velocity",
                        joint=1,
                        value=2,
                        unit="rad/s",
                        quote="joint 1 torque limit is 90 Nm",
                    ),
                ]
            )
            return draft, None
        return ErrorsDraft(items=[]), None

    monkeypatch.setattr(agent, "chat_json", fake_chat)
    spec = agent.build_spec("franka_panda", run_id="synthetic")
    assert calls == 2
    assert len(spec.joint_torque_limits_nm) == 1
    assert spec.joint_torque_limits_nm[0].verified
    assert len(spec.joint_velocity_limits) == 1
    assert not spec.joint_velocity_limits[0].verified
    assert agent.verified_only(spec).joint_velocity_limits == []


def test_build_spec_ignores_model_verified_flag(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(agent, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(agent, "CALL_LOG_PATH", tmp_path / "calls.jsonl")
    monkeypatch.setattr(agent, "_client", lambda: FakeTavily())
    outputs = iter(
        [
            LimitsDraft.model_validate(
                {
                    "items": [
                        {
                            "page_id": "P1",
                            "quantity": "torque",
                            "joint": 1,
                            "value": 200,
                            "unit": "Nm",
                            "quote": "joint 1 torque limit is 90 Nm",
                            "verified": True,
                        }
                    ]
                }
            ),
            ErrorsDraft(items=[]),
        ]
    )
    monkeypatch.setattr(agent, "chat_json", lambda *args, **kwargs: (next(outputs), None))
    spec = agent.build_spec("franka_panda")
    assert not spec.joint_torque_limits_nm[0].verified


def test_build_spec_propagates_llm_errors(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(agent, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(agent, "CALL_LOG_PATH", tmp_path / "calls.jsonl")
    monkeypatch.setattr(agent, "_client", lambda: FakeTavily())

    def fail(*args, **kwargs):
        raise LLMUnavailable("synthetic failure")

    monkeypatch.setattr(agent, "chat_json", fail)
    with pytest.raises(LLMUnavailable, match="synthetic failure"):
        agent.build_spec("franka_panda")


def test_tavily_live_call_retries_transport_failure_four_times(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(agent, "_circuit", CircuitBreaker())
    errors = [requests.ConnectionError("offline") for _ in range(3)]
    calls = 0
    delays: list[float] = []

    def live(timeout: float):
        nonlocal calls
        calls += 1
        if errors:
            raise errors.pop(0)
        return {"results": []}

    result = agent._run_live(
        "search",
        "run-a",
        live,
        call_log_path=tmp_path / "calls.jsonl",
        call_id="call-a",
        sleep=delays.append,
        rng=lambda: 0.5,
    )

    assert result == {"results": []}
    assert calls == 4
    assert delays == [0.5, 1.0, 2.0]


def test_tavily_retry_after_is_capped(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(agent, "_circuit", CircuitBreaker())
    calls = 0
    delays: list[float] = []

    def live(timeout: float):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TavilyKeylessLimitError("slow down", retry_after_seconds=45)
        return {"results": []}

    agent._run_live(
        "search",
        "run-a",
        live,
        call_log_path=tmp_path / "calls.jsonl",
        call_id="call-a",
        sleep=delays.append,
        rng=lambda: 0,
    )

    assert delays == [30]


def test_tavily_cache_miss_without_credentials_is_unavailable(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(agent, "_circuit", CircuitBreaker())
    monkeypatch.setattr(agent, "_client", lambda: (_ for _ in ()).throw(ValueError("missing key")))

    with pytest.raises(KnowledgeUnavailable, match="client unavailable"):
        agent._search(
            "query",
            ("franka.de",),
            cache_dir=tmp_path / "cache",
            call_log_path=tmp_path / "calls.jsonl",
        )


def _spec(*items: Evidence, errors: list[Evidence] | None = None) -> HardwareSpec:
    return HardwareSpec(
        robot="franka_panda",
        joint_torque_limits_nm=[],
        joint_velocity_limits=[],
        joint_position_limits=list(items),
        error_taxonomy=errors or [],
    )


def test_compare_reports_real_mismatch_and_degree_conversion() -> None:
    limits = UrdfLimits(
        effort=np.array([87.0] * 7),
        velocity=np.array([2.175] * 7),
        lower=np.array([-1.0] * 7),
        upper=np.array([1.0] * 7),
    )
    torque = Evidence(
        claim="joint 1 torque",
        value=80,
        unit="N m",
        source_url="https://docs.test",
        quote="joint 1 torque limit is 80 N m",
        verified=True,
    )
    upper = Evidence(
        claim="joint 1 upper",
        value=60,
        unit="deg",
        source_url="https://docs.test",
        quote="joint 1 upper limit is 60 degrees",
        verified=True,
    )
    mismatches, not_compared = compare.compare_with_urdf(
        HardwareSpec(
            robot="franka_panda",
            joint_torque_limits_nm=[torque],
            joint_velocity_limits=[],
            joint_position_limits=[upper],
            error_taxonomy=[],
        ),
        limits,
    )
    assert [item.field for item in mismatches] == ["torque", "upper"]
    assert mismatches[0].joint == 0
    assert mismatches[0].sim_more_permissive is True
    assert mismatches[1].doc == pytest.approx(np.pi / 3)
    assert not_compared == []


def test_compare_skips_equal_unverified_and_unknown_units() -> None:
    limits = UrdfLimits(
        effort=np.array([87.0] * 7),
        velocity=np.array([2.175] * 7),
        lower=np.array([-1.0] * 7),
        upper=np.array([1.0] * 7),
    )
    equal = Evidence(
        claim="joint 1 torque",
        value=87,
        unit="Nm",
        source_url="https://docs.test",
        quote="joint 1 torque is 87 Nm",
        verified=True,
    )
    unknown = Evidence(
        claim="joint 1 upper",
        value=1,
        unit="furlong",
        source_url="https://docs.test",
        quote="joint 1 upper is 1 furlong",
        verified=True,
    )
    unverified = unknown.model_copy(update={"verified": False})
    spec = HardwareSpec(
        robot="franka_panda",
        joint_torque_limits_nm=[equal],
        joint_velocity_limits=[],
        joint_position_limits=[unknown, unverified],
        error_taxonomy=[],
    )
    mismatches, not_compared = compare.compare_with_urdf(spec, limits)
    assert mismatches == []
    assert not_compared == [unknown]


def test_compare_within_relative_tolerance_is_not_reported() -> None:
    limits = UrdfLimits(
        effort=np.array([87.0] * 7),
        velocity=np.array([2.175] * 7),
        lower=np.array([-1.0] * 7),
        upper=np.array([1.0] * 7),
    )
    close = Evidence(
        claim="joint 1 torque",
        value=87.05,
        unit="Nm",
        source_url="https://docs.test",
        quote="joint 1 torque is 87.05 Nm in this guide",
        verified=True,
    )
    mismatches, _ = compare.compare_with_urdf(_spec(close), limits)
    assert mismatches == []


def test_error_mapping_uses_verified_candidate_and_evidence_name(monkeypatch) -> None:
    evidence = Evidence(
        claim="Error E17",
        value="E17 drive fault",
        unit=None,
        source_url="https://docs.test",
        quote="Fault code E17 means joint drive fault.",
        verified=True,
    )
    spec = _spec(errors=[evidence, evidence.model_copy(update={"verified": False})])

    def fake_chat(role, system, user, schema, **kwargs):
        assert '<evidence id="E1">' in user
        assert "E2" not in user
        return agent.MappingsDraft(
            items=[
                agent.MappingDraft(kind="torque_saturation", evidence_id="E1", confidence="high")
            ]
        ), None

    monkeypatch.setattr(agent, "chat_json", fake_chat)
    result = agent.map_failures_to_real_errors(["torque_saturation", "collision"], spec)
    assert result["torque_saturation"].real_error_name == "E17 drive fault"
    assert result["torque_saturation"].evidence == evidence
    assert result["collision"].real_error_name is None


def test_error_mapping_invalid_id_is_no_match(monkeypatch) -> None:
    evidence = Evidence(
        claim="Error E17",
        value="E17",
        source_url="https://docs.test",
        quote="Fault code E17 is documented in this manual.",
        verified=True,
    )
    monkeypatch.setattr(
        agent,
        "chat_json",
        lambda *args, **kwargs: (
            agent.MappingsDraft(
                items=[agent.MappingDraft(kind="collision", evidence_id="E9", confidence="high")]
            ),
            None,
        ),
    )
    result = agent.map_failures_to_real_errors(["collision"], _spec(errors=[evidence]))
    assert result["collision"].real_error_name is None
    assert result["collision"].confidence is None


def test_error_mapping_rejects_invalid_failure_kind() -> None:
    with pytest.raises(ValueError):
        agent.map_failures_to_real_errors(["not_a_failure"], _spec())  # type: ignore[list-item]


def test_known_issue_hits_allow_github_as_corroboration_only(monkeypatch) -> None:
    received = {}

    def fake_search(query, domains, **kwargs):
        received["query"] = query
        received["domains"] = domains
        return [
            {"url": "https://github.com/franka/issue/1", "title": "Issue", "content": "Synthetic"}
        ]

    monkeypatch.setattr(agent, "_search", fake_search)
    hits = agent.search_known_issues("franka_panda", "collision")
    assert "Franka Emika Panda" in received["query"]
    assert received["domains"][-1] == "github.com"
    assert hits == [
        {"url": "https://github.com/franka/issue/1", "title": "Issue", "content": "Synthetic"}
    ]


def test_error_mapping_without_verified_candidates_is_no_match() -> None:
    result = agent.map_failures_to_real_errors(["collision"], _spec())
    assert result["collision"].real_error_name is None
    assert result["collision"].evidence is None
    assert result["collision"].confidence is None
