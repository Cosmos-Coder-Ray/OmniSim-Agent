from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnisim.agents.baselines import scenario_bounds
from omnisim.core.clustering import ClusterSummary, effective_params
from omnisim.knowledge.verifier import verified_only
from omnisim.llm import nebius
from omnisim.llm.budget import BudgetExceeded, LLMError, Usage
from omnisim.llm.nebius import LLMOutputError
from omnisim.schemas import (
    DiagnosisReport,
    EpisodeResult,
    ErrorMapping,
    Evidence,
    HardwareSpec,
    NumericClaim,
    RecommendedConstraint,
    ScenarioParams,
)

PROMPT_PATH = Path(__file__).parents[1] / "llm" / "prompts" / "diagnosis_system.md"
_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_JOINT_LABEL = re.compile(r"\bjoint\s+[1-7]\b", re.IGNORECASE)
_DIGIT = re.compile(r"\d")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

ChatJSON = Callable[..., tuple[BaseModel, Usage]]


class DiagnosisError(RuntimeError):
    pass


class _DiagnosisDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root_cause: str
    mechanism: str
    numeric_claims: list[NumericClaim] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]
    recommended_constraints: list[RecommendedConstraint] = Field(default_factory=list)
    known_issue_ids: list[str] = Field(default_factory=list)


def _round(value: float | int) -> float | int:
    if isinstance(value, int):
        return value
    rounded = round(value, 6)
    return 0.0 if rounded == 0 else rounded


def _round_data(value: object) -> object:
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return _round(value)
    if isinstance(value, dict):
        return {key: _round_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_data(item) for item in value]
    return value


def measured_facts(
    cluster: ClusterSummary,
    representative: EpisodeResult,
    minimal: ScenarioParams,
) -> dict:
    event = max(
        (
            item
            for item in representative.failures
            if item.kind == cluster.kind and item.joint == cluster.joint
        ),
        key=lambda item: item.severity,
    )
    representative_params = effective_params(representative.params)
    minimal_params = effective_params(minimal)
    return {
        "cluster_id": cluster.cluster_id,
        "kind": cluster.kind,
        "joint_label": f"joint {cluster.joint + 1}" if cluster.joint is not None else "global",
        "cluster_size": len(cluster.episode_ids),
        "best_risk": _round(cluster.best_risk),
        "max_severity": _round(cluster.max_severity),
        "representative": {
            "params": {
                name: _round(getattr(representative_params, name))
                for name in ScenarioParams.model_fields
                if name != "seed"
            },
            "risk": _round(representative.risk),
            "max_tau_ratio": _round(representative.max_tau_ratio),
            "max_vel_ratio": _round(representative.max_vel_ratio),
            "final_ee_error_m": _round(representative.final_ee_error_m),
            "min_manipulability": _round(representative.min_manipulability),
            "event": {
                "kind": event.kind,
                "severity": _round(event.severity),
                "time_s": _round(event.t),
                "detail": _round_data(event.detail),
            },
        },
        "minimal": {
            "params": {
                name: _round(getattr(minimal_params, name))
                for name in ScenarioParams.model_fields
                if name != "seed"
            }
        },
    }


def _numeric_values(value: object, path: str = "") -> dict[str, float]:
    values: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            values.update(_numeric_values(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{path}.{index}" if path else str(index)
            values.update(_numeric_values(item, child))
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        values[path] = float(value)
    return values


def _valid_prose(text: str) -> bool:
    return _DIGIT.search(_JOINT_LABEL.sub("", text)) is None


def _clean_text(text: str) -> str:
    text = _URL.sub("", text)
    text = _NUMBER.sub(" ", text)
    return " ".join(text.split())


def _evidence_candidates(spec: HardwareSpec) -> tuple[dict[str, Evidence], list[dict[str, object]]]:
    verified = verified_only(spec)
    entries = [
        *verified.joint_torque_limits_nm,
        *verified.joint_velocity_limits,
        *verified.joint_position_limits,
        *verified.error_taxonomy,
    ]
    by_id: dict[str, Evidence] = {}
    prompt_entries: list[dict[str, object]] = []
    for index, evidence in enumerate(entries, 1):
        evidence_id = f"V{index}"
        by_id[evidence_id] = evidence
        prompt_entries.append(
            {
                "id": evidence_id,
                "claim": _URL.sub("", evidence.claim),
                "value": _URL.sub("", evidence.value)
                if isinstance(evidence.value, str)
                else evidence.value,
                "unit": _URL.sub("", evidence.unit) if evidence.unit else None,
                "quote": _URL.sub("", evidence.quote),
            }
        )
    return by_id, prompt_entries


def _known_issue_candidates(
    known_issues: list[dict[str, str]],
) -> tuple[dict[str, str], list[dict[str, str]]]:
    urls: dict[str, str] = {}
    prompt_entries: list[dict[str, str]] = []
    for index, issue in enumerate(known_issues, 1):
        url = issue.get("url", "")
        if not url.startswith(("https://", "http://")):
            continue
        issue_id = f"K{index}"
        urls[issue_id] = url
        content = _clean_text(f"{issue.get('title', '')} {issue.get('content', '')}")
        if content:
            prompt_entries.append({"id": issue_id, "content": content})
    return urls, prompt_entries


def _resolve(
    draft: _DiagnosisDraft,
    *,
    cluster: ClusterSummary,
    facts: dict,
    mapping: ErrorMapping | None,
    evidence_by_id: dict[str, Evidence],
    known_issue_urls: dict[str, str],
) -> DiagnosisReport:
    for text in [
        draft.root_cause,
        draft.mechanism,
        *(claim.text for claim in draft.numeric_claims),
    ]:
        if not text.strip() or not _valid_prose(text):
            raise ValueError("diagnosis prose must be non-empty and contain no unsupported digits")

    numeric_facts = _numeric_values(facts)
    for claim in draft.numeric_claims:
        value = numeric_facts.get(claim.basis)
        if value is None or not math.isfinite(claim.value):
            raise ValueError(f"unsupported numeric claim basis: {claim.basis}")
        if not math.isclose(claim.value, value, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"numeric claim does not match measured fact: {claim.basis}")

    bounds = scenario_bounds()
    for constraint in draft.recommended_constraints:
        if constraint.parameter not in bounds or constraint.parameter == "seed":
            raise ValueError(f"unsupported constraint parameter: {constraint.parameter}")
        value = numeric_facts.get(constraint.basis)
        if value is None or not math.isfinite(constraint.value):
            raise ValueError(f"unsupported constraint basis: {constraint.basis}")
        if not math.isclose(constraint.value, value, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"constraint does not match measured fact: {constraint.basis}")
        low, high = bounds[constraint.parameter]
        if not low <= constraint.value <= high:
            raise ValueError(f"constraint is outside scenario bounds: {constraint.parameter}")

    if any(evidence_id not in evidence_by_id for evidence_id in draft.evidence_ids):
        raise ValueError("diagnosis referenced unverified or unknown evidence")
    if any(issue_id not in known_issue_urls for issue_id in draft.known_issue_ids):
        raise ValueError("diagnosis referenced an unknown known-issue id")

    evidence_refs = list(dict.fromkeys(draft.evidence_ids))
    issue_ids = list(dict.fromkeys(draft.known_issue_ids))
    return DiagnosisReport(
        cluster_id=cluster.cluster_id,
        root_cause=draft.root_cause.strip(),
        mechanism=draft.mechanism.strip(),
        numeric_claims=draft.numeric_claims,
        real_error_mapping=mapping,
        evidence_refs=[evidence_by_id[evidence_id] for evidence_id in evidence_refs],
        confidence=draft.confidence,
        recommended_constraints=draft.recommended_constraints,
        corroborating_urls=[known_issue_urls[issue_id] for issue_id in issue_ids],
    )


def diagnose(
    cluster: ClusterSummary,
    representative: EpisodeResult,
    minimal: ScenarioParams,
    spec: HardwareSpec,
    *,
    mapping: ErrorMapping | None = None,
    known_issues: list[dict[str, str]] | None = None,
    chat_json: ChatJSON | None = None,
    run_id: str | None = None,
) -> DiagnosisReport:
    facts = measured_facts(cluster, representative, minimal)
    evidence_by_id, evidence_prompt = _evidence_candidates(spec)
    known_issue_urls, known_issue_prompt = _known_issue_candidates(known_issues or [])
    user = json.dumps(
        {
            "measured_facts": facts,
            "verified_evidence": evidence_prompt,
            "known_issues": known_issue_prompt,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    system = PROMPT_PATH.read_text(encoding="utf-8")
    request = chat_json or nebius.chat_json

    def ask(content: str) -> _DiagnosisDraft:
        response, _ = request(
            "deep",
            system,
            content,
            _DiagnosisDraft,
            temperature=0,
            max_tokens=4096,
            retries=0,
            run_id=run_id,
        )
        if not isinstance(response, _DiagnosisDraft):
            response = _DiagnosisDraft.model_validate(response)
        return response

    try:
        draft = ask(user)
    except BudgetExceeded:
        raise
    except LLMOutputError:
        try:
            draft = ask(
                f"{user}\n\nRepair the previous response so it matches the required JSON schema."
            )
        except BudgetExceeded:
            raise
        except (LLMError, ValueError) as repair_error:
            raise DiagnosisError(f"diagnosis repair failed: {repair_error}") from repair_error
    except LLMError as error:
        raise DiagnosisError(f"diagnosis request failed: {error}") from error

    try:
        return _resolve(
            draft,
            cluster=cluster,
            facts=facts,
            mapping=mapping,
            evidence_by_id=evidence_by_id,
            known_issue_urls=known_issue_urls,
        )
    except ValueError as error:
        try:
            repaired = ask(f"{user}\n\nRepair the previous response. Validation error: {error}")
            return _resolve(
                repaired,
                cluster=cluster,
                facts=facts,
                mapping=mapping,
                evidence_by_id=evidence_by_id,
                known_issue_urls=known_issue_urls,
            )
        except BudgetExceeded:
            raise
        except (LLMError, ValueError) as repair_error:
            raise DiagnosisError(f"diagnosis validation failed: {repair_error}") from repair_error
