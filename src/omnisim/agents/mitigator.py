from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnisim.agents.baselines import PARAM_NAMES, scenario_bounds
from omnisim.agents.diagnostician import measured_facts
from omnisim.core.clustering import ClusterSummary
from omnisim.core.shield import build_latency_speed_table, build_payload_speed_table
from omnisim.core.shrinker import ShrinkResult
from omnisim.core.verify import VerificationResult
from omnisim.knowledge.verifier import verified_only
from omnisim.llm import nebius
from omnisim.llm.budget import BudgetExceeded, LLMError, Usage
from omnisim.schemas import DiagnosisReport, EpisodeResult, Evidence, HardwareSpec, SafeEnvelope
from omnisim.sim.constants import TICK_MS
from omnisim.sim.detectors import DetectorConfig
from omnisim.sim.limits import UrdfLimits, load_urdf_limits
from omnisim.sim.torque import InverseDynamicsPredictor

PROMPT_PATH = Path(__file__).parents[1] / "llm" / "prompts" / "mitigation_system.md"
ChatJSON = Callable[..., tuple[BaseModel, Usage]]
TorquePredictor = Callable[[float, float, UrdfLimits], float]


class _ConstraintDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    parameter: str
    bound: Literal["max", "min"]
    value: float
    basis: str


class EnvelopeDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    constraints: list[_ConstraintDraft] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class MitigationError(RuntimeError):
    pass


def _numeric_values(value: object, path: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            output.update(_numeric_values(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{path}.{index}" if path else str(index)
            output.update(_numeric_values(item, child))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[path] = float(value)
    return output


def facts_from_campaign(
    clusters: Sequence[ClusterSummary],
    episodes: Sequence[EpisodeResult],
    shrinks: Mapping[str, ShrinkResult],
) -> dict[str, dict]:
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    facts: dict[str, dict] = {}
    for cluster in clusters:
        members = [episode_by_id[item] for item in cluster.episode_ids if item in episode_by_id]
        if not members:
            continue
        representative = max(members, key=lambda item: (item.risk, item.episode_id))
        minimal = shrinks.get(cluster.cluster_id)
        facts[cluster.cluster_id] = measured_facts(
            cluster,
            representative,
            minimal.minimal if minimal is not None else representative.params,
        )
    return facts


def load_domain_bounds(path: Path | None = None) -> dict[str, tuple[float, float]]:
    bounds = scenario_bounds()
    if path is None:
        return bounds
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != set(PARAM_NAMES):
        raise ValueError("domain bounds must define every ScenarioParams field except seed")
    schema_bounds = scenario_bounds()
    validated: dict[str, tuple[float, float]] = {}
    for name, value in raw.items():
        if (
            not isinstance(value, list)
            or len(value) != 2
            or isinstance(value[0], bool)
            or isinstance(value[1], bool)
            or not isinstance(value[0], (int, float))
            or not isinstance(value[1], (int, float))
        ):
            raise ValueError(f"invalid domain bounds for {name}")
        low, high = float(value[0]), float(value[1])
        schema_low, schema_high = schema_bounds[name]
        if (
            not math.isfinite(low)
            or not math.isfinite(high)
            or low >= high
            or low < schema_low
            or high > schema_high
        ):
            raise ValueError(f"domain bounds are outside ScenarioParams limits: {name}")
        validated[name] = (low, high)
    return validated


def _verified_evidence(spec: HardwareSpec) -> tuple[dict[str, Evidence], list[dict[str, object]]]:
    verified = verified_only(spec)
    entries = [
        *verified.joint_torque_limits_nm,
        *verified.joint_velocity_limits,
        *verified.joint_position_limits,
        *verified.error_taxonomy,
    ]
    by_id = {f"V{index}": item for index, item in enumerate(entries, 1)}
    prompt = [
        {
            "id": evidence_id,
            "claim": evidence.claim,
            "value": evidence.value,
            "unit": evidence.unit,
            "quote": evidence.quote,
        }
        for evidence_id, evidence in by_id.items()
    ]
    return by_id, prompt


def _resolve_draft(
    draft: EnvelopeDraft,
    *,
    facts_by_cluster: Mapping[str, Mapping[str, object]],
    evidence_by_id: Mapping[str, object],
) -> tuple[dict[str, float], list, list[str]]:
    values: dict[str, float] = {}
    advisories: list[str] = []
    for constraint in draft.constraints:
        if constraint.parameter not in PARAM_NAMES:
            raise ValueError(f"unsupported constraint parameter: {constraint.parameter}")
        if constraint.basis not in {
            f"representative.params.{constraint.parameter}",
            f"minimal.params.{constraint.parameter}",
        }:
            raise ValueError("constraint basis must identify the same measured parameter")
        facts = facts_by_cluster.get(constraint.cluster_id)
        if facts is None:
            raise ValueError(f"unknown cluster: {constraint.cluster_id}")
        fact_value = _numeric_values(dict(facts)).get(constraint.basis)
        if fact_value is None or not math.isfinite(constraint.value):
            raise ValueError(f"unsupported constraint basis: {constraint.basis}")
        if not math.isclose(constraint.value, fact_value, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"constraint does not match measured fact: {constraint.basis}")
        if constraint.parameter not in {"payload_kg", "latency_ticks", "kp_scale"}:
            advisories.append(
                f"monitor {constraint.parameter} near {fact_value:g}; the shield cannot enforce it"
            )
            continue
        if constraint.bound != "max":
            advisories.append(
                f"monitor minimum {constraint.parameter} of {fact_value:g}; the shield only enforces maximums"
            )
            continue
        old = values.get(constraint.parameter)
        values[constraint.parameter] = min(old, fact_value) if old is not None else fact_value
    if any(evidence_id not in evidence_by_id for evidence_id in draft.evidence_ids):
        raise ValueError("draft referenced unverified or unknown evidence")
    evidence = [evidence_by_id[item] for item in dict.fromkeys(draft.evidence_ids)]
    return values, evidence, list(dict.fromkeys(advisories))


def _fallback_values(
    diagnoses: Mapping[str, DiagnosisReport],
    shrinks: Mapping[str, ShrinkResult],
    facts_by_cluster: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, float], list[str]]:
    candidates: dict[str, list[float]] = {
        name: [] for name in ("payload_kg", "latency_ticks", "kp_scale")
    }
    advisories: list[str] = []
    for cluster_id, report in diagnoses.items():
        facts = facts_by_cluster.get(cluster_id, {})
        numeric = _numeric_values(dict(facts))
        for constraint in report.recommended_constraints:
            fact_value = numeric.get(constraint.basis)
            if fact_value is None or not math.isclose(
                constraint.value, fact_value, rel_tol=1e-6, abs_tol=1e-6
            ):
                continue
            if constraint.parameter not in candidates or constraint.bound != "max":
                advisories.append(
                    f"monitor {constraint.parameter}; the shield cannot enforce this diagnosis constraint"
                )
                continue
            candidates[constraint.parameter].append(fact_value)
    for cluster_id, shrink_result in shrinks.items():
        if cluster_id in diagnoses:
            continue
        params = shrink_result.minimal
        candidates["payload_kg"].append(params.payload_kg)
        candidates["latency_ticks"].append(float(params.latency_ticks))
        candidates["kp_scale"].append(params.kp_scale)
        for field in (
            "link_mass_scale",
            "joint_damping",
            "sensor_noise_rad",
            "torque_limit_scale",
            "push_force_n",
            "push_time_s",
        ):
            advisories.append(
                f"monitor {field}; diagnosis was unavailable and the shield cannot enforce it"
            )
    return (
        {name: min(items) for name, items in candidates.items() if items},
        list(dict.fromkeys(advisories)),
    )


def _diagnosis_advisories(
    diagnoses: Mapping[str, DiagnosisReport],
    facts_by_cluster: Mapping[str, Mapping[str, object]],
) -> list[str]:
    enforceable = {"payload_kg", "latency_ticks", "kp_scale"}
    advisories: list[str] = []
    for cluster_id, report in diagnoses.items():
        facts = _numeric_values(dict(facts_by_cluster.get(cluster_id, {})))
        for constraint in report.recommended_constraints:
            value = facts.get(constraint.basis)
            if value is None or not math.isclose(
                value, constraint.value, rel_tol=1e-6, abs_tol=1e-6
            ):
                continue
            if constraint.parameter not in enforceable or constraint.bound != "max":
                advisories.append(
                    f"monitor {constraint.parameter} near {value:g}; the shield cannot enforce this constraint"
                )
    return list(dict.fromkeys(advisories))


def _inward(value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return low + 0.9 * (min(high, max(low, value)) - low)


def mitigate(
    *,
    robot: str,
    spec: HardwareSpec,
    diagnoses: Mapping[str, DiagnosisReport],
    shrinks: Mapping[str, ShrinkResult],
    facts_by_cluster: Mapping[str, Mapping[str, object]],
    limits: UrdfLimits | None = None,
    domain_bounds: Mapping[str, tuple[float, float]] | None = None,
    domain_path: Path | None = None,
    previous: SafeEnvelope | None = None,
    tuning_feedback: VerificationResult | None = None,
    chat_json: ChatJSON | None = None,
    predictor: TorquePredictor | None = None,
    run_id: str | None = None,
    cfg: DetectorConfig = DetectorConfig(),
) -> SafeEnvelope:
    active_bounds = dict(
        domain_bounds if domain_bounds is not None else load_domain_bounds(domain_path)
    )
    schema_bounds = scenario_bounds()
    if set(active_bounds) != set(PARAM_NAMES):
        raise ValueError("domain bounds must cover every ScenarioParams field except seed")
    for name, (low, high) in active_bounds.items():
        schema_low, schema_high = schema_bounds[name]
        if (
            not math.isfinite(low)
            or not math.isfinite(high)
            or low >= high
            or low < schema_low
            or high > schema_high
        ):
            raise ValueError(f"domain bounds are outside ScenarioParams limits: {name}")
    active_limits = limits or load_urdf_limits()
    evidence_by_id, evidence_prompt = _verified_evidence(spec)
    llm_values: dict[str, float] = {}
    evidence: list = []
    advisories: list[str] = []
    fallback = False
    fallback_reason: str | None = None

    if facts_by_cluster:
        request = chat_json or nebius.chat_json
        system = PROMPT_PATH.read_text(encoding="utf-8")
        user = json.dumps(
            {
                "robot": robot,
                "facts_by_cluster": facts_by_cluster,
                "diagnoses": {
                    key: report.model_dump(mode="json") for key, report in diagnoses.items()
                },
                "verified_evidence": evidence_prompt,
                "previous_envelope": previous.model_dump(mode="json") if previous else None,
                "tuning_feedback": tuning_feedback.to_dict() if tuning_feedback else None,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

        def ask(content: str) -> tuple[dict[str, float], list, list[str]]:
            response, _ = request(
                "deep",
                system,
                content,
                EnvelopeDraft,
                temperature=0,
                max_tokens=4096,
                retries=0,
                run_id=run_id,
            )
            if not isinstance(response, EnvelopeDraft):
                response = EnvelopeDraft.model_validate(response)
            return _resolve_draft(
                response,
                facts_by_cluster=facts_by_cluster,
                evidence_by_id=evidence_by_id,
            )

        try:
            llm_values, evidence, advisories = ask(user)
        except BudgetExceeded as error:
            fallback = True
            fallback_reason = f"mitigation token budget exhausted: {error}"
        except (LLMError, ValueError) as error:
            try:
                llm_values, evidence, advisories = ask(
                    f"{user}\n\nRepair the previous JSON. Validation error: {error}"
                )
            except BudgetExceeded as repair_error:
                fallback = True
                fallback_reason = f"mitigation token budget exhausted: {repair_error}"
            except (LLMError, ValueError) as repair_error:
                fallback = True
                fallback_reason = f"mitigation draft failed: {type(repair_error).__name__}"
    else:
        fallback = True
        fallback_reason = "no diagnosis facts were available"

    if fallback:
        llm_values, fallback_advisories = _fallback_values(diagnoses, shrinks, facts_by_cluster)
        advisories = fallback_advisories
    advisories.extend(_diagnosis_advisories(diagnoses, facts_by_cluster))
    advisories = list(dict.fromkeys(advisories))

    payload_bounds = active_bounds["payload_kg"]
    latency_bounds = active_bounds["latency_ticks"]
    kp_bounds = active_bounds["kp_scale"]
    max_payload = min(payload_bounds[1], llm_values.get("payload_kg", payload_bounds[1]))
    max_latency_ticks = min(latency_bounds[1], llm_values.get("latency_ticks", latency_bounds[1]))
    kp_max = min(kp_bounds[1], llm_values.get("kp_scale", kp_bounds[1]))
    if fallback:
        max_payload = _inward(max_payload, payload_bounds)
        max_latency_ticks = _inward(max_latency_ticks, latency_bounds)
        kp_max = _inward(kp_max, kp_bounds)
    max_payload = max(payload_bounds[0], min(payload_bounds[1], max_payload))
    max_latency_ticks = max(latency_bounds[0], min(latency_bounds[1], max_latency_ticks))
    kp_max = max(kp_bounds[0], min(kp_bounds[1], kp_max))

    if predictor is None:
        with InverseDynamicsPredictor() as active_predictor:
            payload_table = build_payload_speed_table(
                max_payload,
                cfg.tau_ratio_threshold,
                active_limits,
                predictor=active_predictor,
            )
    else:
        payload_table = build_payload_speed_table(
            max_payload,
            cfg.tau_ratio_threshold,
            active_limits,
            predictor=predictor,
        )
    latency_table = build_latency_speed_table()
    return SafeEnvelope(
        robot=robot,
        version="1",
        limits={
            "max_payload_kg": round(max_payload, 6),
            "max_latency_ms": round(max_latency_ticks * TICK_MS, 6),
            "torque_utilization_cap": cfg.tau_ratio_threshold,
            "speed_scale_max_by_payload_kg": payload_table,
            "speed_scale_max_by_latency_ticks": latency_table,
        },
        controller={"kp_scale_max": round(kp_max, 6)},
        domain_randomization_bounds={
            name: [low, high] for name, (low, high) in sorted(active_bounds.items())
        },
        evidence=evidence,
        verification={
            "mitigator_fallback": fallback,
            "fallback_reason": fallback_reason,
            "monitoring_advisories": advisories,
        },
    )
