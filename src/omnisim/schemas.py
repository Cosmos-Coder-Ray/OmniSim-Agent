from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

FailureKind = Literal[
    "torque_saturation",
    "joint_limit",
    "velocity_limit",
    "tracking_error",
    "singularity",
    "collision",
    "instability",
]
StrategyName = Literal["nemotron", "random", "cmaes"]
CampaignEventType = Literal[
    "campaign_started",
    "hypothesis",
    "scenario_batch",
    "episode_done",
    "failure_found",
    "cluster_updated",
    "spec_ready",
    "shrink_done",
    "diagnosis_ready",
    "campaign_done",
]
Point3D = tuple[float, float, float]


class ScenarioParams(BaseModel):
    payload_kg: float = Field(default=0.0, ge=0.0, le=3.0)
    link_mass_scale: float = Field(default=1.0, ge=0.7, le=1.5)
    joint_damping: float = Field(default=0.0, ge=0.0, le=5.0)
    latency_ticks: int = Field(default=0, ge=0, le=12)
    sensor_noise_rad: float = Field(default=0.0, ge=0.0, le=0.02)
    torque_limit_scale: float = Field(default=1.0, ge=0.5, le=1.0)
    speed_scale: float = Field(default=1.0, ge=0.5, le=2.0)
    kp_scale: float = Field(default=1.0, ge=0.5, le=2.0)
    push_force_n: float = Field(default=0.0, ge=0.0, le=40.0)
    push_time_s: float = Field(default=2.0, ge=0.0, le=5.0)
    seed: int = 0


class FailureEvent(BaseModel):
    kind: FailureKind
    t: float
    joint: int | None = None
    severity: float
    detail: dict = Field(default_factory=dict)


class EpisodeResult(BaseModel):
    episode_id: str
    params: ScenarioParams
    failures: list[FailureEvent]
    risk: float
    max_tau_ratio: float
    max_vel_ratio: float
    final_ee_error_m: float
    min_manipulability: float
    trajectory_path: str | None = None


class Evidence(BaseModel):
    claim: str
    value: float | str
    unit: str | None = None
    source_url: str
    quote: str
    verified: bool


class HardwareSpec(BaseModel):
    robot: str
    joint_torque_limits_nm: list[Evidence]
    joint_velocity_limits: list[Evidence]
    joint_position_limits: list[Evidence]
    error_taxonomy: list[Evidence]


class Mismatch(BaseModel):
    joint: int = Field(ge=0)
    field: Literal["torque", "velocity", "lower", "upper"]
    urdf: float
    doc: float
    source_url: str
    sim_more_permissive: bool


class ErrorMapping(BaseModel):
    kind: FailureKind
    real_error_name: str | None = None
    evidence: Evidence | None = None
    confidence: Literal["low", "medium", "high"] | None = None


class SafeEnvelope(BaseModel):
    robot: str
    version: str
    limits: dict
    controller: dict
    domain_randomization_bounds: dict
    evidence: list[Evidence]
    verification: dict


class ReplayFrame(BaseModel):
    t: float
    links: list[Point3D] = Field(min_length=8, max_length=8)
    ee: Point3D
    tau_ratio: list[float] = Field(min_length=7, max_length=7)
    vel_ratio: list[float] = Field(min_length=7, max_length=7)
    flags: list[FailureKind]


class Replay(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    fps: int
    frames: list[ReplayFrame]
    events: list[FailureEvent]
    source: Literal["live", "recorded", "mock"]


class CampaignPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CampaignStartedData(CampaignPayload):
    run_id: str
    robot: str
    strategy: StrategyName
    seed: int
    budget: int
    started_at: datetime
    baseline_risk: float | None
    baseline_max_tau_ratio: float | None
    baseline_max_vel_ratio: float | None
    baseline_final_ee_error_m: float | None
    baseline_error: str | None


class HypothesisData(CampaignPayload):
    batch_index: int
    hypothesis: str
    origins: list[Literal["llm", "exploit", "explore", "random", "cma"]]
    params: list[ScenarioParams]


class ScenarioBatchData(CampaignPayload):
    batch_index: int
    episode_ids: list[str]
    sims_used: int


class EpisodeDoneData(CampaignPayload):
    episode_id: str
    ok: bool
    risk: float | None = None
    failure_kinds: list[FailureKind] = Field(default_factory=list)
    duration_s: float | None = None
    ticks: int | None = None
    control_hz: int | None = None
    error: str | None = None


class FailureFoundData(CampaignPayload):
    episode_id: str
    cluster_id: str
    kind: FailureKind
    joint: int | None
    risk: float
    severity: float


class ClusterUpdatedData(CampaignPayload):
    cluster_count: int
    base_key_count: int
    cluster_ids: list[str]


class SpecReadyData(CampaignPayload):
    ok: bool
    robot: str | None = None
    mismatch_count: int = 0
    error: str | None = None


class ShrinkDoneData(CampaignPayload):
    cluster_id: str
    episode_id: str
    sims_used: int
    converged: bool
    trajectory_path: str | None = None


class DiagnosisReadyData(CampaignPayload):
    cluster_id: str
    ok: bool
    confidence: Literal["low", "medium", "high"] | None = None
    error: str | None = None


class CampaignDoneData(CampaignPayload):
    status: Literal["completed", "cancelled", "error", "token_cap", "no_failures_found"]
    stop_reason: Literal["budget", "clusters", "token_cap", "none"]
    sims_used: int
    shrink_sims: int
    clusters: int
    duration_s: float


class NumericClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    basis: str
    value: float


class RecommendedConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameter: str
    bound: Literal["max", "min"]
    value: float
    basis: str


class DiagnosisReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    root_cause: str
    mechanism: str
    numeric_claims: list[NumericClaim]
    real_error_mapping: ErrorMapping | None
    evidence_refs: list[Evidence]
    confidence: Literal["low", "medium", "high"]
    recommended_constraints: list[RecommendedConstraint]
    corroborating_urls: list[str]


_CAMPAIGN_EVENT_DATA: dict[str, type[BaseModel]] = {
    "campaign_started": CampaignStartedData,
    "hypothesis": HypothesisData,
    "scenario_batch": ScenarioBatchData,
    "episode_done": EpisodeDoneData,
    "failure_found": FailureFoundData,
    "cluster_updated": ClusterUpdatedData,
    "spec_ready": SpecReadyData,
    "shrink_done": ShrinkDoneData,
    "diagnosis_ready": DiagnosisReadyData,
    "campaign_done": CampaignDoneData,
}


class CampaignEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seq: int = Field(ge=1)
    ts: datetime
    type: CampaignEventType
    data: dict

    @model_validator(mode="after")
    def validate_event_data(self) -> Self:
        payload = _CAMPAIGN_EVENT_DATA[self.type].model_validate(self.data)
        object.__setattr__(self, "data", payload.model_dump(mode="json"))
        return self


class CampaignResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    robot: str
    budget: int
    strategy: StrategyName
    seed: int
    episodes: list[EpisodeResult]
    events: list[CampaignEvent]
    source: Literal["live", "recorded", "mock"]


class FirstFailureSummary(BaseModel):
    mean: float
    std: float
    per_seed: list[int | None]
    seeds_with_failure: int
    mean_is_lower_bound: bool


class UniqueFailureKindsSummary(BaseModel):
    kinds: list[FailureKind]
    mean: float
    std: float


class MeanStd(BaseModel):
    mean: float
    std: float


class BenchmarkStrategy(BaseModel):
    strategy: StrategyName
    sims_to_first_failure: FirstFailureSummary
    cumulative_failures: list[float]
    unique_failure_kinds: UniqueFailureKindsSummary
    best_risk: MeanStd


class BenchmarkResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    budget: int
    seed_count: int
    strategies: list[BenchmarkStrategy]
    source: Literal["live", "recorded", "mock"]

    @model_validator(mode="after")
    def validate_strategy_metrics(self) -> Self:
        names = [item.strategy for item in self.strategies]
        if len(names) != len(set(names)):
            raise ValueError("strategy names must be unique")

        for item in self.strategies:
            if len(item.cumulative_failures) != self.budget:
                raise ValueError("cumulative_failures length must match budget")
            if len(item.sims_to_first_failure.per_seed) != self.seed_count:
                raise ValueError("per_seed length must match seed_count")
        return self
