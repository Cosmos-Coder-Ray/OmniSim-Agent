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


class CampaignEvent(BaseModel):
    type: CampaignEventType
    data: dict


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
