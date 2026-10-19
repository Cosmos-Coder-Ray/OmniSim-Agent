from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnisim.schemas import StrategyName

QUICK_BUDGET = 10
QUICK_STOP_AFTER_CLUSTERS = 1
QUICK_MAX_DIAGNOSED = 1
DEFAULT_BUDGET = 40


class CampaignStage(StrEnum):
    SEARCH = "search"
    SHRINK = "shrink"
    SPEC = "spec"
    DIAGNOSIS = "diagnosis"
    MITIGATION = "mitigation"
    VERIFICATION = "verification"
    EXPORTS = "exports"
    DONE = "done"


class CampaignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    robot: Literal["franka_panda"] = "franka_panda"
    strategy: StrategyName = "nemotron"
    seed: int = 17
    budget: int | None = Field(default=None, ge=1)
    quick: bool = False
    verify: bool = False

    @model_validator(mode="after")
    def validate_profile(self) -> CampaignRequest:
        if self.quick and self.verify:
            raise ValueError("quick profile cannot request verification")
        if self.quick and self.budget is not None and self.budget > QUICK_BUDGET:
            raise ValueError(f"quick profile budget cannot exceed {QUICK_BUDGET}")
        return self

    @property
    def profile(self) -> Literal["quick", "full", "full+verify"]:
        if self.quick:
            return "quick"
        return "full+verify" if self.verify else "full"

    @property
    def effective_budget(self) -> int:
        return self.budget or (QUICK_BUDGET if self.quick else DEFAULT_BUDGET)

    def planned_stages(self) -> list[CampaignStage]:
        stages = [
            CampaignStage.SEARCH,
            CampaignStage.SHRINK,
            CampaignStage.SPEC,
            CampaignStage.DIAGNOSIS,
        ]
        if self.verify:
            stages.extend(
                (CampaignStage.MITIGATION, CampaignStage.VERIFICATION, CampaignStage.EXPORTS)
            )
        stages.append(CampaignStage.DONE)
        return stages


class CampaignAccepted(BaseModel):
    campaign_id: str
    status: str
    source: Literal["live", "recorded"]
    profile: str
    budget: int
    queue_position: int | None = None
    stages: list[CampaignStage]


class CampaignProgress(BaseModel):
    last_event_id: int = 0
    sims_used: int = 0
    clusters: int = 0


class CampaignState(BaseModel):
    campaign_id: str
    status: str
    stage: CampaignStage | None = None
    stages: list[CampaignStage]
    source: Literal["live", "recorded"]
    config: dict[str, object]
    progress: CampaignProgress
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    queue_position: int | None = None


class CampaignCancellation(BaseModel):
    campaign_id: str
    status: Literal["cancelling"] = "cancelling"


class ApiError(BaseModel):
    error: str
    message: str
    replay: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    demo_mode: bool
    demo_ready: bool
    recorded_runs: list[str]
    queue: dict[str, int]
    token_usage: dict[str, object]
    knowledge_usage: dict[str, int | str]
    model_resolution: dict[str, str | None] | None
