from __future__ import annotations

import json
import os
from datetime import UTC, datetime

from omnisim.api.jobs import JobConfig
from omnisim.api.models import CampaignRequest
from omnisim.core.campaign_verify import run_campaign_verification
from omnisim.core.orchestrator import CampaignOutcome, campaign_events


def write_stage(config: JobConfig, stage: str) -> None:
    run_dir = config.paths.runs / config.campaign_id
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / "stage.json"
    temporary = target.with_name(f".stage.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps({"stage": stage, "updated_at": datetime.now(UTC).isoformat()}) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _write_job_result(config: JobConfig, status: str, error: str | None = None) -> None:
    run_dir = config.paths.runs / config.campaign_id
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / "job_result.json"
    temporary = target.with_name(f".job_result.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps({"status": status, "error": error}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _campaign_outcome(stream) -> CampaignOutcome:
    while True:
        try:
            next(stream)
        except StopIteration as stop:
            return stop.value


def run_campaign_job(config: JobConfig) -> None:
    request = CampaignRequest.model_validate(config.request)
    cancel_path = config.cancel_flag
    try:
        outcome = _campaign_outcome(
            campaign_events(
                request.robot,
                request.strategy,
                seed=request.seed,
                budget=request.effective_budget,
                workers=1,
                stop_after_clusters=1 if request.quick else 3,
                max_diagnosed=1 if request.quick else 2,
                run_id=config.campaign_id,
                runs_root=config.paths.runs,
                on_stage=lambda stage: write_stage(config, stage),
                cancel_requested=cancel_path.exists,
            )
        )
        campaign_status = str(outcome.result.events[-1].data["status"])
        if request.verify and campaign_status != "cancelled":
            run_campaign_verification(
                outcome,
                robot=request.robot,
                seed=request.seed,
                workers=1,
                exports_root=config.paths.exports,
                stage_callback=lambda stage: write_stage(config, stage),
            )
        write_stage(config, "done")
        _write_job_result(config, campaign_status)
    except Exception as error:
        write_stage(config, "done")
        _write_job_result(config, "error", type(error).__name__)
