from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

from omnisim.api.jobs import JobConfig, JobExit, JobRunner, ProcessJobRunner
from omnisim.api.models import (
    ApiError,
    CampaignAccepted,
    CampaignCancellation,
    CampaignProgress,
    CampaignRequest,
    CampaignStage,
    CampaignState,
)
from omnisim.api.paths import ApiPaths
from omnisim.api.store import ApiStore, JobRecord
from omnisim.api.stream import EventTailer
from omnisim.api.worker import run_campaign_job
from omnisim.config import Settings
from omnisim.core.orchestrator import _now
from omnisim.llm.budget import get_usage_summary
from omnisim.schemas import CampaignEvent

CAMPAIGN_ID = re.compile(r"^[0-9a-f]{32}$")
KEEPALIVE_S = 10.0


def _stage_file(path: Path) -> CampaignStage | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return CampaignStage(value["stage"])
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _run_path(paths: ApiPaths, campaign_id: str) -> Path | None:
    if not CAMPAIGN_ID.fullmatch(campaign_id):
        return None
    root = paths.runs.resolve()
    target = paths.runs / campaign_id
    if target.is_symlink() or not target.resolve().is_relative_to(root):
        return None
    return target


def _events(path: Path) -> list[CampaignEvent]:
    return EventTailer(path / "events.jsonl").read_complete()


def _planned_stages(record: JobRecord | None) -> list[CampaignStage]:
    if record is None:
        return []
    try:
        return CampaignRequest.model_validate(record.config).planned_stages()
    except ValidationError:
        return []


def _progress(events: list[CampaignEvent]) -> CampaignProgress:
    progress = CampaignProgress()
    for event in events:
        progress.last_event_id = event.seq
        if event.type == "scenario_batch":
            progress.sims_used = int(event.data.get("sims_used", progress.sims_used))
        elif event.type == "cluster_updated":
            progress.clusters = int(event.data.get("cluster_count", progress.clusters))
        elif event.type == "campaign_done":
            progress.sims_used = int(event.data.get("sims_used", progress.sims_used))
            progress.clusters = int(event.data.get("clusters", progress.clusters))
    return progress


def _profile_minimum(settings: Settings, campaign: CampaignRequest) -> int:
    return {
        "quick": settings.OMNISIM_API_QUICK_MIN_TOKENS,
        "full": settings.OMNISIM_API_FULL_MIN_TOKENS,
        "full+verify": settings.OMNISIM_API_FULL_VERIFY_MIN_TOKENS,
    }[campaign.profile]


def _profile_timeout(settings: Settings, campaign: CampaignRequest) -> int:
    return {
        "quick": settings.OMNISIM_API_QUICK_TIMEOUT_S,
        "full": settings.OMNISIM_API_FULL_TIMEOUT_S,
        "full+verify": settings.OMNISIM_API_FULL_VERIFY_TIMEOUT_S,
    }[campaign.profile]


def _read_job_result(path: Path) -> dict[str, str] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) and isinstance(value.get("status"), str) else None


def _reconcile(store: ApiStore, paths: ApiPaths) -> None:
    for record in store.unfinished():
        run_dir = _run_path(paths, record.campaign_id)
        stage = _stage_file(run_dir / "stage.json") if run_dir is not None else None
        store.set_status(
            record.campaign_id, "completed" if stage == CampaignStage.DONE else "interrupted"
        )


def _error_response(status: int, error: str, message: str) -> JSONResponse:
    body = ApiError(error=error, message=message)
    return JSONResponse(status_code=status, content=body.model_dump())


def create_app(
    settings: Settings | None = None,
    *,
    job_runner: JobRunner | None = None,
    clock: Callable[[], datetime] = _now,
    paths: ApiPaths | None = None,
) -> FastAPI:
    active_settings = settings or Settings.from_env()
    active_paths = paths or ApiPaths.from_root()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store = ApiStore(active_paths.database)
        store.initialize()
        _reconcile(store, active_paths)

        def job_started(config: JobConfig) -> None:
            store.set_status(config.campaign_id, "running", at=clock().astimezone(UTC).isoformat())

        def job_finished(config: JobConfig, result: JobExit) -> None:
            status = result.status
            saved = _read_job_result(active_paths.runs / config.campaign_id / "job_result.json")
            if result.exitcode == 0 and result.status == "completed" and saved is not None:
                status = saved["status"]
            store.set_status(config.campaign_id, status, at=clock().astimezone(UTC).isoformat())

        runner = (
            job_runner
            if job_runner is not None
            else ProcessJobRunner(
                run_campaign_job,
                max_concurrent=active_settings.OMNISIM_API_MAX_CONCURRENT_CAMPAIGNS,
                max_queue=active_settings.OMNISIM_API_MAX_QUEUE,
                on_started=job_started,
                on_finished=job_finished,
            )
        )
        app.state.store = store
        app.state.runner = runner
        runner.start()
        try:
            yield
        finally:
            runner.shutdown()

    app = FastAPI(title="OmniSim-Agent API", version="1.0", lifespan=lifespan)
    app.state.settings = active_settings
    app.state.paths = active_paths
    app.state.clock = clock

    @app.post(
        "/api/campaigns",
        response_model=CampaignAccepted,
        status_code=202,
        operation_id="createCampaign",
        responses={429: {"model": ApiError}, 422: {"model": ApiError}},
    )
    def create_campaign(request: Request, campaign: CampaignRequest) -> CampaignAccepted | Response:
        del request
        store: ApiStore = app.state.store
        if campaign.effective_budget > active_settings.OMNISIM_API_MAX_BUDGET:
            return _error_response(422, "invalid_budget", "budget exceeds the configured maximum")
        usage = get_usage_summary(
            cap=active_settings.OMNISIM_DAILY_TOKEN_CAP,
            db_path=active_paths.data / "llm_usage.sqlite3",
            now=clock(),
        )
        if int(usage["remaining"]) < _profile_minimum(active_settings, campaign):
            return _error_response(
                429,
                "token_budget_unavailable",
                "remaining daily token budget is below the profile admission estimate",
            )

        campaign_id = uuid4().hex
        effective_request = campaign.model_dump(exclude_none=True)
        effective_request["budget"] = campaign.effective_budget
        record = store.create(campaign_id, effective_request)
        config = JobConfig(
            campaign_id=campaign_id,
            request=effective_request,
            paths=active_paths,
            cancel_flag=active_paths.control / f"{campaign_id}.cancel",
            timeout_s=_profile_timeout(active_settings, campaign),
        )
        accepted, queue_position = app.state.runner.submit(config)
        if not accepted:
            store.delete(campaign_id)
            return _error_response(429, "queue_full", "campaign queue is full")
        return CampaignAccepted(
            campaign_id=record.campaign_id,
            status="queued",
            source="live",
            profile=campaign.profile,
            budget=campaign.effective_budget,
            queue_position=queue_position,
            stages=campaign.planned_stages(),
        )

    @app.get(
        "/api/campaigns/{campaign_id}",
        response_model=CampaignState,
        operation_id="getCampaignState",
        responses={404: {"model": ApiError}},
    )
    def campaign_state(campaign_id: str, request: Request) -> CampaignState | Response:
        del request
        run_dir = _run_path(active_paths, campaign_id)
        if run_dir is None:
            return _error_response(404, "not_found", "campaign was not found")
        record = app.state.store.get(campaign_id)
        if record is None:
            return _error_response(404, "not_found", "campaign was not found")
        events = _events(run_dir)
        stage = _stage_file(run_dir / "stage.json")
        return CampaignState(
            campaign_id=campaign_id,
            status=record.status,
            stage=stage,
            stages=_planned_stages(record),
            source="live",
            config=record.config,
            progress=_progress(events),
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
            queue_position=app.state.runner.queue_position(campaign_id),
        )

    @app.get(
        "/api/campaigns/{campaign_id}/events",
        operation_id="streamCampaignEvents",
        responses={400: {"model": ApiError}, 404: {"model": ApiError}},
    )
    async def stream_campaign_events(
        campaign_id: str,
        request: Request,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
        after: int | None = Query(default=None, ge=0),
    ) -> Response:
        run_dir = _run_path(active_paths, campaign_id)
        if run_dir is None or app.state.store.get(campaign_id) is None:
            return _error_response(404, "not_found", "campaign was not found")
        requested_id = after
        if last_event_id is not None:
            try:
                requested_id = int(last_event_id)
            except ValueError:
                return _error_response(400, "invalid_event_id", "Last-Event-ID must be an integer")
        if requested_id is not None and requested_id < 0:
            return _error_response(400, "invalid_event_id", "event ID must not be negative")
        tailer = EventTailer(run_dir / "events.jsonl")
        tailer.resume_after(requested_id or 0)
        last_sent = requested_id or 0

        async def stream() -> AsyncGenerator[str, None]:
            nonlocal last_sent
            last_keepalive = asyncio.get_running_loop().time()
            while True:
                for event in tailer.read_complete():
                    if event.seq <= last_sent:
                        continue
                    last_sent = event.seq
                    data = json.dumps(event.model_dump(mode="json"), separators=(",", ":"))
                    yield f"id: {event.seq}\nevent: {event.type}\ndata: {data}\n\n"
                    if event.type == "campaign_done":
                        return
                current = app.state.store.get(campaign_id)
                if current is not None and current.status == "interrupted":
                    yield 'event: stream_end\ndata: {"status":"interrupted"}\n\n'
                    return
                now = asyncio.get_running_loop().time()
                if now - last_keepalive >= KEEPALIVE_S:
                    yield ": keep-alive\n\n"
                    last_keepalive = now
                if await request.is_disconnected():
                    return
                await asyncio.sleep(0.1)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.delete(
        "/api/campaigns/{campaign_id}",
        response_model=CampaignCancellation,
        status_code=202,
        operation_id="cancelCampaign",
        responses={404: {"model": ApiError}, 409: {"model": ApiError}},
    )
    def cancel_campaign(campaign_id: str, request: Request) -> CampaignCancellation | Response:
        del request
        if not CAMPAIGN_ID.fullmatch(campaign_id):
            return _error_response(404, "not_found", "campaign was not found")
        record = app.state.store.get(campaign_id)
        if record is None:
            return _error_response(404, "not_found", "campaign was not found")
        if record.status not in {"queued", "running"}:
            return _error_response(409, "campaign_not_active", "campaign is not active")
        if not app.state.runner.cancel(campaign_id):
            return _error_response(409, "campaign_not_active", "campaign is not active")
        return CampaignCancellation(campaign_id=campaign_id)

    return app


app = create_app()
