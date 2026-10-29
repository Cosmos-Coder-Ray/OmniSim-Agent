from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import re
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import openai
from fastapi import FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from openai import OpenAI
from pydantic import ValidationError
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from omnisim.api.jobs import JobConfig, JobExit, JobRunner, ProcessJobRunner
from omnisim.api.models import (
    ApiError,
    CampaignAccepted,
    CampaignCancellation,
    CampaignProgress,
    CampaignRequest,
    CampaignStage,
    CampaignState,
    ExportsResponse,
    HealthResponse,
)
from omnisim.api.paths import ApiPaths
from omnisim.api.store import ApiStore, JobRecord
from omnisim.api.stream import EventTailer
from omnisim.api.worker import run_campaign_job
from omnisim.audit_log import bind_log_context, configure_logging, reset_log_context
from omnisim.config import Settings
from omnisim.core.orchestrator import _now
from omnisim.knowledge.usage import get_knowledge_usage
from omnisim.llm.budget import LLMUnavailable, get_usage_summary
from omnisim.llm.call_log import provider_host
from omnisim.llm.models import Role, resolve_model
from omnisim.schemas import BenchmarkResult, CampaignEvent, CampaignResult, EpisodeResult

CAMPAIGN_ID = re.compile(r"^[0-9a-f]{32}$")
DEMO_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
KEEPALIVE_S = 10.0
log = logging.getLogger(__name__)
EXPORT_FILES = frozenset(
    {
        "index.html",
        "recorded_benchmark.json",
        "repros.json",
        "ros2_parameters.yaml",
        "safe_envelope.json",
        "test_safe_envelope.py",
        "verification.json",
    }
)
TRAJECTORY_PART = re.compile(r"^[A-Za-z0-9_.-]+$")
TERMINAL_STATUSES = frozenset(
    {"completed", "cancelled", "error", "interrupted", "token_cap", "no_failures_found"}
)


class PayloadTooLarge(Exception):
    pass


class RequestBodyLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    await self._reject(send)
                    return
            except ValueError:
                pass
        size = 0

        async def limited_receive() -> Message:
            nonlocal size
            message = await receive()
            if message["type"] == "http.request":
                size += len(message.get("body", b""))
                if size > self.max_bytes:
                    raise PayloadTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except PayloadTooLarge:
            await self._reject(send)

    async def _reject(self, send: Send) -> None:
        body = (
            b'{"error":"payload_too_large","message":"request body exceeds maximum allowed size"}'
        )
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


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


def _demo_path(paths: ApiPaths, name: str) -> Path | None:
    if not DEMO_NAME.fullmatch(name):
        return None
    root = paths.demo_runs.resolve()
    target = paths.demo_runs / name
    if target.is_symlink() or not target.resolve().is_relative_to(root):
        return None
    return target if (target / "campaign.json").is_file() else None


def _demo_names(paths: ApiPaths) -> list[str]:
    if not paths.demo_runs.is_dir():
        return []
    return sorted(
        path.name
        for path in paths.demo_runs.iterdir()
        if DEMO_NAME.fullmatch(path.name) and path.is_dir() and not path.is_symlink()
    )


def _recorded_campaign(paths: ApiPaths, name: str) -> tuple[Path, CampaignResult] | None:
    directory = _demo_path(paths, name)
    if directory is None:
        return None
    campaign_path = directory / "campaign.json"
    if campaign_path.is_symlink() or not campaign_path.resolve().is_relative_to(
        directory.resolve()
    ):
        return None
    try:
        campaign = CampaignResult.model_validate_json(campaign_path.read_bytes())
    except (OSError, ValidationError):
        return None
    if campaign.source != "recorded":
        return None
    return directory, campaign


def _replay_link(paths: ApiPaths, settings: Settings) -> str | None:
    name = settings.OMNISIM_API_DEMO_RUN
    return f"/api/campaigns/{name}" if _recorded_campaign(paths, name) is not None else None


def _configured_demo_missing(settings: Settings, campaign_id: str) -> bool:
    return settings.OMNISIM_DEMO_MODE and campaign_id == settings.OMNISIM_API_DEMO_RUN


def _demo_unavailable() -> JSONResponse:
    return _error_response(
        404, "demo_unavailable", "the configured recorded campaign is unavailable"
    )


def _events(path: Path) -> list[CampaignEvent]:
    events_path = path / "events.jsonl"
    if events_path.is_symlink():
        return []
    try:
        if not events_path.resolve().is_relative_to(path.resolve()):
            return []
    except OSError:
        return []
    return EventTailer(events_path).read_complete()


def _episodes(path: Path, campaign_id: str) -> list[EpisodeResult] | None:
    episodes_path = path / "episodes.jsonl"
    try:
        if episodes_path.is_symlink() or not episodes_path.resolve().is_relative_to(path.resolve()):
            raise OSError("episode log is outside campaign directory")
        content = episodes_path.read_bytes()
    except FileNotFoundError:
        return []
    except OSError:
        log.error("campaign episodes log is unavailable for %s", campaign_id)
        return None

    complete_lines = content.split(b"\n")
    complete_lines.pop()

    episodes: list[EpisodeResult] = []
    try:
        for line in complete_lines:
            if not line:
                raise ValueError("empty episode record")
            episodes.append(EpisodeResult.model_validate_json(line))
    except (ValidationError, ValueError):
        log.error("campaign episodes log is malformed for %s", campaign_id)
        return None
    return episodes


def _campaign_warnings(
    events: list[CampaignEvent], episodes: list[EpisodeResult], cache_hits: int = 0
) -> list[str]:
    warnings: list[str] = []
    fallback_batches = sum(
        event.type == "hypothesis"
        and any(origin != "llm" for origin in event.data.get("origins", []))
        for event in events
    )
    spec_failures = sum(
        event.type == "spec_ready" and not event.data.get("ok", False) for event in events
    )
    diagnosis_failures = sum(
        event.type == "diagnosis_ready" and not event.data.get("ok", False) for event in events
    )
    episode_failures = sum(bool(episode.failures) for episode in episodes)
    if fallback_batches:
        warnings.append(f"{fallback_batches} Nemotron batches included deterministic proposals.")
    if spec_failures:
        warnings.append(f"{spec_failures} hardware evidence stages were unavailable.")
    if diagnosis_failures:
        warnings.append(f"{diagnosis_failures} diagnoses were unavailable.")
    if episode_failures:
        warnings.append(f"{episode_failures} episodes triggered safety detectors.")
    if any(
        event.type == "campaign_done" and event.data.get("status") == "token_cap"
        for event in events
    ):
        warnings.append("The campaign reached its token budget.")
    if cache_hits:
        warnings.append(f"{cache_hits} manufacturer evidence lookups used the local cache.")
    return warnings


def _run_artifact(path: Path, name: str) -> dict[str, object] | None:
    target = path / name
    if target.is_symlink() or not target.is_file():
        return None
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _run_cache_hits(path: Path, run_id: str) -> int:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-2000:]
    except OSError:
        return 0
    hits = 0
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("run_id") == run_id and row.get("cache") == "hit":
            hits += 1
    return hits


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


def _recorded_state(name: str, directory: Path, campaign: CampaignResult) -> CampaignState:
    verify = (directory / "exports" / name / "verification.json").is_file()
    request = CampaignRequest(
        robot="franka_panda",
        strategy=campaign.strategy,
        seed=campaign.seed,
        budget=campaign.budget,
        verify=verify,
    )
    terminal = next(
        (event for event in reversed(campaign.events) if event.type == "campaign_done"), None
    )
    started = next((event for event in campaign.events if event.type == "campaign_started"), None)
    started_at = (
        str(started.data["started_at"])
        if started is not None and started.data.get("started_at") is not None
        else None
    )
    return CampaignState(
        campaign_id=name,
        status=str(terminal.data["status"]) if terminal is not None else "completed",
        stage=CampaignStage.DONE,
        stages=request.planned_stages(),
        source="recorded",
        config={
            "robot": campaign.robot,
            "strategy": campaign.strategy,
            "seed": campaign.seed,
            "budget": campaign.budget,
            "verify": verify,
        },
        progress=_progress(campaign.events),
        episodes=campaign.episodes,
        created_at=started_at,
        started_at=started_at,
        finished_at=(str(terminal.ts) if terminal is not None else None),
        warnings=_campaign_warnings(campaign.events, campaign.episodes),
    )


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
        result = (
            _read_job_result(run_dir / "job_result.json")
            if stage == CampaignStage.DONE and run_dir is not None
            else None
        )
        status = result["status"] if result is not None else "completed"
        if stage != CampaignStage.DONE:
            status = "interrupted"
        store.set_status(record.campaign_id, status)


def _resolve_models(settings: Settings) -> dict[str, str | None]:
    roles: tuple[Role, ...] = ("fast", "deep", "ultra")
    if not settings.NEBIUS_API_KEY or not settings.NEBIUS_BASE_URL:
        return dict.fromkeys(roles)
    client: OpenAI | None = None
    try:
        client = OpenAI(
            api_key=settings.NEBIUS_API_KEY,
            base_url=settings.NEBIUS_BASE_URL,
            max_retries=0,
            timeout=settings.OMNISIM_API_MODEL_RESOLUTION_TIMEOUT_S,
        )
        return {role: resolve_model(client, role, settings=settings) for role in roles}
    except (openai.OpenAIError, LLMUnavailable, ValueError):
        return dict.fromkeys(roles)
    finally:
        if client is not None:
            client.close()


def _start_model_resolution(app: FastAPI, settings: Settings) -> None:
    app.state.model_resolution = None

    def resolve() -> None:
        app.state.model_resolution = _resolve_models(settings)

    thread = threading.Thread(target=resolve, name="omnisim-model-resolution", daemon=True)
    thread.start()


def _file_response(path: Path, name: str) -> Response:
    is_report = name == "index.html"
    media_type = "text/html" if is_report else mimetypes.guess_type(name)[0]
    headers = {
        "Cache-Control": "private, no-cache",
        "Content-Disposition": f'{"inline" if is_report else "attachment"}; filename="{name}"',
        "X-Content-Type-Options": "nosniff",
    }
    if is_report:
        headers["Content-Security-Policy"] = (
            "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; "
            "form-action 'none'; frame-ancestors 'none'"
        )
    return Response(
        path.read_bytes(),
        media_type=media_type or "application/octet-stream",
        headers=headers,
    )


def _pending_or_missing(record: JobRecord, message: str) -> JSONResponse:
    if record.status in {"queued", "running"}:
        return _error_response(202, "pending", message)
    return _error_response(404, "not_found", message)


def _error_response(
    status: int,
    error: str,
    message: str,
    *,
    replay: str | None = None,
) -> JSONResponse:
    body = ApiError(error=error, message=message, replay=replay)
    return JSONResponse(status_code=status, content=body.model_dump(exclude_none=True))


def create_app(
    settings: Settings | None = None,
    *,
    job_runner: JobRunner | None = None,
    clock: Callable[[], datetime] = _now,
    paths: ApiPaths | None = None,
) -> FastAPI:
    active_settings = settings or Settings.from_env()
    active_paths = paths or ApiPaths.from_root()

    def client_ip(request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded and active_settings.OMNISIM_TRUST_PROXY_HEADERS:
            return forwarded.split(",", 1)[0].strip()
        return get_remote_address(request)

    limiter = Limiter(key_func=client_ip)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        store = ApiStore(active_paths.database)
        store.initialize()
        _reconcile(store, active_paths)
        app.state.recorded_runs = _demo_names(active_paths)
        app.state.demo_ready = (
            _recorded_campaign(active_paths, active_settings.OMNISIM_API_DEMO_RUN) is not None
        )
        if active_settings.OMNISIM_DEMO_MODE and not app.state.demo_ready:
            log.warning(
                "configured demo run is unavailable",
                extra={"demo_run": active_settings.OMNISIM_API_DEMO_RUN},
            )
        if not active_settings.OMNISIM_DEMO_MODE or active_settings.NEBIUS_API_KEY:
            _start_model_resolution(app, active_settings)

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
    app.state.model_resolution = None
    app.state.demo_ready = False
    app.state.recorded_runs = []
    app.state.limiter = limiter
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_bytes=active_settings.OMNISIM_API_MAX_BODY_BYTES,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            origin.strip()
            for origin in active_settings.OMNISIM_API_CORS_ORIGINS.split(",")
            if origin.strip()
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Last-Event-ID"],
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        request_id = request.headers.get("x-request-id")
        if request_id is None or not re.fullmatch(r"[0-9a-f]{32}", request_id):
            request_id = uuid4().hex
        tokens = bind_log_context(request_id=request_id)
        try:
            response = await call_next(request)
            response.headers.setdefault("X-Request-ID", request_id)
            response.headers.setdefault("X-Content-Type-Options", "nosniff")
            response.headers.setdefault("Referrer-Policy", "no-referrer")
            return response
        finally:
            reset_log_context(tokens)

    def rate_limit_handler(request: Request, exc: Exception) -> JSONResponse:
        del exc
        return _error_response(
            429,
            "rate_limited",
            "request rate limit exceeded",
            replay=_replay_link(active_paths, active_settings),
        )

    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

    def validation_handler(request: Request, exc: Exception) -> JSONResponse:
        del request, exc
        return _error_response(422, "invalid_request", "request validation failed")

    app.add_exception_handler(RequestValidationError, validation_handler)

    @app.get(
        "/api/health",
        response_model=HealthResponse,
        operation_id="getHealth",
    )
    @limiter.limit(active_settings.OMNISIM_API_OTHER_RATE_LIMIT)
    def health(request: Request) -> HealthResponse:
        del request
        token_usage = get_usage_summary(
            cap=active_settings.OMNISIM_DAILY_TOKEN_CAP,
            db_path=active_paths.data / "llm_usage.sqlite3",
            now=clock(),
        )
        reasons: list[str] = []
        cap = active_settings.OMNISIM_DAILY_TOKEN_CAP
        used = int(token_usage.get("prompt_tokens", 0)) + int(
            token_usage.get("completion_tokens", 0)
        )
        if cap > 0 and (cap - used) / cap <= 0.1:
            reasons.append("token_headroom_low")
        cutoff = (clock().astimezone(UTC)).timestamp() - 600
        for status, finished_at in app.state.store.recent_outcomes(
            datetime.fromtimestamp(cutoff, UTC).isoformat()
        ):
            if status in {"error", "interrupted"}:
                reasons.append("recent_job_failure")
                break
        for log_path in (
            active_paths.data / "llm_calls.jsonl",
            active_paths.data / "knowledge_calls.jsonl",
        ):
            try:
                rows = log_path.read_text(encoding="utf-8").splitlines()[-200:]
            except OSError:
                continue
            failures = 0
            for line in reversed(rows):
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and row.get("failure_kind") == "transport":
                    failures += 1
                    if failures >= 3:
                        reasons.append("provider_transport_failures")
                        break
                elif isinstance(row, dict) and row.get("ok") is True:
                    break
            if "provider_transport_failures" in reasons:
                break
        return HealthResponse(
            status="degraded" if reasons else "ok",
            reasons=reasons,
            provider_host=provider_host(),
            demo_mode=active_settings.OMNISIM_DEMO_MODE,
            demo_ready=app.state.demo_ready,
            recorded_runs=app.state.recorded_runs,
            queue=app.state.runner.queue_snapshot(),
            token_usage=token_usage,
            knowledge_usage=get_knowledge_usage(
                path=active_paths.data / "knowledge_calls.jsonl",
                now=clock(),
            ),
            model_resolution=app.state.model_resolution,
        )

    @app.get(
        "/api/benchmark",
        response_model=BenchmarkResult,
        operation_id="getBenchmark",
        responses={404: {"model": ApiError}, 503: {"model": ApiError}},
    )
    @limiter.limit(active_settings.OMNISIM_API_OTHER_RATE_LIMIT)
    def benchmark(request: Request) -> BenchmarkResult | Response:
        del request
        path = active_paths.data / "benchmark.json"
        if path.is_symlink() or not path.is_file():
            return _error_response(404, "not_found", "benchmark is unavailable")
        try:
            return BenchmarkResult.model_validate_json(path.read_bytes())
        except ValidationError:
            return _error_response(503, "invalid_benchmark", "saved benchmark is invalid")

    @app.post(
        "/api/campaigns",
        response_model=CampaignAccepted,
        status_code=202,
        operation_id="createCampaign",
        responses={429: {"model": ApiError}, 422: {"model": ApiError}},
    )
    @limiter.limit(active_settings.OMNISIM_API_POST_RATE_LIMIT)
    def create_campaign(request: Request, campaign: CampaignRequest) -> CampaignAccepted | Response:
        del request
        if campaign.effective_budget > active_settings.OMNISIM_API_MAX_BUDGET:
            return _error_response(422, "invalid_budget", "budget exceeds the configured maximum")
        if active_settings.OMNISIM_DEMO_MODE:
            demo = _recorded_campaign(active_paths, active_settings.OMNISIM_API_DEMO_RUN)
            if demo is None:
                return _demo_unavailable()
            directory, recorded = demo
            state = _recorded_state(active_settings.OMNISIM_API_DEMO_RUN, directory, recorded)
            verify = state.config.get("verify") is True
            profile = "quick" if recorded.budget <= 10 else ("full+verify" if verify else "full")
            return CampaignAccepted(
                campaign_id=active_settings.OMNISIM_API_DEMO_RUN,
                status=state.status,
                source="recorded",
                profile=profile,
                budget=recorded.budget,
                stages=state.stages,
            )
        store: ApiStore = app.state.store
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
                replay=_replay_link(active_paths, active_settings),
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
            return _error_response(
                429,
                "queue_full",
                "campaign queue is full",
                replay=_replay_link(active_paths, active_settings),
            )
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
        responses={404: {"model": ApiError}, 500: {"model": ApiError}},
    )
    @limiter.limit(active_settings.OMNISIM_API_OTHER_RATE_LIMIT)
    def campaign_state(campaign_id: str, request: Request) -> CampaignState | Response:
        del request
        if not CAMPAIGN_ID.fullmatch(campaign_id):
            demo = _recorded_campaign(active_paths, campaign_id)
            if demo is not None:
                return _recorded_state(campaign_id, *demo)
            if _configured_demo_missing(active_settings, campaign_id):
                return _demo_unavailable()
            return _error_response(404, "not_found", "campaign was not found")
        run_dir = _run_path(active_paths, campaign_id)
        if run_dir is None:
            return _error_response(404, "not_found", "campaign was not found")
        record = app.state.store.get(campaign_id)
        if record is None:
            return _error_response(404, "not_found", "campaign was not found")
        events = _events(run_dir)
        stage = _stage_file(run_dir / "stage.json")
        episodes = None
        if record.status in TERMINAL_STATUSES:
            episodes = _episodes(run_dir, campaign_id)
            if episodes is None:
                return _error_response(500, "internal_error", "campaign episodes are unavailable")
        warnings = _campaign_warnings(
            events,
            episodes or [],
            _run_cache_hits(active_paths.data / "knowledge_calls.jsonl", campaign_id),
        )
        return CampaignState(
            campaign_id=campaign_id,
            status=record.status,
            stage=stage,
            stages=_planned_stages(record),
            source="live",
            config=record.config,
            progress=_progress(events),
            episodes=episodes,
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
            queue_position=app.state.runner.queue_position(campaign_id),
            warnings=warnings,
        )

    @app.get("/api/campaigns/{campaign_id}/evidence", operation_id="getCampaignEvidence")
    @limiter.limit(active_settings.OMNISIM_API_OTHER_RATE_LIMIT)
    def campaign_evidence(campaign_id: str, request: Request) -> Response:
        del request
        recorded = not CAMPAIGN_ID.fullmatch(campaign_id)
        if recorded:
            demo = _recorded_campaign(active_paths, campaign_id)
            if demo is None:
                return (
                    _demo_unavailable()
                    if _configured_demo_missing(active_settings, campaign_id)
                    else _error_response(404, "not_found", "campaign was not found")
                )
            run_dir = demo[0]
            run_id = (
                demo[1].events[0].data.get("run_id", campaign_id) if demo[1].events else campaign_id
            )
        else:
            run_dir = _run_path(active_paths, campaign_id)
            if run_dir is None or app.state.store.get(campaign_id) is None:
                return _error_response(404, "not_found", "campaign was not found")
            run_id = campaign_id
        artifact = _run_artifact(run_dir, "spec.json")
        if artifact is None:
            return _error_response(404, "not_found", "campaign evidence was not found")
        calls: list[dict[str, object]] = []
        try:
            lines = (
                (active_paths.data / "knowledge_calls.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[-2000:]
            )
        except OSError:
            lines = []
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("run_id") == run_id:
                calls.append(row)
        fetched = [str(row["fetched_at"]) for row in calls if row.get("fetched_at")]
        provenance = {
            "cached": any(row.get("cache") == "hit" for row in calls),
            "fetch_time_start": min(fetched) if fetched else None,
            "fetch_time_end": max(fetched) if fetched else None,
            "live_calls": sum(row.get("cache") == "miss" for row in calls),
            "cache_hits": sum(row.get("cache") == "hit" for row in calls),
        }
        return JSONResponse(
            {
                "campaign_id": campaign_id,
                "source": "recorded" if recorded else "live",
                **artifact,
                "provenance": provenance,
            }
        )

    @app.get("/api/campaigns/{campaign_id}/verification", operation_id="getCampaignVerification")
    @limiter.limit(active_settings.OMNISIM_API_OTHER_RATE_LIMIT)
    def campaign_verification(campaign_id: str, request: Request) -> Response:
        del request
        recorded = not CAMPAIGN_ID.fullmatch(campaign_id)
        if recorded:
            demo = _recorded_campaign(active_paths, campaign_id)
            if demo is None:
                return (
                    _demo_unavailable()
                    if _configured_demo_missing(active_settings, campaign_id)
                    else _error_response(404, "not_found", "campaign was not found")
                )
            run_dir = demo[0]
        else:
            run_dir = _run_path(active_paths, campaign_id)
            if run_dir is None or app.state.store.get(campaign_id) is None:
                return _error_response(404, "not_found", "campaign was not found")
        artifact = _run_artifact(run_dir, "verification.json")
        if artifact is None:
            return _error_response(404, "not_found", "campaign verification was not found")
        return JSONResponse(
            {"campaign_id": campaign_id, "source": "recorded" if recorded else "live", **artifact}
        )

    @app.get(
        "/api/campaigns/{campaign_id}/events",
        operation_id="streamCampaignEvents",
        responses={400: {"model": ApiError}, 404: {"model": ApiError}},
    )
    @limiter.limit(active_settings.OMNISIM_API_OTHER_RATE_LIMIT)
    async def stream_campaign_events(
        campaign_id: str,
        request: Request,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
        after: int | None = Query(default=None, ge=0),
        pace_ms: int = Query(default=100),
    ) -> Response:
        recorded = False
        record: JobRecord | None = None
        if CAMPAIGN_ID.fullmatch(campaign_id):
            run_dir = _run_path(active_paths, campaign_id)
            record = app.state.store.get(campaign_id)
            if run_dir is None or record is None:
                return _error_response(404, "not_found", "campaign was not found")
        else:
            demo = _recorded_campaign(active_paths, campaign_id)
            if demo is None:
                if _configured_demo_missing(active_settings, campaign_id):
                    return _demo_unavailable()
                return _error_response(404, "not_found", "campaign was not found")
            run_dir = demo[0]
            recorded = True
        requested_id = after
        if last_event_id is not None:
            try:
                requested_id = int(last_event_id)
            except ValueError:
                return _error_response(400, "invalid_event_id", "Last-Event-ID must be an integer")
        if requested_id is not None and requested_id < 0:
            return _error_response(400, "invalid_event_id", "event ID must not be negative")
        events_path = run_dir / "events.jsonl"
        if events_path.is_symlink() or not events_path.resolve().is_relative_to(run_dir.resolve()):
            return _error_response(404, "not_found", "campaign events were not found")
        tailer = EventTailer(events_path)
        tailer.resume_after(requested_id or 0)
        last_sent = requested_id or 0

        async def stream() -> AsyncGenerator[str, None]:
            nonlocal last_sent
            last_keepalive = asyncio.get_running_loop().time()
            while True:
                for event in tailer.read_complete():
                    if event.type == "campaign_done" and event.seq <= last_sent:
                        return
                    if event.seq <= last_sent:
                        continue
                    last_sent = event.seq
                    data = json.dumps(event.model_dump(mode="json"), separators=(",", ":"))
                    yield f"id: {event.seq}\nevent: {event.type}\ndata: {data}\n\n"
                    if event.type == "campaign_done":
                        return
                    if recorded and pace_ms > 0:
                        await asyncio.sleep(min(pace_ms, 2000) / 1000)
                current = app.state.store.get(campaign_id) if not recorded else None
                if current is not None and current.status == "interrupted":
                    yield 'event: stream_end\ndata: {"status":"interrupted"}\n\n'
                    return
                if recorded:
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
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-OmniSim-Source": "recorded" if recorded else "live",
            },
        )

    @app.get(
        "/api/campaigns/{campaign_id}/trajectories/{trajectory_path:path}",
        operation_id="getCampaignTrajectory",
        responses={202: {"model": ApiError}, 400: {"model": ApiError}, 404: {"model": ApiError}},
    )
    @limiter.limit(active_settings.OMNISIM_API_OTHER_RATE_LIMIT)
    def campaign_trajectory(campaign_id: str, trajectory_path: str, request: Request) -> Response:
        del request
        recorded = not CAMPAIGN_ID.fullmatch(campaign_id)
        if recorded:
            demo = _recorded_campaign(active_paths, campaign_id)
            if demo is None:
                if _configured_demo_missing(active_settings, campaign_id):
                    return _demo_unavailable()
                return _error_response(404, "not_found", "campaign was not found")
            run_dir = demo[0]
            record = None
        else:
            run_dir = _run_path(active_paths, campaign_id)
            record = app.state.store.get(campaign_id)
            if run_dir is None or record is None:
                return _error_response(404, "not_found", "campaign was not found")
        parts = Path(trajectory_path).parts
        if (
            not parts
            or Path(trajectory_path).is_absolute()
            or any(part in {"", ".", ".."} or not TRAJECTORY_PART.fullmatch(part) for part in parts)
            or not trajectory_path.endswith(".json")
        ):
            return _error_response(400, "invalid_path", "trajectory path is invalid")
        root = run_dir / "trajectories"
        target = root.joinpath(*parts)
        if any(path.is_symlink() for path in (root, *target.parents[:-1], target)):
            return _error_response(404, "not_found", "trajectory was not found")
        try:
            if not target.resolve().is_relative_to(root.resolve()):
                return _error_response(404, "not_found", "trajectory was not found")
        except OSError:
            return _error_response(404, "not_found", "trajectory was not found")
        if not target.is_file():
            if recorded:
                return _error_response(404, "not_found", "trajectory was not found")
            assert record is not None
            return _pending_or_missing(record, "trajectory is not available yet")
        return _file_response(target, target.name)

    @app.get(
        "/api/exports/{campaign_id}",
        response_model=ExportsResponse,
        operation_id="listCampaignExports",
        responses={202: {"model": ApiError}, 404: {"model": ApiError}},
    )
    @limiter.limit(active_settings.OMNISIM_API_OTHER_RATE_LIMIT)
    def campaign_exports(campaign_id: str, request: Request) -> ExportsResponse | Response:
        del request
        recorded = not CAMPAIGN_ID.fullmatch(campaign_id)
        if recorded:
            demo = _recorded_campaign(active_paths, campaign_id)
            if demo is None:
                if _configured_demo_missing(active_settings, campaign_id):
                    return _demo_unavailable()
                return _error_response(404, "not_found", "campaign was not found")
            base = demo[0] / "exports"
            status = _recorded_state(campaign_id, *demo).status
            record = None
        else:
            record = app.state.store.get(campaign_id)
            if record is None:
                return _error_response(404, "not_found", "campaign was not found")
            base = active_paths.exports
            status = record.status
        root = base.parent.resolve()
        export_root = base.resolve()
        directory = base / campaign_id
        if (
            base.is_symlink()
            or not export_root.is_relative_to(root)
            or directory.is_symlink()
            or not directory.resolve().is_relative_to(export_root)
        ):
            return _error_response(404, "not_found", "campaign exports were not found")
        artifacts = (
            sorted(
                path.name
                for path in directory.iterdir()
                if path.is_file() and not path.is_symlink() and path.name in EXPORT_FILES
            )
            if directory.is_dir()
            else []
        )
        if not artifacts:
            if recorded:
                return _error_response(404, "not_found", "campaign exports were not found")
            assert record is not None
            return _pending_or_missing(record, "campaign exports are not available yet")
        return ExportsResponse(campaign_id=campaign_id, status=status, artifacts=artifacts)

    @app.get(
        "/api/exports/{campaign_id}/{artifact_name}",
        operation_id="getCampaignExport",
        responses={202: {"model": ApiError}, 404: {"model": ApiError}},
    )
    @limiter.limit(active_settings.OMNISIM_API_OTHER_RATE_LIMIT)
    def campaign_export(campaign_id: str, artifact_name: str, request: Request) -> Response:
        del request
        if artifact_name not in EXPORT_FILES:
            return _error_response(404, "not_found", "export was not found")
        recorded = not CAMPAIGN_ID.fullmatch(campaign_id)
        if recorded:
            demo = _recorded_campaign(active_paths, campaign_id)
            if demo is None:
                if (
                    active_settings.OMNISIM_DEMO_MODE
                    and campaign_id == active_settings.OMNISIM_API_DEMO_RUN
                ):
                    return _error_response(
                        404,
                        "demo_unavailable",
                        "the configured recorded campaign is unavailable",
                    )
                return _error_response(404, "not_found", "campaign was not found")
            base = demo[0] / "exports"
            record = None
        else:
            record = app.state.store.get(campaign_id)
            if record is None:
                return _error_response(404, "not_found", "campaign was not found")
            base = active_paths.exports
        root = base.parent.resolve()
        export_root = base.resolve()
        directory = base / campaign_id
        path = directory / artifact_name
        if (
            base.is_symlink()
            or not export_root.is_relative_to(root)
            or directory.is_symlink()
            or not directory.resolve().is_relative_to(export_root)
        ):
            return _error_response(404, "not_found", "export was not found")
        if path.is_symlink() or not path.is_file():
            if recorded:
                return _error_response(404, "not_found", "export was not found")
            assert record is not None
            return _pending_or_missing(record, "export is not available yet")
        return _file_response(path, artifact_name)

    @app.delete(
        "/api/campaigns/{campaign_id}",
        response_model=CampaignCancellation,
        status_code=202,
        operation_id="cancelCampaign",
        responses={404: {"model": ApiError}, 409: {"model": ApiError}},
    )
    @limiter.limit(active_settings.OMNISIM_API_OTHER_RATE_LIMIT)
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
