from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from omnisim.api.jobs import JobConfig
from omnisim.api.main import create_app
from omnisim.api.paths import ApiPaths
from omnisim.api.store import ApiStore
from omnisim.api.worker import write_stage
from omnisim.config import Settings
from omnisim.schemas import CampaignEvent, CampaignResult, EpisodeResult, ScenarioParams


class FakeRunner:
    def __init__(self) -> None:
        self.configs: list[JobConfig] = []
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def submit(self, config: JobConfig) -> tuple[bool, int | None]:
        self.configs.append(config)
        return True, None

    def queue_position(self, campaign_id: str) -> int | None:
        return None

    def queue_snapshot(self) -> dict[str, int]:
        return {"queued": 0, "running": 0, "max_concurrent": 1, "max_queue": 3}

    def cancel(self, campaign_id: str) -> bool:
        return any(config.campaign_id == campaign_id for config in self.configs)

    def shutdown(self, grace_s: float = 5.0) -> None:
        del grace_s
        self.closed = True


def app_for(
    root: Path,
    *,
    runner: FakeRunner | None = None,
    settings: Settings | None = None,
):
    active_settings = settings or Settings(OMNISIM_API_POST_RATE_LIMIT="100/hour")
    return create_app(
        active_settings,
        job_runner=runner or FakeRunner(),
        paths=ApiPaths.from_root(root),
    )


def test_app_construction_does_not_create_api_files(tmp_path: Path) -> None:
    root = tmp_path / "not-created"
    app_for(root)

    assert not root.exists()


def test_stage_writer_replaces_stage_file_atomically(tmp_path: Path) -> None:
    paths = ApiPaths.from_root(tmp_path)
    config = JobConfig(
        campaign_id="c" * 32,
        request={},
        paths=paths,
        cancel_flag=paths.control / "cancel",
        timeout_s=180,
    )

    write_stage(config, "search")
    write_stage(config, "done")

    assert (
        json.loads((paths.runs / config.campaign_id / "stage.json").read_text())["stage"] == "done"
    )
    assert list((paths.runs / config.campaign_id).glob(".stage.*.tmp")) == []


def test_campaign_post_sets_profile_defaults_and_ordered_stages(tmp_path: Path) -> None:
    runner = FakeRunner()
    with TestClient(app_for(tmp_path, runner=runner)) as client:
        response = client.post("/api/campaigns", json={"quick": True})

    assert response.status_code == 202
    accepted = response.json()
    assert accepted["profile"] == "quick"
    assert accepted["budget"] == 10
    assert accepted["stages"] == ["search", "shrink", "spec", "diagnosis", "done"]
    assert runner.configs[0].timeout_s == 180
    assert runner.configs[0].request["budget"] == 10


def test_quick_campaign_cannot_request_verification(tmp_path: Path) -> None:
    with TestClient(app_for(tmp_path)) as client:
        response = client.post("/api/campaigns", json={"quick": True, "verify": True})

    assert response.status_code == 422


def test_campaign_cancel_requests_cooperative_stop(tmp_path: Path) -> None:
    runner = FakeRunner()
    with TestClient(app_for(tmp_path, runner=runner)) as client:
        accepted = client.post("/api/campaigns", json={})
        response = client.delete(f"/api/campaigns/{accepted.json()['campaign_id']}")

    assert response.status_code == 202
    assert response.json()["status"] == "cancelling"


def test_startup_reconciles_terminal_stage_and_interrupted_job(tmp_path: Path) -> None:
    paths = ApiPaths.from_root(tmp_path)
    store = ApiStore(paths.database)
    store.initialize()
    done_id = "a" * 32
    stale_id = "b" * 32
    config = {"robot": "franka_panda", "strategy": "random", "seed": 17, "budget": 10}
    store.create(done_id, config)
    store.create(stale_id, config)
    done_dir = paths.runs / done_id
    stale_dir = paths.runs / stale_id
    done_dir.mkdir(parents=True)
    stale_dir.mkdir(parents=True)
    (done_dir / "stage.json").write_text('{"stage":"done"}', encoding="utf-8")
    (done_dir / "job_result.json").write_text('{"status":"token_cap"}', encoding="utf-8")
    (stale_dir / "stage.json").write_text('{"stage":"diagnosis"}', encoding="utf-8")

    with TestClient(create_app(Settings(), job_runner=FakeRunner(), paths=paths)) as client:
        done = client.get(f"/api/campaigns/{done_id}")
        stale = client.get(f"/api/campaigns/{stale_id}")

    assert done.json()["status"] == "token_cap"
    assert done.json()["stage"] == "done"
    assert stale.json()["status"] == "interrupted"
    assert stale.json()["stage"] == "diagnosis"


def test_event_stream_ends_for_interrupted_job(tmp_path: Path) -> None:
    runner = FakeRunner()
    app = app_for(tmp_path, runner=runner)
    with TestClient(app) as client:
        accepted = client.post("/api/campaigns", json={"quick": True})
        campaign_id = accepted.json()["campaign_id"]
        app.state.store.set_status(campaign_id, "interrupted")

        response = client.get(f"/api/campaigns/{campaign_id}/events")

    assert response.status_code == 200
    assert 'event: stream_end\ndata: {"status":"interrupted"}' in response.text


def test_health_returns_cached_state_without_waiting_for_model_resolution(
    tmp_path: Path, monkeypatch
) -> None:
    import threading

    import omnisim.api.main as api_main

    entered = threading.Event()
    release = threading.Event()

    def resolve_model(*args, **kwargs):
        del args, kwargs
        entered.set()
        release.wait(timeout=2)
        return "nvidia/nemotron-test"

    monkeypatch.setattr(api_main, "resolve_model", resolve_model)
    settings = Settings(NEBIUS_API_KEY="key", NEBIUS_BASE_URL="https://example.invalid/v1/")
    try:
        with TestClient(app_for(tmp_path, settings=settings)) as client:
            assert entered.wait(timeout=1)
            response = client.get("/api/health")
            assert response.status_code == 200
            assert response.json()["model_resolution"] is None
    finally:
        release.set()


def test_benchmark_returns_recorded_summary(tmp_path: Path) -> None:
    paths = ApiPaths.from_root(tmp_path)
    paths.data.mkdir(parents=True)
    benchmark = {
        "budget": 1,
        "seed_count": 1,
        "strategies": [],
        "source": "recorded",
    }
    (paths.data / "benchmark.json").write_text(json.dumps(benchmark), encoding="utf-8")

    with TestClient(create_app(Settings(), job_runner=FakeRunner(), paths=paths)) as client:
        response = client.get("/api/benchmark")

    assert response.status_code == 200
    assert response.json()["source"] == "recorded"


def test_export_list_and_report_headers(tmp_path: Path) -> None:
    paths = ApiPaths.from_root(tmp_path)
    store = ApiStore(paths.database)
    store.initialize()
    campaign_id = "d" * 32
    store.create(campaign_id, {}, status="completed")
    export_dir = paths.exports / campaign_id
    export_dir.mkdir(parents=True)
    (export_dir / "index.html").write_text("<h1>report</h1>", encoding="utf-8")
    (export_dir / "private.txt").write_text("not served", encoding="utf-8")

    with TestClient(create_app(Settings(), job_runner=FakeRunner(), paths=paths)) as client:
        listing = client.get(f"/api/exports/{campaign_id}")
        report = client.get(f"/api/exports/{campaign_id}/index.html")
        unknown = client.get(f"/api/exports/{campaign_id}/private.txt")

    assert listing.status_code == 200
    assert listing.json()["artifacts"] == ["index.html"]
    assert report.status_code == 200
    assert "frame-ancestors 'none'" in report.headers["content-security-policy"]
    assert report.headers["x-content-type-options"] == "nosniff"
    assert report.headers["content-disposition"].startswith("inline;")
    assert unknown.status_code == 404


def test_campaign_trajectory_is_confined_to_run_trajectory_directory(tmp_path: Path) -> None:
    paths = ApiPaths.from_root(tmp_path)
    store = ApiStore(paths.database)
    store.initialize()
    campaign_id = "e" * 32
    store.create(campaign_id, {}, status="completed")
    trajectory_dir = paths.runs / campaign_id / "trajectories"
    trajectory_dir.mkdir(parents=True)
    (trajectory_dir / "episode.json").write_text('{"steps": []}', encoding="utf-8")

    with TestClient(create_app(Settings(), job_runner=FakeRunner(), paths=paths)) as client:
        trajectory = client.get(f"/api/campaigns/{campaign_id}/trajectories/episode.json")
        traversal = client.get(f"/api/campaigns/{campaign_id}/trajectories/..%2Fstage.json")

    assert trajectory.status_code == 200
    assert trajectory.json() == {"steps": []}
    assert traversal.status_code in {400, 404}


def _episode(episode_id: str, *, risk: float = 0.1) -> EpisodeResult:
    return EpisodeResult(
        episode_id=episode_id,
        params=ScenarioParams(seed=17),
        failures=[],
        risk=risk,
        max_tau_ratio=0.2,
        max_vel_ratio=0.3,
        final_ee_error_m=0.01,
        min_manipulability=0.5,
    )


def test_active_campaign_episodes_are_unavailable_until_terminal(tmp_path: Path) -> None:
    paths = ApiPaths.from_root(tmp_path)
    with TestClient(create_app(Settings(), job_runner=FakeRunner(), paths=paths)) as client:
        campaign_id = client.post("/api/campaigns", json={"quick": True}).json()["campaign_id"]
        for status in ("queued", "running"):
            ApiStore(paths.database).set_status(campaign_id, status)
            response = client.get(f"/api/campaigns/{campaign_id}")
            assert response.status_code == 200
            assert response.json()["status"] == status
            assert response.json()["episodes"] is None


@pytest.mark.parametrize(
    "status",
    ["completed", "cancelled", "error", "interrupted", "token_cap", "no_failures_found"],
)
def test_terminal_campaign_statuses_return_episode_list(tmp_path: Path, status: str) -> None:
    paths = ApiPaths.from_root(tmp_path)
    with TestClient(create_app(Settings(), job_runner=FakeRunner(), paths=paths)) as client:
        campaign_id = client.post("/api/campaigns", json={"quick": True}).json()["campaign_id"]
        run_dir = paths.runs / campaign_id
        run_dir.mkdir(parents=True)
        (run_dir / "episodes.jsonl").write_text(
            _episode("done").model_dump_json() + "\n", encoding="utf-8"
        )
        ApiStore(paths.database).set_status(campaign_id, status)

        response = client.get(f"/api/campaigns/{campaign_id}")

    assert response.status_code == 200
    assert [episode["episode_id"] for episode in response.json()["episodes"]] == ["done"]


def test_terminal_campaign_returns_ordered_complete_records_and_ignores_partial_line(
    tmp_path: Path,
) -> None:
    paths = ApiPaths.from_root(tmp_path)
    with TestClient(create_app(Settings(), job_runner=FakeRunner(), paths=paths)) as client:
        campaign_id = client.post("/api/campaigns", json={"quick": True}).json()["campaign_id"]
        run_dir = paths.runs / campaign_id
        run_dir.mkdir(parents=True)
        first = _episode("first")
        second = _episode("second", risk=0.2)
        content = (
            first.model_dump_json()
            + "\n"
            + second.model_dump_json()
            + "\n"
            + '{"episode_id":"partial'
        )
        (run_dir / "episodes.jsonl").write_text(content, encoding="utf-8")
        ApiStore(paths.database).set_status(campaign_id, "completed")

        response = client.get(f"/api/campaigns/{campaign_id}")

    assert response.status_code == 200
    assert [episode["episode_id"] for episode in response.json()["episodes"]] == ["first", "second"]
    assert [episode["risk"] for episode in response.json()["episodes"]] == [0.1, 0.2]


def test_terminal_campaign_without_episode_records_returns_empty_list(tmp_path: Path) -> None:
    paths = ApiPaths.from_root(tmp_path)
    with TestClient(create_app(Settings(), job_runner=FakeRunner(), paths=paths)) as client:
        campaign_id = client.post("/api/campaigns", json={"quick": True}).json()["campaign_id"]
        run_dir = paths.runs / campaign_id
        run_dir.mkdir(parents=True)
        (run_dir / "episodes.jsonl").write_bytes(b"")
        ApiStore(paths.database).set_status(campaign_id, "no_failures_found")

        response = client.get(f"/api/campaigns/{campaign_id}")

    assert response.status_code == 200
    assert response.json()["episodes"] == []


def test_malformed_complete_episode_line_returns_generic_error_without_partial_data(
    tmp_path: Path, caplog
) -> None:
    paths = ApiPaths.from_root(tmp_path)
    with TestClient(create_app(Settings(), job_runner=FakeRunner(), paths=paths)) as client:
        campaign_id = client.post("/api/campaigns", json={"quick": True}).json()["campaign_id"]
        run_dir = paths.runs / campaign_id
        run_dir.mkdir(parents=True)
        safe = _episode("safe").model_dump_json()
        sensitive = '{"private_payload":"episode contents must not escape"}'
        (run_dir / "episodes.jsonl").write_text(f"{safe}\n{sensitive}\n", encoding="utf-8")
        ApiStore(paths.database).set_status(campaign_id, "error")

        response = client.get(f"/api/campaigns/{campaign_id}")

    assert response.status_code == 500
    assert response.json() == {
        "error": "internal_error",
        "message": "campaign episodes are unavailable",
    }
    assert "episode contents must not escape" not in response.text
    assert str(paths.runs) not in response.text
    assert "campaign episodes log is malformed" in caplog.text
    assert "episode contents must not escape" not in caplog.text


def test_unknown_campaign_state_returns_not_found(tmp_path: Path) -> None:
    with TestClient(app_for(tmp_path)) as client:
        response = client.get("/api/campaigns/unknown")

    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def _write_demo(paths: ApiPaths, name: str = "hero") -> None:
    directory = paths.demo_runs / name
    directory.mkdir(parents=True)
    event = CampaignEvent(
        seq=1,
        ts=datetime(2026, 10, 19, tzinfo=UTC),
        type="campaign_done",
        data={
            "status": "completed",
            "stop_reason": "budget",
            "sims_used": 8,
            "shrink_sims": 0,
            "clusters": 0,
            "duration_s": 1.0,
        },
    )
    campaign = CampaignResult(
        robot="franka_panda",
        budget=8,
        strategy="random",
        seed=17,
        episodes=[_episode("recorded_first"), _episode("recorded_second", risk=0.2)],
        events=[event],
        source="recorded",
    )
    (directory / "campaign.json").write_text(campaign.model_dump_json(), encoding="utf-8")
    (directory / "events.jsonl").write_text(event.model_dump_json() + "\n", encoding="utf-8")
    trajectories = directory / "trajectories"
    trajectories.mkdir()
    (trajectories / "nominal.json").write_text('{"episode_id":"nominal"}', encoding="utf-8")
    (trajectories / "shrunk_tracking_error_global.json").write_text(
        '{"episode_id":"shrunk"}', encoding="utf-8"
    )
    exports = directory / "exports" / name
    exports.mkdir(parents=True)
    (exports / "index.html").write_text("<h1>recorded</h1>", encoding="utf-8")


def test_demo_mode_serves_only_installed_recording_and_skips_live_services(
    tmp_path: Path, monkeypatch
) -> None:
    import omnisim.api.main as api_main

    def fail_resolution(*args, **kwargs):
        raise AssertionError("demo mode must not resolve live models")

    monkeypatch.setattr(api_main, "resolve_model", fail_resolution)
    paths = ApiPaths.from_root(tmp_path)
    _write_demo(paths)
    runner = FakeRunner()
    settings = Settings(
        OMNISIM_DEMO_MODE=True,
        OMNISIM_API_DEMO_RUN="hero",
        NEBIUS_API_KEY="configured-but-not-used",
        NEBIUS_BASE_URL="https://example.invalid/v1/",
    )

    with TestClient(create_app(settings, job_runner=runner, paths=paths)) as client:
        health = client.get("/api/health")
        accepted = client.post("/api/campaigns", json={"quick": True})
        state = client.get("/api/campaigns/hero")
        stream = client.get("/api/campaigns/hero/events?pace_ms=2001")
        report = client.get("/api/exports/hero/index.html")

    assert health.json()["demo_ready"] is True
    assert health.json()["recorded_runs"] == ["hero"]
    assert health.json()["model_resolution"] is None
    assert accepted.json()["source"] == "recorded"
    assert accepted.json()["campaign_id"] == "hero"
    assert runner.configs == []
    assert state.json()["source"] == "recorded"
    assert state.json()["stage"] == "done"
    assert [episode["episode_id"] for episode in state.json()["episodes"]] == [
        "recorded_first",
        "recorded_second",
    ]
    assert stream.headers["x-omnisim-source"] == "recorded"
    assert "event: campaign_done" in stream.text
    assert report.status_code == 200


def test_recorded_campaign_serves_nominal_and_shrunk_trajectories(tmp_path: Path) -> None:
    paths = ApiPaths.from_root(tmp_path)
    _write_demo(paths)

    with TestClient(create_app(Settings(), job_runner=FakeRunner(), paths=paths)) as client:
        nominal = client.get("/api/campaigns/hero/trajectories/nominal.json")
        shrunk = client.get("/api/campaigns/hero/trajectories/shrunk_tracking_error_global.json")
        unknown_trajectory = client.get("/api/campaigns/hero/trajectories/missing.json")
        unknown_campaign = client.get("/api/campaigns/missing/trajectories/nominal.json")

    assert nominal.status_code == 200
    assert nominal.json() == {"episode_id": "nominal"}
    assert shrunk.status_code == 200
    assert shrunk.json() == {"episode_id": "shrunk"}
    assert unknown_trajectory.status_code == 404
    assert unknown_campaign.status_code == 404


def test_missing_configured_demo_is_unavailable_and_warns_once(tmp_path: Path, caplog) -> None:
    runner = FakeRunner()
    settings = Settings(OMNISIM_DEMO_MODE=True, OMNISIM_API_DEMO_RUN="missing")

    with TestClient(app_for(tmp_path, runner=runner, settings=settings)) as client:
        health = client.get("/api/health")
        response = client.post("/api/campaigns", json={"quick": True})

    warnings = [
        record
        for record in caplog.records
        if record.message == "configured demo run is unavailable"
    ]
    assert len(warnings) == 1
    assert health.json()["demo_ready"] is False
    assert response.status_code == 404
    assert response.json()["error"] == "demo_unavailable"
    assert runner.configs == []


def test_rate_limit_omits_replay_link_when_no_recording_exists(tmp_path: Path) -> None:
    settings = Settings(OMNISIM_API_POST_RATE_LIMIT="1/minute")

    with TestClient(app_for(tmp_path, settings=settings)) as client:
        first = client.post("/api/campaigns", json={"quick": True})
        limited = client.post("/api/campaigns", json={"quick": True})

    assert first.status_code == 202
    assert limited.status_code == 429
    assert "replay" not in limited.json()


def test_profile_token_admission_uses_profile_minimum(tmp_path: Path) -> None:
    settings = Settings(
        OMNISIM_DAILY_TOKEN_CAP=50000,
        OMNISIM_API_QUICK_MIN_TOKENS=10000,
        OMNISIM_API_FULL_MIN_TOKENS=60000,
        OMNISIM_API_FULL_VERIFY_MIN_TOKENS=70000,
    )
    runner = FakeRunner()

    with TestClient(app_for(tmp_path, runner=runner, settings=settings)) as client:
        quick = client.post("/api/campaigns", json={"quick": True})
        full = client.post("/api/campaigns", json={})

    assert quick.status_code == 202
    assert full.status_code == 429
    assert full.json()["error"] == "token_budget_unavailable"


def test_rate_limit_includes_replay_link_when_recording_is_installed(tmp_path: Path) -> None:
    paths = ApiPaths.from_root(tmp_path)
    _write_demo(paths)
    settings = Settings(OMNISIM_API_POST_RATE_LIMIT="1/minute")

    with TestClient(create_app(settings, job_runner=FakeRunner(), paths=paths)) as client:
        first = client.post("/api/campaigns", json={"quick": True})
        limited = client.post("/api/campaigns", json={"quick": True})

    assert first.status_code == 202
    assert limited.status_code == 429
    assert limited.json()["replay"] == "/api/campaigns/hero"


def test_missing_export_is_pending_only_while_campaign_is_active(tmp_path: Path) -> None:
    paths = ApiPaths.from_root(tmp_path)
    with TestClient(create_app(Settings(), job_runner=FakeRunner(), paths=paths)) as client:
        accepted = client.post("/api/campaigns", json={"quick": True})
        campaign_id = accepted.json()["campaign_id"]
        pending = client.get(f"/api/exports/{campaign_id}/safe_envelope.json")
        ApiStore(paths.database).set_status(campaign_id, "completed")
        missing = client.get(f"/api/exports/{campaign_id}/safe_envelope.json")

    assert accepted.status_code == 202
    assert pending.status_code == 202
    assert missing.status_code == 404


def test_cors_and_security_headers_are_configurable(tmp_path: Path) -> None:
    settings = Settings(OMNISIM_API_CORS_ORIGINS="https://dashboard.example")

    with TestClient(app_for(tmp_path, settings=settings)) as client:
        response = client.get("/api/health", headers={"Origin": "https://dashboard.example"})

    assert response.headers["access-control-allow-origin"] == "https://dashboard.example"
    assert response.headers["x-content-type-options"] == "nosniff"
