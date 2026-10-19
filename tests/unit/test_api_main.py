from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from omnisim.api.jobs import JobConfig
from omnisim.api.main import create_app
from omnisim.api.paths import ApiPaths
from omnisim.api.store import ApiStore
from omnisim.api.worker import write_stage
from omnisim.config import Settings


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
    (stale_dir / "stage.json").write_text('{"stage":"diagnosis"}', encoding="utf-8")

    with TestClient(create_app(Settings(), job_runner=FakeRunner(), paths=paths)) as client:
        done = client.get(f"/api/campaigns/{done_id}")
        stale = client.get(f"/api/campaigns/{stale_id}")

    assert done.json()["status"] == "completed"
    assert done.json()["stage"] == "done"
    assert stale.json()["status"] == "interrupted"
    assert stale.json()["stage"] == "diagnosis"


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
