from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import httpx
import uvicorn

from omnisim.api.jobs import JobConfig, ProcessJobRunner
from omnisim.api.main import create_app
from omnisim.api.paths import ApiPaths
from omnisim.config import Settings


def _test_campaign_job(config: JobConfig) -> None:
    run_dir = config.paths.runs / config.campaign_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.json").write_text('{"spec":{"robot":"Panda"}}', encoding="utf-8")
    (run_dir / "verification.json").write_text('{"confirmation":{"passed":true}}', encoding="utf-8")
    export_dir = config.paths.exports / config.campaign_id
    export_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "safe_envelope.json",
        "ros2_parameters.yaml",
        "test_safe_envelope.py",
        "index.html",
    ):
        (export_dir / name).write_text("safe artifact", encoding="utf-8")


def test_api_serves_health_and_run_artifacts_over_http(tmp_path: Path) -> None:
    paths = ApiPaths.from_root(tmp_path)
    paths.data.mkdir(parents=True, exist_ok=True)
    (paths.data / "llm_calls.jsonl").write_text(
        "".join(
            '{"failure_kind":"transport","ok":false,"error":"sentinel-secret"}\n' for _ in range(3)
        ),
        encoding="utf-8",
    )
    runner = ProcessJobRunner(
        _test_campaign_job,
        max_concurrent=1,
        max_queue=1,
        on_started=lambda config: None,
        on_finished=lambda config, result: None,
    )
    app = create_app(Settings(), job_runner=runner, paths=paths)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=False)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.started
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5) as client:
            accepted = client.post("/api/campaigns", json={"quick": True})
            campaign_id = accepted.json()["campaign_id"]
            deadline = time.monotonic() + 10
            while not (paths.runs / campaign_id / "verification.json").is_file():
                assert time.monotonic() < deadline
                time.sleep(0.02)
            health = client.get("/api/health")
            evidence = client.get(f"/api/campaigns/{campaign_id}/evidence")
            verification = client.get(f"/api/campaigns/{campaign_id}/verification")
            artifacts = client.get(f"/api/exports/{campaign_id}")
            downloads = [
                client.get(f"/api/exports/{campaign_id}/{name}")
                for name in (
                    "safe_envelope.json",
                    "ros2_parameters.yaml",
                    "test_safe_envelope.py",
                    "index.html",
                )
            ]
        assert health.status_code == 200
        assert health.json()["status"] == "degraded"
        assert "provider_transport_failures" in health.json()["reasons"]
        assert evidence.json()["spec"]["robot"] == "Panda"
        assert verification.json()["confirmation"]["passed"] is True
        assert set(artifacts.json()["artifacts"]) == {
            "index.html",
            "ros2_parameters.yaml",
            "safe_envelope.json",
            "test_safe_envelope.py",
        }
        assert all(response.status_code == 200 for response in downloads)
        assert "sentinel-secret" not in repr((health.json(), evidence.json(), verification.json()))
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert runner._thread is not None and not runner._thread.is_alive()
