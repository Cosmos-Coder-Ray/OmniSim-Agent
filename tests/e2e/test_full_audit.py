from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import httpx
import uvicorn

from omnisim.api.main import create_app
from omnisim.api.paths import ApiPaths
from omnisim.config import Settings


class IdleRunner:
    def start(self) -> None:
        pass

    def submit(self, config):
        del config
        return True, None

    def queue_position(self, campaign_id):
        del campaign_id
        return None

    def queue_snapshot(self):
        return {"queued": 0, "running": 0, "max_concurrent": 1, "max_queue": 0}

    def cancel(self, campaign_id):
        del campaign_id
        return False

    def shutdown(self, grace_s=5.0):
        del grace_s


def test_api_serves_health_and_run_artifacts_over_http(tmp_path: Path) -> None:
    paths = ApiPaths.from_root(tmp_path)
    campaign_id = "f" * 32
    run_dir = paths.runs / campaign_id
    run_dir.mkdir(parents=True)
    (run_dir / "spec.json").write_text('{"spec":{"robot":"Panda"}}', encoding="utf-8")
    (run_dir / "verification.json").write_text('{"confirmation":{"passed":true}}', encoding="utf-8")
    app = create_app(Settings(), job_runner=IdleRunner(), paths=paths)
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
        app.state.store.create(campaign_id, {"quick": True})
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5) as client:
            health = client.get("/api/health")
            evidence = client.get(f"/api/campaigns/{campaign_id}/evidence")
            verification = client.get(f"/api/campaigns/{campaign_id}/verification")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert evidence.json()["spec"]["robot"] == "Panda"
        assert verification.json()["confirmation"]["passed"] is True
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        assert not thread.is_alive()
