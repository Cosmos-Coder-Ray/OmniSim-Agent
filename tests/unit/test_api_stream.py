from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from omnisim.api.stream import EventTailer
from omnisim.schemas import CampaignEvent


def event(
    seq: int, kind: Literal["campaign_started", "campaign_done"] = "campaign_started"
) -> CampaignEvent:
    if kind == "campaign_done":
        data = {
            "status": "completed",
            "stop_reason": "budget",
            "sims_used": 0,
            "shrink_sims": 0,
            "clusters": 0,
            "duration_s": 0.0,
        }
    else:
        data = {
            "run_id": "test-run",
            "robot": "franka_panda",
            "strategy": "random",
            "seed": 17,
            "budget": 10,
            "started_at": "2026-10-19T00:00:00Z",
            "baseline_risk": None,
            "baseline_max_tau_ratio": None,
            "baseline_max_vel_ratio": None,
            "baseline_final_ee_error_m": None,
            "baseline_error": None,
        }
    return CampaignEvent(seq=seq, ts=datetime(2026, 10, 19, tzinfo=UTC), type=kind, data=data)


def test_tailer_waits_for_newline_before_parsing_partial_event(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    first = event(1).model_dump_json()
    second = event(2, "campaign_done").model_dump_json()
    path.write_bytes(f"{first}\n{second[:20]}".encode())
    tailer = EventTailer(path)

    assert [item.seq for item in tailer.read_complete()] == [1]
    assert tailer.offset == path.stat().st_size

    with path.open("ab") as stream:
        stream.write(f"{second[20:]}\n".encode())
    assert [item.seq for item in tailer.read_complete()] == [2]
    assert tailer.read_complete() == []


def test_tailer_resumes_after_last_event_id(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    events = [event(1), event(2), event(3, "campaign_done")]
    path.write_text("".join(item.model_dump_json() + "\n" for item in events), encoding="utf-8")
    tailer = EventTailer(path)

    tailer.resume_after(1)

    assert [item.seq for item in tailer.read_complete()] == [2, 3]
