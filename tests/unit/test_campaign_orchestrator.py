from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from omnisim.core.orchestrator import campaign_events, run_campaign
from omnisim.llm.budget import BudgetExceeded
from omnisim.schemas import (
    CampaignEvent,
    DiagnosisReport,
    EpisodeResult,
    FailureEvent,
    HardwareSpec,
    ScenarioParams,
)
from omnisim.sim.limits import UrdfLimits
from omnisim.sim.runner import EpisodeError

FIXED_TIME = datetime(2026, 10, 18, tzinfo=UTC)


class FakeRunner:
    def __init__(
        self, *, failures: bool = False, first_error: bool = False, all_errors: bool = False
    ) -> None:
        self.failures = failures
        self.first_error = first_error
        self.all_errors = all_errors
        self.count = 0
        self.closed = False

    def run_batch(self, params_list, limits, *, workers, run_dir=None, cfg=None):
        results = []
        for params in params_list:
            episode_id = f"ep_{self.count:03d}"
            if self.all_errors or (self.first_error and self.count == 0):
                results.append(EpisodeError(episode_id, params, "WorkerTimeout", "timed out"))
            else:
                failures = (
                    [FailureEvent(kind="tracking_error", t=1.0, severity=1.2)]
                    if self.failures
                    else []
                )
                path = str(run_dir / f"{episode_id}.json") if run_dir else None
                results.append(
                    EpisodeResult(
                        episode_id=episode_id,
                        params=params,
                        failures=failures,
                        risk=1.2 if failures else 0.2,
                        max_tau_ratio=0.4,
                        max_vel_ratio=0.3,
                        final_ee_error_m=0.01,
                        min_manipulability=0.2,
                        trajectory_path=path,
                    )
                )
            self.count += 1
        return results

    def close(self) -> None:
        self.closed = True


def fake_limits() -> UrdfLimits:
    values = np.ones(7, dtype=np.float64)
    return UrdfLimits(values, values, -values, values)


def fake_spec() -> HardwareSpec:
    return HardwareSpec(
        robot="Panda",
        joint_torque_limits_nm=[],
        joint_velocity_limits=[],
        joint_position_limits=[],
        error_taxonomy=[],
    )


def nominal_run(params, limits, *, episode_id, run_dir, cfg):
    return EpisodeResult(
        episode_id=episode_id,
        params=params,
        failures=[],
        risk=0.2,
        max_tau_ratio=0.4,
        max_vel_ratio=0.3,
        final_ee_error_m=0.01,
        min_manipulability=0.2,
    )


def compare(spec, limits):
    return [], []


def mapping(kinds, spec, *, run_id=None):
    return {}


def unused_build_spec(*args, **kwargs):
    raise AssertionError("precomputed spec should be reused")


def diagnose_cluster(cluster, representative, minimal, spec, **kwargs):
    return DiagnosisReport(
        cluster_id=cluster.cluster_id,
        root_cause="measured perturbation increased risk at joint 2",
        mechanism="the tested motion deviated after the perturbation",
        numeric_claims=[],
        real_error_mapping=kwargs.get("mapping"),
        evidence_refs=[],
        confidence="low",
        recommended_constraints=[],
        corroborating_urls=[],
    )


def run_options(root: Path, runner: FakeRunner, *, budget: int = 4, **options):
    return {
        "robot": "Panda",
        "strategy": "random",
        "seed": 17,
        "budget": budget,
        "run_id": "fixed-run",
        "runs_root": root,
        "runner": runner,
        "run_episode_fn": nominal_run,
        "spec": fake_spec(),
        "build_spec_fn": unused_build_spec,
        "compare_fn": compare,
        "mapping_fn": mapping,
        "known_issues_fn": lambda robot, kind, *, run_id=None: [],
        "diagnose_fn": diagnose_cluster,
        "limits": fake_limits(),
        "clock": lambda: FIXED_TIME,
        **options,
    }


def normalized_events(events: list[CampaignEvent]) -> list[object]:
    def strip(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: strip(item) for key, item in value.items() if key not in {"ts", "duration_s"}
            }
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    return [strip(event.model_dump(mode="json")) for event in events]


def test_budget_stop_persists_events_and_relative_paths(tmp_path: Path) -> None:
    runner = FakeRunner()
    outcome = run_campaign(**run_options(tmp_path, runner))

    assert outcome.result.events[0].type == "campaign_started"
    assert outcome.result.events[-1].data == {
        "status": "no_failures_found",
        "stop_reason": "budget",
        "sims_used": 4,
        "shrink_sims": 0,
        "clusters": 0,
        "duration_s": 0.0,
    }
    assert runner.closed
    assert all(
        not Path(episode.trajectory_path).is_absolute()
        for episode in outcome.result.episodes
        if episode.trajectory_path
    )
    saved = [
        CampaignEvent.model_validate_json(line)
        for line in (outcome.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [item.model_dump(mode="json") for item in saved] == [
        item.model_dump(mode="json") for item in outcome.result.events
    ]
    persisted = json.loads((outcome.run_dir / "campaign.json").read_text(encoding="utf-8"))
    assert persisted["events"] == [item.model_dump(mode="json") for item in outcome.result.events]


def test_cluster_stop_counts_base_keys_and_episode_errors_consume_budget(tmp_path: Path) -> None:
    runner = FakeRunner(failures=True, first_error=True)
    outcome = run_campaign(
        **run_options(tmp_path, runner, budget=8, stop_after_clusters=1, max_diagnosed=0)
    )

    done = outcome.result.events[-1].data
    assert done["status"] == "completed"
    assert done["stop_reason"] == "clusters"
    assert done["sims_used"] == 4
    assert len(outcome.result.episodes) == 3
    assert any(
        event.type == "episode_done" and event.data["ok"] is False
        for event in outcome.result.events
    )
    assert done["clusters"] == 1


def test_cluster_stop_threshold_counts_base_keys_not_subclusters(tmp_path: Path) -> None:
    outcome = run_campaign(
        **run_options(
            tmp_path,
            FakeRunner(failures=True),
            budget=12,
            stop_after_clusters=2,
            max_diagnosed=0,
        )
    )

    done = outcome.result.events[-1].data
    assert done["status"] == "completed"
    assert done["stop_reason"] == "budget"
    assert done["sims_used"] == 12
    assert done["clusters"] == 2


def test_campaign_with_no_completed_search_episodes_is_an_error(tmp_path: Path) -> None:
    outcome = run_campaign(**run_options(tmp_path, FakeRunner(all_errors=True), budget=4))

    done = outcome.result.events[-1].data
    assert done["status"] == "error"
    assert done["stop_reason"] == "none"
    assert outcome.result.episodes == []


def test_nominal_failure_emits_started_then_error(tmp_path: Path) -> None:
    runner = FakeRunner()

    def failed_nominal(*args, **kwargs):
        raise RuntimeError("nominal failed")

    outcome = run_campaign(**run_options(tmp_path, runner, run_episode_fn=failed_nominal))

    assert [event.type for event in outcome.result.events] == [
        "campaign_started",
        "campaign_done",
    ]
    assert outcome.result.events[0].data["baseline_risk"] is None
    assert outcome.result.events[0].data["baseline_error"] == "RuntimeError: nominal failed"
    assert outcome.result.events[-1].data["status"] == "error"
    assert outcome.result.events[-1].data["stop_reason"] == "none"
    assert runner.closed


def test_token_cap_is_a_separate_status_from_stop_reason(tmp_path: Path) -> None:
    runner = FakeRunner()

    def capped_chat(*args, **kwargs):
        raise BudgetExceeded("cap reached")

    outcome = run_campaign(
        **run_options(
            tmp_path,
            runner,
            strategy="nemotron",
            chat_json=capped_chat,
        )
    )

    done = outcome.result.events[-1].data
    assert done["status"] == "token_cap"
    assert done["stop_reason"] == "token_cap"
    assert done["sims_used"] == 0


def test_generator_close_persists_cancellation_and_run_id_is_exclusive(tmp_path: Path) -> None:
    runner = FakeRunner()
    generator = campaign_events(**run_options(tmp_path, runner))
    first = next(generator)
    assert first.type == "campaign_started"
    generator.close()

    persisted = json.loads((tmp_path / "fixed-run" / "campaign.json").read_text(encoding="utf-8"))
    assert persisted["events"][-1]["data"]["status"] == "cancelled"
    assert runner.closed
    with pytest.raises(FileExistsError):
        run_campaign(**run_options(tmp_path, FakeRunner()))


def test_fixed_run_is_deterministic_after_removing_time_fields(tmp_path: Path) -> None:
    first = run_campaign(**run_options(tmp_path / "first", FakeRunner()))
    second = run_campaign(**run_options(tmp_path / "second", FakeRunner()))

    assert normalized_events(first.result.events) == normalized_events(second.result.events)


def test_shrink_writes_the_minimal_repro_with_a_relative_path(tmp_path: Path) -> None:
    runner = FakeRunner(failures=True)

    def failing_shrink_run(params, limits, *, episode_id, run_dir, cfg):
        failures = (
            []
            if episode_id == "nominal"
            else [FailureEvent(kind="tracking_error", t=1.0, severity=1.2)]
        )
        path = None
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            replay = run_dir / f"{episode_id}.json"
            replay.write_text("{}", encoding="utf-8")
            path = str(replay)
        return EpisodeResult(
            episode_id=episode_id,
            params=params,
            failures=failures,
            risk=1.2 if failures else 0.2,
            max_tau_ratio=0.4,
            max_vel_ratio=0.3,
            final_ee_error_m=0.01,
            min_manipulability=0.2,
            trajectory_path=path,
        )

    outcome = run_campaign(
        **run_options(
            tmp_path,
            runner,
            stop_after_clusters=1,
            max_diagnosed=1,
            run_episode_fn=failing_shrink_run,
        )
    )

    cluster_id = "tracking_error:global"
    shrink_result = outcome.shrinks[cluster_id]
    assert all(
        getattr(shrink_result.minimal, name) == getattr(ScenarioParams(), name)
        for name in ScenarioParams.model_fields
        if name not in {"seed"}
    )
    assert shrink_result.sims_used <= 26
    trajectory = "trajectories/shrunk_tracking_error_global.json"
    assert outcome.run_dir.joinpath(*trajectory.split("/")).is_file()
    saved = json.loads((outcome.run_dir / "shrinks.json").read_text(encoding="utf-8"))
    assert saved[cluster_id]["trajectory_path"] == trajectory


def test_diagnosis_validation_error_is_reported_without_failing_sim_search(
    tmp_path: Path,
) -> None:
    runner = FakeRunner(failures=True)

    def invalid_diagnosis(*args, **kwargs):
        raise ValueError("unsupported numeric claim")

    outcome = run_campaign(
        **run_options(
            tmp_path,
            runner,
            stop_after_clusters=1,
            max_diagnosed=1,
            diagnose_fn=invalid_diagnosis,
        )
    )

    ready = next(event for event in outcome.result.events if event.type == "diagnosis_ready")
    assert ready.data["ok"] is False
    assert "unsupported numeric claim" in ready.data["error"]
    assert outcome.result.events[-1].data["status"] == "completed"
