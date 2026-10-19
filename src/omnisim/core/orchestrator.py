from __future__ import annotations

import json
import re
from collections.abc import Callable, Generator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from omnisim.agents.baselines import Trial
from omnisim.agents.diagnostician import diagnose
from omnisim.agents.factory import make_strategy
from omnisim.core.campaign import BatchRunner, iter_trial_batches
from omnisim.core.clustering import ClusteringResult, ClusterSummary, atlas_points, cluster_episodes
from omnisim.core.shield import EpisodeBlocked
from omnisim.core.shrinker import ShrinkResult, shrink
from omnisim.knowledge.compare import compare_with_urdf
from omnisim.knowledge.tavily_agent import (
    build_spec,
    map_failures_to_real_errors,
    search_known_issues,
)
from omnisim.llm.budget import BudgetExceeded
from omnisim.schemas import (
    CampaignDoneData,
    CampaignEvent,
    CampaignEventType,
    CampaignResult,
    DiagnosisReport,
    EpisodeResult,
    ErrorMapping,
    FailureEvent,
    FailureKind,
    HardwareSpec,
    Mismatch,
    ScenarioParams,
    StrategyName,
)
from omnisim.sim.detectors import DetectorConfig
from omnisim.sim.limits import UrdfLimits, load_urdf_limits
from omnisim.sim.runner import EpisodeRunner, run_episode

RUNS_DIR = Path(__file__).resolve().parents[3] / "runs"
Clock = Callable[[], datetime]
StageCallback = Callable[[str], None]
CancelCheck = Callable[[], bool]
EpisodeFn = Callable[..., EpisodeResult | EpisodeBlocked]
BuildSpecFn = Callable[..., HardwareSpec]
CompareFn = Callable[..., tuple[list[Mismatch], list]]
MappingFn = Callable[..., dict[FailureKind, ErrorMapping]]
KnownIssuesFn = Callable[..., list[dict[str, str]]]
DiagnoseFn = Callable[..., DiagnosisReport]
RunStatus = Literal["completed", "cancelled", "error", "token_cap", "no_failures_found"]
StopReason = Literal["budget", "clusters", "token_cap", "none"]


@dataclass(frozen=True, slots=True)
class CampaignOutcome:
    result: CampaignResult
    run_dir: Path
    clusters: tuple[ClusterSummary, ...]
    spec: HardwareSpec | None
    mismatches: tuple[Mismatch, ...]
    error_mappings: dict[FailureKind, ErrorMapping]
    shrinks: dict[str, ShrinkResult]
    diagnoses: dict[str, DiagnosisReport]
    error: str | None = None


def _now() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("campaign clock must return timezone-aware datetimes")
    return value.astimezone(UTC)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _selected_event(events: list[FailureEvent]) -> FailureEvent:
    order = {kind: index for index, kind in enumerate(FailureKind.__args__)}
    return min(
        events,
        key=lambda event: (
            -event.severity,
            order[event.kind],
            event.joint is not None,
            event.joint if event.joint is not None else -1,
        ),
    )


def _safe_cluster_name(cluster_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", cluster_id)


def _relative_result(result: EpisodeResult, run_dir: Path) -> EpisodeResult:
    if result.trajectory_path is None:
        return result
    path = Path(result.trajectory_path)
    if path.is_absolute():
        path = path.resolve().relative_to(run_dir.resolve())
    return result.model_copy(update={"trajectory_path": path.as_posix()})


def _episode_result(result: EpisodeResult | EpisodeBlocked) -> EpisodeResult:
    if isinstance(result, EpisodeBlocked):
        raise RuntimeError("campaign episode unexpectedly returned EpisodeBlocked")
    return result


def campaign_events(
    robot: str,
    strategy: StrategyName,
    *,
    seed: int,
    budget: int,
    workers: int = 1,
    stop_after_clusters: int = 3,
    max_diagnosed: int = 2,
    run_id: str | None = None,
    runs_root: Path = RUNS_DIR,
    runner: BatchRunner | None = None,
    run_episode_fn: EpisodeFn = run_episode,
    chat_json: Callable | None = None,
    spec: HardwareSpec | None = None,
    build_spec_fn: BuildSpecFn = build_spec,
    compare_fn: CompareFn = compare_with_urdf,
    mapping_fn: MappingFn = map_failures_to_real_errors,
    known_issues_fn: KnownIssuesFn = search_known_issues,
    diagnose_fn: DiagnoseFn = diagnose,
    clock: Clock = _now,
    on_stage: StageCallback | None = None,
    cancel_requested: CancelCheck | None = None,
    limits: UrdfLimits | None = None,
    cfg: DetectorConfig = DetectorConfig(),
) -> Generator[CampaignEvent, None, CampaignOutcome]:
    if budget <= 0 or workers <= 0 or stop_after_clusters <= 0 or max_diagnosed < 0:
        raise ValueError("budget, workers, and stop_after_clusters must be positive")

    started_at = _as_utc(clock())
    campaign_id = run_id or f"{robot}_{strategy}_{seed}_{started_at:%Y%m%dT%H%M%SZ}"
    if campaign_id in {"", ".", ".."} or Path(campaign_id).name != campaign_id:
        raise ValueError("run_id must be a single path component")
    runs_root.mkdir(parents=True, exist_ok=True)
    run_dir = runs_root / campaign_id
    run_dir.mkdir(exist_ok=False)
    trajectory_dir = run_dir / "trajectories"
    trajectory_dir.mkdir()
    episodes_path = run_dir / "episodes.jsonl"
    events_path = run_dir / "events.jsonl"

    active_runner = runner or EpisodeRunner()
    active_limits = limits or load_urdf_limits()
    episodes: list[EpisodeResult] = []
    events: list[CampaignEvent] = []
    trials: list[Trial] = []
    clusters: tuple[ClusterSummary, ...] = ()
    spec_result = spec
    mismatches: tuple[Mismatch, ...] = ()
    error_mappings: dict[FailureKind, ErrorMapping] = {}
    shrinks: dict[str, ShrinkResult] = {}
    diagnoses: dict[str, DiagnosisReport] = {}
    seq = 0
    finished = False
    error_message: str | None = None
    stop_reason: StopReason = "budget"
    status: RunStatus = "completed"

    events_file = events_path.open("w", encoding="utf-8")
    episodes_file = episodes_path.open("w", encoding="utf-8")

    def set_stage(stage: str) -> None:
        if on_stage is not None:
            on_stage(stage)

    def is_cancelled() -> bool:
        return cancel_requested is not None and cancel_requested()

    def emit(event_type: CampaignEventType, data: dict) -> CampaignEvent:
        nonlocal seq
        event = CampaignEvent(seq=seq + 1, ts=_as_utc(clock()), type=event_type, data=data)
        seq = event.seq
        events.append(event)
        events_file.write(event.model_dump_json() + "\n")
        events_file.flush()
        return event

    def save_outputs() -> CampaignResult:
        result = CampaignResult(
            robot=robot,
            budget=budget,
            strategy=strategy,
            seed=seed,
            episodes=episodes,
            events=events,
            source="live",
        )
        _write_json(run_dir / "campaign.json", result.model_dump(mode="json"))
        return result

    def complete(done_status: RunStatus, reason: StopReason) -> CampaignOutcome:
        duration = max(0.0, (_as_utc(clock()) - started_at).total_seconds())
        emit(
            "campaign_done",
            CampaignDoneData(
                status=done_status,
                stop_reason=reason,
                sims_used=len(trials),
                shrink_sims=sum(item.sims_used for item in shrinks.values()),
                clusters=len(clusters),
                duration_s=duration,
            ).model_dump(),
        )
        result = save_outputs()
        return CampaignOutcome(
            result=result,
            run_dir=run_dir,
            clusters=clusters,
            spec=spec_result,
            mismatches=mismatches,
            error_mappings=error_mappings,
            shrinks=shrinks,
            diagnoses=diagnoses,
            error=error_message,
        )

    def write_clustering(current: ClusteringResult) -> None:
        _write_json(
            run_dir / "clusters.json",
            {
                "clusters": [asdict(cluster) for cluster in current.clusters],
                "assignments": current.assignments,
                "atlas_points": atlas_points(
                    episodes, current.assignments, "payload_kg", "push_force_n"
                ),
            },
        )

    def append_episode(result: EpisodeResult) -> None:
        episodes.append(result)
        episodes_file.write(result.model_dump_json() + "\n")
        episodes_file.flush()

    def persist_auxiliary() -> None:
        nonlocal clusters
        current = cluster_episodes(episodes, seed)
        clusters = current.clusters
        write_clustering(current)
        _write_json(
            run_dir / "shrinks.json",
            {
                cluster_id: {
                    "cluster_id": cluster_id,
                    "minimal": result.minimal.model_dump(mode="json"),
                    "sims_used": result.sims_used,
                    "converged": result.converged,
                    "trace": result.trace,
                    "trajectory_path": (
                        f"trajectories/shrunk_{_safe_cluster_name(cluster_id)}.json"
                        if (
                            trajectory_dir / f"shrunk_{_safe_cluster_name(cluster_id)}.json"
                        ).exists()
                        else None
                    ),
                }
                for cluster_id, result in shrinks.items()
            },
        )
        _write_json(
            run_dir / "diagnoses.json",
            {
                cluster_id: report.model_dump(mode="json")
                for cluster_id, report in diagnoses.items()
            },
        )

    set_stage("search")
    try:
        nominal: EpisodeResult | None = None
        nominal_error: str | None = None
        try:
            nominal = _episode_result(
                run_episode_fn(
                    ScenarioParams(seed=seed),
                    active_limits,
                    episode_id="nominal",
                    run_dir=trajectory_dir,
                    cfg=cfg,
                )
            )
            nominal = _relative_result(nominal, run_dir)
            _write_json(run_dir / "nominal.json", nominal.model_dump(mode="json"))
        except Exception as error:
            nominal_error = f"{type(error).__name__}: {error}"
            _write_json(run_dir / "nominal.json", {"result": None, "error": nominal_error})

        yield emit(
            "campaign_started",
            {
                "run_id": campaign_id,
                "robot": robot,
                "strategy": strategy,
                "seed": seed,
                "budget": budget,
                "started_at": started_at,
                "baseline_risk": nominal.risk if nominal else None,
                "baseline_max_tau_ratio": nominal.max_tau_ratio if nominal else None,
                "baseline_max_vel_ratio": nominal.max_vel_ratio if nominal else None,
                "baseline_final_ee_error_m": nominal.final_ee_error_m if nominal else None,
                "baseline_error": nominal_error,
            },
        )
        if nominal_error is not None:
            persist_auxiliary()
            outcome = complete("error", "none")
            finished = True
            return outcome

        if is_cancelled():
            status = "cancelled"
            stop_reason = "none"

        active_strategy = make_strategy(strategy, seed=seed, budget=budget, chat_json=chat_json)
        reached_cluster_limit = False

        def should_stop(history: list[Trial]) -> bool:
            nonlocal reached_cluster_limit
            search_episodes = [trial.result for trial in history if trial.result is not None]
            current = cluster_episodes(search_episodes, seed)
            reached_cluster_limit = len(current.base_keys) >= stop_after_clusters
            return reached_cluster_limit

        try:
            batches = (
                iter_trial_batches(
                    active_strategy,
                    active_runner,
                    active_limits,
                    benchmark_seed=seed,
                    budget=budget,
                    workers=workers,
                    cfg=cfg,
                    stop_predicate=should_stop,
                    run_dir=trajectory_dir,
                )
                if status != "cancelled"
                else ()
            )
            for batch in batches:
                if is_cancelled():
                    status = "cancelled"
                    stop_reason = "none"
                    break
                yield emit(
                    "hypothesis",
                    {
                        "batch_index": batch.index,
                        "hypothesis": batch.hypothesis,
                        "origins": batch.origins,
                        "params": [trial.params for trial in batch.trials],
                    },
                )
                yield emit(
                    "scenario_batch",
                    {
                        "batch_index": batch.index,
                        "episode_ids": [
                            trial.result.episode_id if trial.result else f"ep_{len(trials) + i:03d}"
                            for i, trial in enumerate(batch.trials)
                        ],
                        "sims_used": len(trials) + len(batch.trials),
                    },
                )
                for trial in batch.trials:
                    if is_cancelled():
                        status = "cancelled"
                        stop_reason = "none"
                        break
                    if trial.result is None:
                        trials.append(trial)
                        yield emit(
                            "episode_done",
                            {
                                "episode_id": f"ep_{len(trials) - 1:03d}",
                                "ok": False,
                                "error": trial.error or "episode failed",
                            },
                        )
                        continue

                    result = _relative_result(trial.result, run_dir)
                    normalized_trial = Trial(params=trial.params, result=result, error=None)
                    trials.append(normalized_trial)
                    append_episode(result)
                    yield emit(
                        "episode_done",
                        {
                            "episode_id": result.episode_id,
                            "ok": True,
                            "risk": result.risk,
                            "failure_kinds": list(
                                dict.fromkeys(event.kind for event in result.failures)
                            ),
                            "duration_s": batch.duration_s / max(1, len(batch.trials)),
                            "ticks": 360,
                            "control_hz": 60,
                        },
                    )
                    if result.failures:
                        current = cluster_episodes(episodes, seed)
                        selected = _selected_event(result.failures)
                        cluster_id = current.assignments[result.episode_id]
                        if cluster_id is not None:
                            yield emit(
                                "failure_found",
                                {
                                    "episode_id": result.episode_id,
                                    "cluster_id": cluster_id,
                                    "kind": selected.kind,
                                    "joint": selected.joint,
                                    "risk": result.risk,
                                    "severity": selected.severity,
                                },
                            )
                if status == "cancelled":
                    break
                current = cluster_episodes(episodes, seed)
                clusters = current.clusters
                write_clustering(current)
                yield emit(
                    "cluster_updated",
                    {
                        "cluster_count": len(current.clusters),
                        "base_key_count": len(current.base_keys),
                        "cluster_ids": [cluster.cluster_id for cluster in current.clusters],
                    },
                )
        except BudgetExceeded as error:
            status = "token_cap"
            stop_reason = "token_cap"
            error_message = str(error)
        else:
            if status == "completed":
                stop_reason = "clusters" if reached_cluster_limit else "budget"

        final_clustering = cluster_episodes(episodes, seed)
        clusters = final_clustering.clusters
        write_clustering(final_clustering)

        episode_by_id = {episode.episode_id: episode for episode in episodes}
        ranked_clusters = sorted(
            clusters,
            key=lambda item: (-item.best_risk, -item.max_severity, item.cluster_id),
        )
        set_stage("shrink")
        if status != "cancelled":
            for cluster in ranked_clusters[:max_diagnosed]:
                if is_cancelled():
                    status = "cancelled"
                    stop_reason = "none"
                    break
                representative = max(
                    (episode_by_id[episode_id] for episode_id in cluster.episode_ids),
                    key=lambda item: (item.risk, item.episode_id),
                )

                def evaluate(candidate: ScenarioParams) -> EpisodeResult:
                    return _episode_result(
                        run_episode_fn(
                            candidate,
                            active_limits,
                            episode_id=f"shrink_{_safe_cluster_name(cluster.cluster_id)}",
                            run_dir=None,
                            cfg=cfg,
                        )
                    )

                shrunk = shrink(
                    representative.params,
                    (cluster.kind, cluster.joint),
                    evaluate,
                    cap=25,
                )
                if is_cancelled():
                    status = "cancelled"
                    stop_reason = "none"
                    break
                repro = _episode_result(
                    run_episode_fn(
                        shrunk.minimal,
                        active_limits,
                        episode_id=f"shrunk_{_safe_cluster_name(cluster.cluster_id)}",
                        run_dir=trajectory_dir,
                        cfg=cfg,
                    )
                )
                repro = _relative_result(repro, run_dir)
                shrinks[cluster.cluster_id] = ShrinkResult(
                    minimal=shrunk.minimal,
                    sims_used=shrunk.sims_used + 1,
                    converged=shrunk.converged,
                    trace=shrunk.trace,
                )
                yield emit(
                    "shrink_done",
                    {
                        "cluster_id": cluster.cluster_id,
                        "episode_id": repro.episode_id,
                        "sims_used": shrunk.sims_used + 1,
                        "converged": shrunk.converged,
                        "trajectory_path": repro.trajectory_path,
                    },
                )
                persist_auxiliary()

        set_stage("spec")
        try:
            if is_cancelled():
                status = "cancelled"
                stop_reason = "none"
            elif not episodes and trials and status == "completed":
                raise RuntimeError("no campaign episodes completed")
            elif status != "cancelled":
                if spec_result is None:
                    spec_result = build_spec_fn(robot, run_id=campaign_id)
                mismatches_list, not_compared = compare_fn(spec_result, active_limits)
                mismatches = tuple(mismatches_list)
                failure_kinds = list(
                    dict.fromkeys(event.kind for episode in episodes for event in episode.failures)
                )
                if failure_kinds:
                    error_mappings = mapping_fn(failure_kinds, spec_result, run_id=campaign_id)
                _write_json(
                    run_dir / "spec.json",
                    {
                        "spec": spec_result.model_dump(mode="json"),
                        "mismatches": [item.model_dump(mode="json") for item in mismatches],
                        "not_compared": [item.model_dump(mode="json") for item in not_compared],
                        "error_mappings": {
                            key: value.model_dump(mode="json")
                            for key, value in error_mappings.items()
                        },
                    },
                )
                yield emit(
                    "spec_ready",
                    {
                        "ok": True,
                        "robot": spec_result.robot,
                        "mismatch_count": len(mismatches),
                        "spec_origin": "provided" if spec is not None else "built",
                    },
                )
        except BudgetExceeded as error:
            status = "token_cap"
            stop_reason = "token_cap"
            error_message = str(error)
            yield emit("spec_ready", {"ok": False, "error": error_message})
        except Exception as error:
            status = "error"
            stop_reason = "none"
            error_message = f"{type(error).__name__}: {error}"
            yield emit("spec_ready", {"ok": False, "error": error_message})
            _write_json(run_dir / "spec.json", {"spec": None, "error": error_message})

        set_stage("diagnosis")
        if status != "cancelled":
            for cluster in ranked_clusters[:max_diagnosed]:
                if is_cancelled():
                    status = "cancelled"
                    stop_reason = "none"
                    break
                if status == "token_cap":
                    yield emit(
                        "diagnosis_ready",
                        {
                            "cluster_id": cluster.cluster_id,
                            "ok": False,
                            "error": "token cap reached",
                        },
                    )
                    continue
                if spec_result is None:
                    yield emit(
                        "diagnosis_ready",
                        {
                            "cluster_id": cluster.cluster_id,
                            "ok": False,
                            "error": "hardware spec unavailable",
                        },
                    )
                    continue
                representative = max(
                    (episode_by_id[episode_id] for episode_id in cluster.episode_ids),
                    key=lambda item: (item.risk, item.episode_id),
                )
                minimal = shrinks.get(cluster.cluster_id)
                try:
                    known_issues = known_issues_fn(robot, cluster.kind, run_id=campaign_id)
                    report = diagnose_fn(
                        cluster,
                        representative,
                        minimal.minimal if minimal is not None else representative.params,
                        spec_result,
                        mapping=error_mappings.get(cluster.kind),
                        known_issues=known_issues,
                        chat_json=chat_json,
                        run_id=campaign_id,
                    )
                except BudgetExceeded as error:
                    status = "token_cap"
                    stop_reason = "token_cap"
                    error_message = str(error)
                    yield emit(
                        "diagnosis_ready",
                        {"cluster_id": cluster.cluster_id, "ok": False, "error": error_message},
                    )
                except Exception as error:
                    yield emit(
                        "diagnosis_ready",
                        {
                            "cluster_id": cluster.cluster_id,
                            "ok": False,
                            "error": f"{type(error).__name__}: {error}",
                        },
                    )
                else:
                    diagnoses[cluster.cluster_id] = report
                    _write_json(
                        run_dir / "diagnoses.json",
                        {
                            cluster_id: item.model_dump(mode="json")
                            for cluster_id, item in diagnoses.items()
                        },
                    )
                    yield emit(
                        "diagnosis_ready",
                        {
                            "cluster_id": cluster.cluster_id,
                            "ok": True,
                            "confidence": report.confidence,
                        },
                    )

        if status == "completed":
            status = (
                "no_failures_found" if not any(item.failures for item in episodes) else "completed"
            )

        persist_auxiliary()
        outcome = complete(status, stop_reason)
        finished = True
        return outcome
    except GeneratorExit:
        if not finished:
            persist_auxiliary()
            complete("cancelled", "none")
            finished = True
        raise
    except KeyboardInterrupt:
        if not finished:
            persist_auxiliary()
            complete("cancelled", "none")
            finished = True
        raise
    except BudgetExceeded as error:
        error_message = str(error)
        persist_auxiliary()
        outcome = complete("token_cap", "token_cap")
        finished = True
        return outcome
    except Exception as error:
        error_message = f"{type(error).__name__}: {error}"
        persist_auxiliary()
        outcome = complete("error", "none")
        finished = True
        return outcome
    finally:
        episodes_file.close()
        events_file.close()
        active_runner.close()


def run_campaign(
    robot: str,
    strategy: StrategyName,
    *,
    seed: int,
    budget: int,
    workers: int = 1,
    stop_after_clusters: int = 3,
    max_diagnosed: int = 2,
    run_id: str | None = None,
    runs_root: Path = RUNS_DIR,
    runner: BatchRunner | None = None,
    run_episode_fn: EpisodeFn = run_episode,
    chat_json: Callable | None = None,
    spec: HardwareSpec | None = None,
    build_spec_fn: BuildSpecFn = build_spec,
    compare_fn: CompareFn = compare_with_urdf,
    mapping_fn: MappingFn = map_failures_to_real_errors,
    known_issues_fn: KnownIssuesFn = search_known_issues,
    diagnose_fn: DiagnoseFn = diagnose,
    clock: Clock = _now,
    limits: UrdfLimits | None = None,
    cfg: DetectorConfig = DetectorConfig(),
) -> CampaignOutcome:
    generator = campaign_events(
        robot,
        strategy,
        seed=seed,
        budget=budget,
        workers=workers,
        stop_after_clusters=stop_after_clusters,
        max_diagnosed=max_diagnosed,
        run_id=run_id,
        runs_root=runs_root,
        runner=runner,
        run_episode_fn=run_episode_fn,
        chat_json=chat_json,
        spec=spec,
        build_spec_fn=build_spec_fn,
        compare_fn=compare_fn,
        mapping_fn=mapping_fn,
        known_issues_fn=known_issues_fn,
        diagnose_fn=diagnose_fn,
        clock=clock,
        limits=limits,
        cfg=cfg,
    )
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            return stop.value
