from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from omnisim.agents.mitigator import facts_from_campaign, load_domain_bounds, mitigate
from omnisim.core.exports import EXPORTS_DIR, ExportArtifacts, export_campaign
from omnisim.core.orchestrator import CampaignOutcome
from omnisim.core.shield import SafetyShield
from omnisim.core.verify import (
    RefinementResult,
    VerificationCriteria,
    refine_and_verify,
    write_verification,
)
from omnisim.schemas import EpisodeResult, ScenarioParams
from omnisim.sim.detectors import DetectorConfig
from omnisim.sim.limits import load_urdf_limits
from omnisim.sim.runner import EpisodeBlocked, EpisodeError, EpisodeRunner

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class CampaignVerification:
    passed: bool | None
    skipped_reason: str | None = None
    refinement: RefinementResult | None = None
    verification_path: Path | None = None
    artifacts: ExportArtifacts | None = None


def run_campaign_verification(
    outcome: CampaignOutcome,
    *,
    robot: str,
    seed: int,
    workers: int,
    domain_path: Path | None = None,
    exports_root: Path = EXPORTS_DIR,
    stage_callback: Callable[[str], None] | None = None,
    tuning_count: int = 100,
    confirmation_count: int = 200,
    verification_runner=None,
) -> CampaignVerification:
    def set_stage(stage: str) -> None:
        if stage_callback is not None:
            stage_callback(stage)

    if not any(episode.failures for episode in outcome.result.episodes):
        set_stage("mitigation")
        set_stage("verification")
        set_stage("exports")
        return CampaignVerification(None, "campaign failures are absent")
    if not outcome.diagnoses:
        set_stage("mitigation")
        set_stage("verification")
        set_stage("exports")
        return CampaignVerification(None, "campaign diagnoses are absent")
    if outcome.spec is None:
        set_stage("mitigation")
        set_stage("verification")
        set_stage("exports")
        return CampaignVerification(None, "hardware evidence is unavailable")

    spec = outcome.spec
    domain_bounds = load_domain_bounds(domain_path)
    limits = load_urdf_limits()
    facts = facts_from_campaign(outcome.clusters, outcome.result.episodes, outcome.shrinks)

    def draft(iteration, previous, feedback):
        del iteration
        return mitigate(
            robot=robot,
            spec=spec,
            diagnoses=outcome.diagnoses,
            shrinks=outcome.shrinks,
            facts_by_cluster=facts,
            limits=limits,
            domain_bounds=domain_bounds,
            previous=previous,
            tuning_feedback=feedback,
            run_id=outcome.run_dir.name,
        )

    def report_stage(stage: str) -> None:
        set_stage(stage)

    refinement = refine_and_verify(
        draft,
        campaign_seed=seed,
        limits=limits,
        workers=workers,
        cfg=DetectorConfig(),
        domain_bounds=domain_bounds,
        discovery_episodes=outcome.result.episodes,
        stage_callback=report_stage,
        tuning_count=tuning_count,
        confirmation_count=confirmation_count,
        runner=verification_runner,
    )
    scenarios = refinement.confirmation.scenario_results
    cases: list[tuple[int, dict[str, Any]]] = []
    for predicate in (
        lambda off, on: bool(off["kinds"]) and bool(on["complete"]),
        lambda off, on: bool(off["kinds"]) and bool(on["blocked"]),
        lambda off, on: bool(off["kinds"]) and bool(on["kinds"]),
    ):
        for item in scenarios:
            off, on = item["off"], item["on"]
            if off["infrastructure_error"] or on["infrastructure_error"]:
                continue
            if predicate(off, on):
                cases.append((int(item["index"]), item))
        if len(cases) >= 3:
            break
    cases = sorted(dict(cases).items())[:3]
    showcases: list[dict[str, object]] = []
    if cases:
        runner = verification_runner or EpisodeRunner()
        own_runner = verification_runner is None
        try:
            for index, item in cases:
                params = ScenarioParams.model_validate(item["params"])
                off_dir = outcome.run_dir / "trajectories" / f"showcase_{index}_off"
                on_dir = outcome.run_dir / "trajectories" / f"showcase_{index}_on"
                off = runner.run_batch(
                    [params], limits, workers=workers, run_dir=off_dir, cfg=DetectorConfig()
                )[0]
                on = runner.run_batch(
                    [params],
                    limits,
                    workers=workers,
                    shield=SafetyShield(refinement.envelope),
                    run_dir=on_dir,
                    cfg=DetectorConfig(),
                )[0]
                off_kinds = (
                    sorted({event.kind for event in off.failures})
                    if isinstance(off, EpisodeResult)
                    else []
                )
                on_kinds = (
                    sorted({event.kind for event in on.failures})
                    if isinstance(on, EpisodeResult)
                    else []
                )
                if (
                    isinstance(off, (EpisodeError, EpisodeBlocked))
                    or off_kinds != item["off"]["kinds"]
                ):
                    continue
                blocked = isinstance(on, EpisodeBlocked)
                if blocked != item["on"]["blocked"] or on_kinds != item["on"]["kinds"]:
                    continue
                showcases.append(
                    {
                        "scenario_index": index,
                        "params": item["params"],
                        "off_kinds": off_kinds,
                        "on_kinds": on_kinds,
                        "on_blocked": blocked,
                        "off_trajectory": Path(off.trajectory_path)
                        .resolve()
                        .relative_to(outcome.run_dir.resolve())
                        .as_posix()
                        if isinstance(off, EpisodeResult) and off.trajectory_path
                        else None,
                        "on_trajectory": Path(on.trajectory_path)
                        .resolve()
                        .relative_to(outcome.run_dir.resolve())
                        .as_posix()
                        if isinstance(on, EpisodeResult) and on.trajectory_path
                        else None,
                    }
                )
        finally:
            if own_runner:
                runner.close()
    refinement = replace(refinement, showcases=tuple(showcases))
    base_verification = refinement.envelope.verification
    verification = {
        **base_verification,
        "passed": refinement.confirmation.passed,
        "provisional_criteria": {
            "unsafe_rate_max": VerificationCriteria().unsafe_rate_max,
            "blocked_rate_max": VerificationCriteria().blocked_rate_max,
            "completion_rate_min": VerificationCriteria().completion_rate_min,
        },
        "tuning_seed": refinement.tuning_seed,
        "confirmation_seed": refinement.confirmation_seed,
        "tuning": [item.to_dict() for item in refinement.tuning_results],
        "confirmation": refinement.confirmation.to_dict(),
        "showcases": list(refinement.showcases),
    }
    envelope = refinement.envelope.model_copy(update={"verification": verification})
    refinement = replace(refinement, envelope=envelope)
    verification_path = outcome.run_dir / "verification.json"
    write_verification(
        verification_path,
        {
            "tuning_seed": refinement.tuning_seed,
            "confirmation_seed": refinement.confirmation_seed,
            "tuning": [item.to_dict() for item in refinement.tuning_results],
            "confirmation": refinement.confirmation.to_dict(),
        },
    )
    benchmark_path = ROOT / "data" / "benchmark.json"
    set_stage("exports")
    artifacts = export_campaign(
        outcome.run_dir.name,
        envelope=envelope,
        refinement=refinement,
        shrinks=outcome.shrinks,
        diagnoses=outcome.diagnoses,
        spec=spec,
        output_root=exports_root,
        urdf_limits=limits,
        benchmark_summary=benchmark_path if benchmark_path.is_file() else None,
    )
    return CampaignVerification(
        refinement.confirmation.passed,
        refinement=refinement,
        verification_path=verification_path,
        artifacts=artifacts,
    )
