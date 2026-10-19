from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from omnisim.agents.mitigator import facts_from_campaign, load_domain_bounds, mitigate
from omnisim.core.exports import EXPORTS_DIR, ExportArtifacts, export_campaign
from omnisim.core.orchestrator import CampaignOutcome
from omnisim.core.verify import (
    RefinementResult,
    VerificationCriteria,
    refine_and_verify,
    write_verification,
)
from omnisim.sim.detectors import DetectorConfig
from omnisim.sim.limits import load_urdf_limits

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
    )
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
