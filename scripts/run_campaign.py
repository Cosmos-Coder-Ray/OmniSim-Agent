from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

from rich.console import Console

from omnisim.agents.mitigator import facts_from_campaign, load_domain_bounds, mitigate
from omnisim.core.exports import EXPORTS_DIR, export_campaign
from omnisim.core.orchestrator import RUNS_DIR, campaign_events
from omnisim.core.verify import VerificationCriteria, refine_and_verify, write_verification
from omnisim.schemas import CampaignEvent, StrategyName
from omnisim.sim.detectors import DetectorConfig
from omnisim.sim.limits import load_urdf_limits


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an OmniSim adversarial campaign")
    parser.add_argument("--robot", default="franka_panda")
    parser.add_argument("--strategy", choices=("nemotron", "random", "cmaes"), default="nemotron")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--stop-after-clusters", type=int, default=3)
    parser.add_argument("--max-diagnosed", type=int, default=2)
    parser.add_argument("--run-id")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--domain", type=Path)
    return parser


def _print_event(console: Console, event: CampaignEvent) -> None:
    data = event.data
    if event.type == "campaign_started":
        console.print(f"[{event.seq}] campaign started, baseline risk {data['baseline_risk']}")
    elif event.type == "episode_done":
        if data["ok"]:
            kinds = ", ".join(data["failure_kinds"]) or "no failures"
            console.print(f"[{event.seq}] {data['episode_id']}: risk {data['risk']:.3f}, {kinds}")
        else:
            console.print(f"[{event.seq}] {data['episode_id']}: {data['error']}")
    elif event.type in {"spec_ready", "diagnosis_ready"} and not data["ok"]:
        console.print(f"[{event.seq}] {event.type}: {data['error']}")
    elif event.type == "campaign_done":
        console.print(
            f"[{event.seq}] campaign {data['status']} ({data['stop_reason']}), "
            f"{data['sims_used']} simulations, {data['clusters']} clusters"
        )


def _print_verification(console: Console, label: str, result) -> None:
    cycle_cost = (
        f"{result.mean_relative_cycle_time_increase:.1%}"
        if result.mean_relative_cycle_time_increase is not None
        else "n/a"
    )
    console.print(
        f"{label}: fail rate {result.fail_rate_before:.1%} -> {result.fail_rate_after:.1%}, "
        f"blocked {result.blocked_rate:.1%}, completion {result.completion_rate:.1%}, "
        f"cycle-time change {cycle_cost}"
    )


def _verify_campaign(args: argparse.Namespace, outcome, console: Console) -> bool | None:
    has_failures = any(episode.failures for episode in outcome.result.episodes)
    if not has_failures:
        console.print("verification skipped: campaign failures are absent", style="yellow")
        return None
    if not outcome.diagnoses:
        console.print("verification skipped: campaign diagnoses are absent", style="yellow")
        return None
    if outcome.spec is None:
        console.print("verification skipped: hardware evidence is unavailable", style="yellow")
        return None

    domain_bounds = load_domain_bounds(args.domain)
    limits = load_urdf_limits()
    facts = facts_from_campaign(outcome.clusters, outcome.result.episodes, outcome.shrinks)

    def draft(iteration, previous, feedback):
        del iteration
        return mitigate(
            robot=args.robot,
            spec=outcome.spec,
            diagnoses=outcome.diagnoses,
            shrinks=outcome.shrinks,
            facts_by_cluster=facts,
            limits=limits,
            domain_bounds=domain_bounds,
            previous=previous,
            tuning_feedback=feedback,
            run_id=outcome.run_dir.name,
        )

    refinement = refine_and_verify(
        draft,
        campaign_seed=args.seed,
        limits=limits,
        workers=args.workers,
        cfg=DetectorConfig(),
        domain_bounds=domain_bounds,
        discovery_episodes=outcome.result.episodes,
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
    benchmark_path = Path(__file__).resolve().parents[1] / "data" / "benchmark.json"
    artifacts = export_campaign(
        outcome.run_dir.name,
        envelope=envelope,
        refinement=refinement,
        shrinks=outcome.shrinks,
        diagnoses=outcome.diagnoses,
        spec=outcome.spec,
        output_root=EXPORTS_DIR,
        urdf_limits=limits,
        benchmark_summary=benchmark_path if benchmark_path.is_file() else None,
    )
    for index, result in enumerate(refinement.tuning_results, 1):
        _print_verification(console, f"tuning {index}", result)
    _print_verification(console, "confirmation", refinement.confirmation)
    console.print(f"verification: {'passed' if refinement.confirmation.passed else 'failed'}")
    console.print(f"verification file: {verification_path}")
    console.print(f"exports: {artifacts.directory}")
    for path in artifacts.files:
        console.print(path)
    return refinement.confirmation.passed


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    console = Console()
    stream = campaign_events(
        args.robot,
        cast(StrategyName, args.strategy),
        seed=args.seed,
        budget=args.budget,
        workers=args.workers,
        stop_after_clusters=args.stop_after_clusters,
        max_diagnosed=args.max_diagnosed,
        run_id=args.run_id,
        runs_root=args.runs_dir,
    )
    try:
        while True:
            event = next(stream)
            _print_event(console, event)
    except StopIteration as stop:
        outcome = stop.value
    except KeyboardInterrupt:
        stream.close()
        console.print("campaign cancelled; completed work was saved", style="yellow")
        return 130
    except (OSError, ValueError) as error:
        console.print(f"campaign could not start: {error}", style="red")
        return 2

    console.print(f"run directory: {outcome.run_dir}")
    campaign_failed = outcome.result.events[-1].data["status"] in {"error", "token_cap"}
    verification_passed: bool | None = None
    if args.verify:
        try:
            verification_passed = _verify_campaign(args, outcome, console)
        except (OSError, RuntimeError, ValueError) as error:
            console.print(f"verification could not run: {error}", style="red")
            return 2
    if campaign_failed:
        return 2
    return 1 if verification_passed is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
