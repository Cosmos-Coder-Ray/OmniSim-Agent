from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from rich.console import Console

from omnisim.core.campaign_verify import run_campaign_verification
from omnisim.core.orchestrator import RUNS_DIR, campaign_events
from omnisim.schemas import CampaignEvent, StrategyName


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
    verification = run_campaign_verification(
        outcome,
        robot=getattr(args, "robot", "franka_panda"),
        seed=getattr(args, "seed", 17),
        workers=getattr(args, "workers", 1),
        domain_path=getattr(args, "domain", None),
    )
    if verification.skipped_reason is not None:
        console.print(f"verification skipped: {verification.skipped_reason}", style="yellow")
        return None
    refinement = verification.refinement
    if refinement is None:
        return verification.passed
    for index, result in enumerate(refinement.tuning_results, 1):
        _print_verification(console, f"tuning {index}", result)
    _print_verification(console, "confirmation", refinement.confirmation)
    console.print(f"verification: {'passed' if refinement.confirmation.passed else 'failed'}")
    if verification.verification_path is not None:
        console.print(f"verification file: {verification.verification_path}")
    if verification.artifacts is not None:
        console.print(f"exports: {verification.artifacts.directory}")
        for path in verification.artifacts.files:
            console.print(path)
    return verification.passed


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
