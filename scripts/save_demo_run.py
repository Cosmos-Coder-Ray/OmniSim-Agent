from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console

from omnisim.core.demo_run import DEMO_RUNS_DIR, save_demo_run
from omnisim.core.orchestrator import RUNS_DIR
from omnisim.llm.call_log import provider_hosts_for_run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Save a campaign as a recorded demo run")
    parser.add_argument("run_id")
    parser.add_argument("--name", default="hero")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--demo-dir", type=Path, default=DEMO_RUNS_DIR)
    args = parser.parse_args(argv)

    try:
        saved = save_demo_run(
            args.run_id,
            name=args.name,
            force=args.force,
            runs_root=args.runs_dir,
            demo_root=args.demo_dir,
        )
    except (OSError, ValueError) as error:
        print(f"could not save demo run: {error}", file=sys.stderr)
        return 2

    console = Console()
    console.print(f"saved recorded run: {saved.path}")
    hosts = provider_hosts_for_run(args.run_id)
    console.print(f"[bold]provider host(s): {', '.join(hosts) if hosts else 'not recorded'}[/bold]")
    for filename, size in saved.files:
        console.print(f"{filename}  {size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
