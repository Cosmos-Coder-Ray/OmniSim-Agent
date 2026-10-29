from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")
    for index in range(args.runs):
        print(f"E2E fake tier run {index + 1}/{args.runs}", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests/e2e", "-m", "not slow"],
            check=False,
        )
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
