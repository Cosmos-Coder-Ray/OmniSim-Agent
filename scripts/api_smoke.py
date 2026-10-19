from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx

from omnisim.api.smoke import response_leaks

ROOT = Path(__file__).resolve().parents[1]


def _env_file_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return values
    for line in lines:
        name, separator, value = line.partition("=")
        if not separator or name not in {"NEBIUS_API_KEY", "TAVILY_API_KEY"}:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[name] = value
    return values


def _secrets(env_file: Path) -> dict[str, str]:
    file_values = _env_file_values(env_file)
    return {
        name: value
        for name in ("NEBIUS_API_KEY", "TAVILY_API_KEY")
        if (value := os.environ.get(name) or file_values.get(name, "")).strip()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    args = parser.parse_args()
    secrets = _secrets(args.env_file)
    checks = (
        ("/api/health", {200}),
        ("/openapi.json", {200}),
        ("/api/benchmark", {200, 404}),
        ("/api/campaigns/not-a-campaign", {404}),
    )
    failed = False
    try:
        with httpx.Client(base_url=args.base_url, timeout=5.0) as client:
            for path, statuses in checks:
                response = client.get(path)
                leaks = response_leaks(response.content, response.headers, secrets)
                if leaks:
                    failed = True
                    for name, location in leaks:
                        print(f"secret leak: {name} appeared in {location} for {path}")
                if response.status_code not in statuses:
                    failed = True
                    print(f"GET {path} returned HTTP {response.status_code}")
                elif not leaks:
                    print(f"GET {path} returned HTTP {response.status_code}")
    except httpx.RequestError as exc:
        print(f"API smoke request failed: {type(exc).__name__}")
        return 1
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
