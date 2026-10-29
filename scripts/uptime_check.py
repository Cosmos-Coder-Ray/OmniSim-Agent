from __future__ import annotations

import argparse
import json
import sys
from urllib.error import URLError
from urllib.request import urlopen


def check_health(base_url: str, check_demo: bool = False) -> bool:
    url = f"{base_url.rstrip('/')}/api/health"
    try:
        with urlopen(url, timeout=10) as response:
            if response.status != 200:
                print(f"health returned HTTP {response.status}")
                return False
            health = json.loads(response.read())
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"health check failed: {exc}")
        return False
    if (
        not isinstance(health, dict)
        or "token_usage" not in health
        or "knowledge_usage" not in health
    ):
        print("health response is missing usage fields")
        return False
    if check_demo and not health.get("demo_ready") and not health.get("model_resolution"):
        print("health response has no ready demo or resolved models")
        return False
    print("health check passed")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--check-demo", action="store_true")
    args = parser.parse_args()
    return 0 if check_health(args.base_url, args.check_demo) else 1


if __name__ == "__main__":
    sys.exit(main())
