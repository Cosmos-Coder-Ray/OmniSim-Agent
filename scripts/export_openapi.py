from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omnisim.api.main import create_app
from omnisim.config import Settings

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "web" / "src" / "api" / "openapi.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(create_app(Settings()).openapi(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != content:
            print("OpenAPI export is stale; run scripts/export_openapi.py")
            return 1
        print("OpenAPI export is current")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(content, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
