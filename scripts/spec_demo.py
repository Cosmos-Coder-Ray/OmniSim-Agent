import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from omnisim.knowledge import (
    build_spec,
    compare_with_urdf,
    get_knowledge_usage,
    map_failures_to_real_errors,
)
from omnisim.knowledge.cache import CACHE_DIR
from omnisim.schemas import FailureKind

KINDS: tuple[FailureKind, ...] = (
    "torque_saturation",
    "joint_limit",
    "velocity_limit",
    "tracking_error",
    "singularity",
    "collision",
    "instability",
)


def _cached_dates() -> dict[str, str]:
    dates: dict[str, str] = {}
    if not CACHE_DIR.exists():
        return dates
    for path in CACHE_DIR.glob("*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            op, fetched = row["op"], row["fetched_at"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
        dates[op] = max(dates.get(op, ""), fetched)
    return dates


def _print_spec(robot: str) -> None:
    console = Console()
    spec = build_spec(robot)
    table = Table(title=f"{spec.robot} documented limits")
    table.add_column("Status")
    table.add_column("Claim")
    table.add_column("Value")
    table.add_column("Source")
    limits = (
        *spec.joint_torque_limits_nm,
        *spec.joint_velocity_limits,
        *spec.joint_position_limits,
    )
    for ev in limits:
        status = "✓" if ev.verified else "×"
        table.add_row(status, ev.claim, f"{ev.value} {ev.unit or ''}".strip(), ev.source_url)
    console.print(table)

    coverage = Table(title="Verified joint coverage")
    coverage.add_column("Quantity")
    coverage.add_column("Joints")
    for name, evidence in (
        ("torque", spec.joint_torque_limits_nm),
        ("velocity", spec.joint_velocity_limits),
        ("lower", spec.joint_position_limits),
        ("upper", spec.joint_position_limits),
    ):
        joints = {
            ev.claim.split()[1]
            for ev in evidence
            if ev.verified and len(ev.claim.split()) == 3 and ev.claim.endswith(f" {name}")
        }
        coverage.add_row(name, f"{len(joints)}/7")
    console.print(coverage)

    mismatches, not_compared = compare_with_urdf(spec)
    mismatch_table = Table(title="URDF comparison")
    for name in ("Joint", "Field", "URDF", "Documentation", "Sim more permissive"):
        mismatch_table.add_column(name)
    for item in mismatches:
        mismatch_table.add_row(
            str(item.joint),
            item.field,
            f"{item.urdf:g}",
            f"{item.doc:g}",
            str(item.sim_more_permissive),
        )
    console.print(mismatch_table)
    if not_compared:
        console.print("Not compared: " + ", ".join(ev.claim for ev in not_compared))

    mappings = map_failures_to_real_errors(list(KINDS), spec)
    mapping_table = Table(title="Documented error analogues (advisory)")
    mapping_table.add_column("Failure kind")
    mapping_table.add_column("Documented error")
    mapping_table.add_column("Confidence")
    mapping_table.add_column("Evidence")
    for kind in KINDS:
        item = mappings[kind]
        mapping_table.add_row(
            kind,
            item.real_error_name or "no match",
            item.confidence or "",
            item.evidence.source_url if item.evidence else "",
        )
    console.print(mapping_table)
    usage = get_knowledge_usage()
    dates = _cached_dates()
    console.print(
        "Knowledge usage today (UTC): "
        f"searches={usage['searches']} extracts={usage['extracts']} "
        f"cache_hits={usage['cache_hits']} cache_misses={usage['cache_misses']}"
    )
    if dates:
        console.print(
            "Latest cached results: "
            + ", ".join(f"{op} {when}" for op, when in sorted(dates.items()))
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot")
    args = parser.parse_args()
    try:
        _print_spec(args.robot)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
