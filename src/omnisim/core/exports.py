from __future__ import annotations

import html
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from omnisim.core.shield import SafetyShield
from omnisim.core.shrinker import ShrinkResult
from omnisim.core.verify import RefinementResult
from omnisim.schemas import (
    BenchmarkResult,
    DiagnosisReport,
    HardwareSpec,
    SafeEnvelope,
)
from omnisim.sim.limits import UrdfLimits, load_urdf_limits

ROOT = Path(__file__).resolve().parents[3]
EXPORTS_DIR = ROOT / "exports"
_JOINT_CLAIM = re.compile(r"^joint\s+([1-7])\s+(torque|velocity|lower|upper)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ExportArtifacts:
    directory: Path
    files: tuple[Path, ...]


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("export values must be finite")
    return str(round(value, 6))


def _joint_values(
    spec: HardwareSpec, limits: UrdfLimits
) -> dict[str, tuple[list[float], list[bool]]]:
    values = {
        "torque": list(map(float, limits.effort)),
        "velocity": list(map(float, limits.velocity)),
        "lower": list(map(float, limits.lower)),
        "upper": list(map(float, limits.upper)),
    }
    verified = {name: [False] * 7 for name in values}
    for entries in (
        spec.joint_torque_limits_nm,
        spec.joint_velocity_limits,
        spec.joint_position_limits,
    ):
        for item in entries:
            if (
                not item.verified
                or isinstance(item.value, bool)
                or not isinstance(item.value, (int, float))
            ):
                continue
            match = _JOINT_CLAIM.fullmatch(item.claim.strip())
            if match is None:
                continue
            joint = int(match.group(1)) - 1
            name = match.group(2).lower()
            value = float(item.value)
            if not math.isfinite(value):
                continue
            values[name][joint] = value
            verified[name][joint] = True
    return {name: (values[name], verified[name]) for name in values}


def _render_yaml(envelope: SafeEnvelope, spec: HardwareSpec, limits: UrdfLimits) -> str:
    joints = _joint_values(spec, limits)
    lines = [
        "/**:",
        "  ros__parameters:",
        f"    robot: {json.dumps(envelope.robot)}",
        f"    max_payload_kg: {_number(float(envelope.limits['max_payload_kg']))}",
        f"    max_latency_ms: {_number(float(envelope.limits['max_latency_ms']))}",
        f"    torque_utilization_cap: {_number(float(envelope.limits['torque_utilization_cap']))}",
        f"    kp_scale_max: {_number(float(envelope.controller['kp_scale_max']))}  # scale factor",
    ]
    for name, label in (
        ("torque", "joint_torque_limits_nm"),
        ("velocity", "joint_velocity_limits_rad_s"),
        ("lower", "joint_position_lower_rad"),
        ("upper", "joint_position_upper_rad"),
    ):
        values, verified = joints[name]
        missing = [str(index + 1) for index, available in enumerate(verified) if not available]
        if missing:
            lines.append(
                f"    # Unverified joints {', '.join(missing)} use the Panda URDF values from data/urdf_limits.json."
            )
        lines.append(f"    {label}: [{', '.join(_number(value) for value in values)}]")
    lines.extend(
        [
            "    speed_scale_max_by_payload_kg:",
            *[
                f"      - [{_number(float(payload))}, {_number(float(speed))}]"
                for payload, speed in envelope.limits["speed_scale_max_by_payload_kg"]
            ],
            "    speed_scale_max_by_latency_ticks:",
            *[
                f"      - {_number(float(speed))}"
                for speed in envelope.limits["speed_scale_max_by_latency_ticks"]
            ],
            "",
        ]
    )
    return "\n".join(lines)


def _safe_link(url: str) -> str | None:
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    return url


def _report_html(
    envelope: SafeEnvelope,
    refinement: RefinementResult,
    diagnoses: dict[str, DiagnosisReport],
    benchmark: BenchmarkResult | None,
) -> str:
    rows = []
    for evidence in envelope.evidence:
        source = _safe_link(evidence.source_url)
        source_html = (
            f'<a href="{html.escape(source, quote=True)}" rel="noopener noreferrer">source</a>'
            if source is not None
            else html.escape(evidence.source_url)
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(evidence.claim)}</td>"
            f"<td>{html.escape(str(evidence.value))} {html.escape(evidence.unit or '')}</td>"
            f"<td>{html.escape(evidence.quote)}</td>"
            f"<td>{source_html}</td>"
            "</tr>"
        )
    diagnosis_items = "".join(
        f"<li><strong>{html.escape(cluster_id)}</strong>: "
        f"{html.escape(report.root_cause)} {html.escape(report.mechanism)}</li>"
        for cluster_id, report in sorted(diagnoses.items())
    )
    envelope_json = html.escape(
        json.dumps(envelope.model_dump(mode="json"), sort_keys=True, indent=2, allow_nan=False)
    )
    confirmation = refinement.confirmation
    benchmark_block = ""
    if benchmark is not None:
        benchmark_json = html.escape(
            json.dumps(benchmark.model_dump(mode="json"), sort_keys=True, indent=2, allow_nan=False)
        )
        benchmark_block = (
            "<section><h2>Recorded benchmark artifact</h2>"
            f"<p>Source label: {html.escape(benchmark.source)}. This is a saved benchmark summary.</p>"
            f"<pre>{benchmark_json}</pre></section>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(envelope.robot)} safe envelope</title>
<style>
body {{ max-width: 960px; margin: 2rem auto; padding: 0 1rem; font: 16px/1.5 system-ui, sans-serif; color: #17212b; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccd4dc; padding: .5rem; text-align: left; vertical-align: top; }}
pre {{ overflow: auto; padding: 1rem; background: #f2f5f7; }}
.caveat {{ border-left: 4px solid #a15d00; padding: .6rem 1rem; background: #fff6e8; }}
</style>
</head>
<body>
<h1>{html.escape(envelope.robot)} safe operating envelope</h1>
<p>Version {html.escape(envelope.version)}. Confirmation set: {confirmation.n_scenarios} scenarios, seed root {confirmation.seed}.</p>
<h2>Verification</h2>
<ul>
<li>OFF failure rate: {confirmation.fail_rate_before:.3%}</li>
<li>ON failure rate: {confirmation.fail_rate_after:.3%}</li>
<li>Blocked: {confirmation.blocked_count}/{confirmation.n_scenarios} ({confirmation.blocked_rate:.3%})</li>
<li>Completion: {confirmation.completion_count}/{confirmation.n_scenarios} ({confirmation.completion_rate:.3%})</li>
<li>Confirmation passed provisional criteria: {str(confirmation.passed).lower()}</li>
<li>Discovery overlaps within normalized L-infinity epsilon {confirmation.overlap_epsilon}: {confirmation.overlap_count}</li>
</ul>
<p class="caveat">Simulation-only result. It is not hardware certification or permission to operate a robot.</p>
<p class="caveat">The torque table estimates inverse dynamics along the reference trajectory. It omits closed-loop transients and unobservable link mass scaling.</p>
<h2>Envelope</h2>
<pre>{envelope_json}</pre>
<h2>Diagnosis</h2>
<ul>{diagnosis_items}</ul>
<h2>Verified evidence</h2>
<table><thead><tr><th>Claim</th><th>Value</th><th>Quote</th><th>Source</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
{benchmark_block}
</body>
</html>
"""


def _regression_test() -> str:
    return """from __future__ import annotations

import json
from pathlib import Path

from omnisim.core.shield import SafetyShield
from omnisim.schemas import SafeEnvelope, ScenarioParams


def test_shield_blocks_saved_failure_repros() -> None:
    root = Path(__file__).parent
    envelope = SafeEnvelope.model_validate_json((root / "safe_envelope.json").read_text(encoding="utf-8"))
    repros = json.loads((root / "repros.json").read_text(encoding="utf-8"))
    for repro in repros:
        params = ScenarioParams.model_validate(repro["params"])
        decision = SafetyShield(envelope).decide(params)
        assert decision.rejected is repro["shield"]["rejected"], repro["cluster_id"]
        assert list(decision.reasons) == repro["shield"]["reasons"]
        assert decision.params.model_dump(mode="json") == repro["shield"]["params"]
"""


def export_campaign(
    run_id: str,
    *,
    envelope: SafeEnvelope,
    refinement: RefinementResult,
    shrinks: dict[str, ShrinkResult],
    diagnoses: dict[str, DiagnosisReport],
    spec: HardwareSpec,
    output_root: Path = EXPORTS_DIR,
    urdf_limits: UrdfLimits | None = None,
    benchmark_summary: Path | None = None,
) -> ExportArtifacts:
    if run_id in {"", ".", ".."} or Path(run_id).name != run_id:
        raise ValueError("run_id must be a single path component")
    output_root.mkdir(parents=True, exist_ok=True)
    root = output_root.resolve()
    target = root / run_id
    if target.is_symlink() or target.resolve().parent != root:
        raise ValueError("export directory must remain under the export root")
    target.mkdir(exist_ok=True)
    active_limits = urdf_limits or load_urdf_limits()
    shield = SafetyShield(envelope)
    repros = []
    for cluster_id, shrink_result in sorted(shrinks.items()):
        params = shrink_result.minimal
        decision = shield.decide(params)
        repros.append(
            {
                "cluster_id": cluster_id,
                "params": params.model_dump(mode="json"),
                "converged": shrink_result.converged,
                "sims_used": shrink_result.sims_used,
                "shield": {
                    "rejected": decision.rejected,
                    "reasons": list(decision.reasons),
                    "params": decision.params.model_dump(mode="json"),
                },
            }
        )
    files = {
        "safe_envelope.json": _json_bytes(envelope.model_dump(mode="json")),
        "repros.json": _json_bytes(repros),
        "ros2_parameters.yaml": _render_yaml(envelope, spec, active_limits).encode("utf-8"),
        "test_safe_envelope.py": _regression_test().encode("utf-8"),
        "index.html": _report_html(envelope, refinement, diagnoses, None).encode("utf-8"),
        "verification.json": _json_bytes(
            {
                "tuning_seed": refinement.tuning_seed,
                "confirmation_seed": refinement.confirmation_seed,
                "tuning": [result.to_dict() for result in refinement.tuning_results],
                "confirmation": refinement.confirmation.to_dict(),
            }
        ),
    }
    benchmark: BenchmarkResult | None = None
    has_benchmark = benchmark_summary is not None and benchmark_summary.is_file()
    if has_benchmark:
        assert benchmark_summary is not None
        raw_benchmark = benchmark_summary.read_text(encoding="utf-8")
        benchmark = BenchmarkResult.model_validate_json(raw_benchmark)
        shutil.copy2(benchmark_summary, target / "recorded_benchmark.json")
        files["index.html"] = _report_html(envelope, refinement, diagnoses, benchmark).encode(
            "utf-8"
        )
    for name, content in files.items():
        (target / name).write_bytes(content)
    names = sorted(files)
    if has_benchmark:
        names.append("recorded_benchmark.json")
    return ExportArtifacts(target, tuple(target / name for name in names))
