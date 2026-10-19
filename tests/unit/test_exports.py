from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from omnisim.core.exports import export_campaign
from omnisim.core.shrinker import ShrinkResult
from omnisim.core.verify import RefinementResult, VerificationResult
from omnisim.schemas import Evidence, HardwareSpec, SafeEnvelope, ScenarioParams
from omnisim.sim.limits import UrdfLimits


def _limits() -> UrdfLimits:
    return UrdfLimits(
        effort=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]),
        velocity=np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]),
        lower=np.array([-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671]),
        upper=np.array([2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671]),
    )


def _envelope(evidence: list[Evidence] | None = None) -> SafeEnvelope:
    return SafeEnvelope(
        robot="franka_panda",
        version="1",
        limits={
            "max_payload_kg": 2.0,
            "max_latency_ms": 100.0,
            "torque_utilization_cap": 0.95,
            "speed_scale_max_by_payload_kg": [[0.0, 2.0], [2.0, 1.0]],
            "speed_scale_max_by_latency_ticks": [2.0, 1.9, 1.8],
        },
        controller={"kp_scale_max": 1.4},
        domain_randomization_bounds={},
        evidence=evidence or [],
        verification={"passed": True},
    )


def _refinement() -> RefinementResult:
    result = VerificationResult(
        n_scenarios=8,
        seed=120,
        scenario_seeds=tuple(range(20, 28)),
        fail_rate_before=0.25,
        fail_rate_after=0.0,
        blocked_count=1,
        blocked_rate=0.125,
        executed_count=7,
        completion_count=7,
        completion_rate=0.875,
        mean_relative_cycle_time_increase=0.1,
        per_kind={
            "tracking_error": {"off_count": 2, "off_rate": 0.25, "on_count": 0, "on_rate": 0.0}
        },
        overlap_count=1,
        overlap_rate=0.125,
        overlap_epsilon=0.01,
        infrastructure_errors=0,
        passed=True,
        wall_time_s=1.0,
        failing_scenarios=(),
    )
    return RefinementResult(_envelope(), (result,), result, 120, 200)


def _spec(evidence: list[Evidence] | None = None) -> HardwareSpec:
    return HardwareSpec(
        robot="franka_panda",
        joint_torque_limits_nm=evidence or [],
        joint_velocity_limits=[],
        joint_position_limits=[],
        error_taxonomy=[],
    )


def test_exports_are_byte_deterministic_and_yaml_uses_urdf_fallback_comments(
    tmp_path: Path,
) -> None:
    root = tmp_path / "exports"
    shrinks = {"cluster.1": ShrinkResult(ScenarioParams(payload_kg=2.0, seed=27), 5, True, [])}
    kwargs = {
        "run_id": "run_1",
        "envelope": _envelope(),
        "refinement": _refinement(),
        "shrinks": shrinks,
        "diagnoses": {},
        "spec": _spec(),
        "output_root": root,
        "urdf_limits": _limits(),
    }
    first = export_campaign(**kwargs)
    first_bytes = {path.name: path.read_bytes() for path in first.files}
    second = export_campaign(**kwargs)
    second_bytes = {path.name: path.read_bytes() for path in second.files}

    assert first_bytes == second_bytes
    yaml_text = (first.directory / "ros2_parameters.yaml").read_text(encoding="utf-8")
    assert (
        yaml_text
        == """/**:
  ros__parameters:
    robot: "franka_panda"
    max_payload_kg: 2.0
    max_latency_ms: 100.0
    torque_utilization_cap: 0.95
    kp_scale_max: 1.4  # scale factor
    # Unverified joints 1, 2, 3, 4, 5, 6, 7 use the Panda URDF values from data/urdf_limits.json.
    joint_torque_limits_nm: [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]
    # Unverified joints 1, 2, 3, 4, 5, 6, 7 use the Panda URDF values from data/urdf_limits.json.
    joint_velocity_limits_rad_s: [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]
    # Unverified joints 1, 2, 3, 4, 5, 6, 7 use the Panda URDF values from data/urdf_limits.json.
    joint_position_lower_rad: [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671]
    # Unverified joints 1, 2, 3, 4, 5, 6, 7 use the Panda URDF values from data/urdf_limits.json.
    joint_position_upper_rad: [2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671]
    speed_scale_max_by_payload_kg:
      - [0.0, 2.0]
      - [2.0, 1.0]
    speed_scale_max_by_latency_ticks:
      - 2.0
      - 1.9
      - 1.8
"""
    )
    assert (first.directory / "safe_envelope.json").read_bytes().endswith(b"\n")
    assert "minimal_params" not in (first.directory / "repros.json").read_text(encoding="utf-8")


def test_report_escapes_text_and_only_links_http_evidence(tmp_path: Path) -> None:
    unsafe = Evidence(
        claim="<script>alert('claim')</script>",
        value="<b>value</b>",
        source_url="javascript:alert(1)",
        quote="</td><script>alert('quote')</script>",
        verified=True,
    )
    safe = Evidence(
        claim="joint 1 torque",
        value=87.0,
        unit="Nm",
        source_url='https://manufacturer.example/spec?x="y"',
        quote="rated torque is 87 Nm",
        verified=True,
    )
    artifact = export_campaign(
        "report",
        envelope=_envelope([unsafe, safe]),
        refinement=_refinement(),
        shrinks={},
        diagnoses={},
        spec=_spec([unsafe, safe]),
        output_root=tmp_path,
        urdf_limits=_limits(),
    )
    document = (artifact.directory / "index.html").read_text(encoding="utf-8")

    assert "<script>alert('claim')</script>" not in document
    assert "&lt;script&gt;alert(&#x27;claim&#x27;)&lt;/script&gt;" in document
    assert 'href="javascript:' not in document
    assert 'rel="noopener noreferrer"' in document
    assert 'src="http' not in document
    assert not re.search(r"<(?:script|img|link)[^>]+(?:src|href)=\"https?://", document)
    assert "Simulation-only result" in document
    assert "omits closed-loop transients" in document


def test_generated_regression_file_compiles_and_loads_sibling_artifacts(tmp_path: Path) -> None:
    artifact = export_campaign(
        "compile",
        envelope=_envelope(),
        refinement=_refinement(),
        shrinks={"cluster.1": ShrinkResult(ScenarioParams(payload_kg=2.0), 2, True, [])},
        diagnoses={},
        spec=_spec(),
        output_root=tmp_path,
        urdf_limits=_limits(),
    )
    source = (artifact.directory / "test_safe_envelope.py").read_text(encoding="utf-8")

    compile(source, str(artifact.directory / "test_safe_envelope.py"), "exec")
    assert "Path(__file__).parent" in source
    assert "SafetyShield(envelope).decide(params)" in source
    assert "safe_envelope.json" in source
    assert "repros.json" in source


def test_existing_benchmark_is_exported_as_a_labeled_recorded_artifact(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        '{"budget":1,"seed_count":1,"source":"live","strategies":[]}',
        encoding="utf-8",
    )
    artifact = export_campaign(
        "with-benchmark",
        envelope=_envelope(),
        refinement=_refinement(),
        shrinks={},
        diagnoses={},
        spec=_spec(),
        output_root=tmp_path / "exports",
        urdf_limits=_limits(),
        benchmark_summary=benchmark,
    )

    copied = artifact.directory / "recorded_benchmark.json"
    assert copied.read_bytes() == benchmark.read_bytes()
    assert copied in artifact.files
    report = (artifact.directory / "index.html").read_text(encoding="utf-8")
    assert "Recorded benchmark artifact" in report
    assert "Source label: live" in report
