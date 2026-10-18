from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from omnisim.core.orchestrator import RUNS_DIR
from omnisim.schemas import CampaignResult, Replay

ROOT = Path(__file__).resolve().parents[3]
DEMO_RUNS_DIR = ROOT / "data" / "demo_runs"
_ARTIFACTS = (
    "nominal.json",
    "episodes.jsonl",
    "events.jsonl",
    "spec.json",
    "clusters.json",
    "shrinks.json",
    "diagnoses.json",
)


@dataclass(frozen=True, slots=True)
class SavedDemoRun:
    path: Path
    files: tuple[tuple[str, int], ...]


def _component(value: str, label: str) -> str:
    if value in {"", ".", ".."} or Path(value).name != value:
        raise ValueError(f"{label} must be a single path component")
    return value


def _relative_trajectory(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"trajectory path must be run-relative: {value}")
    return path


def save_demo_run(
    run_id: str,
    *,
    name: str = "hero",
    force: bool = False,
    runs_root: Path = RUNS_DIR,
    demo_root: Path = DEMO_RUNS_DIR,
) -> SavedDemoRun:
    run_id = _component(run_id, "run_id")
    name = _component(name, "name")
    resolved_runs = runs_root.resolve()
    source_dir = resolved_runs / run_id
    if source_dir.is_symlink() or source_dir.resolve().parent != resolved_runs:
        raise ValueError("run directory must remain under the runs directory")
    demo_root.mkdir(parents=True, exist_ok=True)
    resolved_demo = demo_root.resolve()
    destination = resolved_demo / name
    if destination.parent != resolved_demo:
        raise ValueError("demo destination must remain under the demo runs directory")
    if destination.is_symlink():
        raise ValueError("demo destination cannot be a symlink")
    if destination.exists() and not force:
        raise FileExistsError(f"demo run already exists: {destination}")

    campaign_path = source_dir / "campaign.json"
    campaign = CampaignResult.model_validate_json(campaign_path.read_text(encoding="utf-8"))
    trajectory_dir = source_dir / "trajectories"
    replays: dict[str, Replay] = {}
    for path in sorted(trajectory_dir.rglob("*.json")):
        replay = Replay.model_validate_json(path.read_text(encoding="utf-8"))
        relative = path.relative_to(trajectory_dir).as_posix()
        replays[relative] = replay.model_copy(update={"source": "recorded"})

    for episode in campaign.episodes:
        if episode.trajectory_path is None:
            continue
        relative = _relative_trajectory(episode.trajectory_path)
        if not relative.parts or relative.parts[0] != "trajectories":
            raise ValueError(
                f"trajectory path must be under trajectories/: {episode.trajectory_path}"
            )
        replay_path = source_dir / relative
        replay = Replay.model_validate_json(replay_path.read_text(encoding="utf-8"))
        if replay.episode_id != episode.episode_id:
            raise ValueError(f"trajectory id does not match episode {episode.episode_id}")
        replay_name = Path(*relative.parts[1:]).as_posix()
        replays[replay_name] = replay.model_copy(update={"source": "recorded"})

    recorded_campaign = CampaignResult.model_validate(
        campaign.model_copy(update={"source": "recorded"}).model_dump(mode="python")
    )
    staging = resolved_demo / f".{name}.tmp-{uuid4().hex}"
    backup = resolved_demo / f".{name}.old-{uuid4().hex}"
    staging.mkdir()
    try:
        (staging / "trajectories").mkdir()
        (staging / "campaign.json").write_text(
            recorded_campaign.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        for filename, replay in replays.items():
            output = staging / "trajectories" / Path(filename)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(replay.model_dump_json(indent=2) + "\n", encoding="utf-8")
        for filename in _ARTIFACTS:
            source = source_dir / filename
            if source.is_file():
                shutil.copy2(source, staging / filename)

        if destination.exists():
            destination.replace(backup)
        try:
            staging.replace(destination)
        except OSError:
            if backup.exists() and not destination.exists():
                backup.replace(destination)
            raise
        if backup.exists():
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    files = tuple(
        (path.relative_to(destination).as_posix(), path.stat().st_size)
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    )
    return SavedDemoRun(destination, files)
