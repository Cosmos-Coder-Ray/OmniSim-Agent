from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class ApiPaths:
    runs: Path
    exports: Path
    demo_runs: Path
    data: Path
    database: Path
    control: Path

    @classmethod
    def from_root(cls, root: Path = ROOT) -> "ApiPaths":
        root = root.resolve()
        data = root / "data"
        return cls(
            runs=root / "runs",
            exports=root / "exports",
            demo_runs=data / "demo_runs",
            data=data,
            database=data / "api.sqlite",
            control=data / "api_jobs",
        )
