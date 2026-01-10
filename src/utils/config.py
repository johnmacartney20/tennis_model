from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    project_root: Path

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models" / "trained"


def get_paths() -> Paths:
    project_root = Path(__file__).resolve().parents[2]
    return Paths(project_root=project_root)
