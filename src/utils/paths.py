"""Central path resolution. Every module asks this one for locations on disk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.config_loader import PROJECT_ROOT


@dataclass(frozen=True)
class ProjectPaths:
    """All input and output locations used by the pipeline."""

    root: Path
    raw: Path
    processed: Path
    splits: Path
    outputs: Path
    models: Path
    predictions: Path
    metrics: Path
    importances: Path
    tables: Path
    figures: Path
    logs: Path

    @property
    def dataset(self) -> Path:
        """Path of the processed dataset parquet file."""
        return self.processed / "dataset.parquet"

    @property
    def report(self) -> Path:
        """Path of the final single-page HTML report."""
        return self.outputs / "REPORT.html"

    @property
    def run_info(self) -> Path:
        """Path of the JSON file recording library versions and settings."""
        return self.outputs / "run_info.json"

    def ensure(self) -> ProjectPaths:
        """Create every directory that does not exist yet.

        Returns:
            The same instance, for chaining.
        """
        for value in self.__dict__.values():
            if isinstance(value, Path) and value != self.root:
                value.mkdir(parents=True, exist_ok=True)
        return self


def build_paths(config: dict[str, Any], root: Path | None = None) -> ProjectPaths:
    """Resolve all paths from the config, relative to the project root.

    Args:
        config: Parsed configuration dictionary.
        root: Optional override of the project root (used by tests).

    Returns:
        A ``ProjectPaths`` instance with all directories created.
    """
    base = Path(root) if root else PROJECT_ROOT
    processed = base / config["paths"]["processed_dir"]
    outputs = base / config["paths"]["outputs_dir"]
    return ProjectPaths(
        root=base,
        raw=base / config["paths"]["raw_dir"],
        processed=processed,
        splits=processed / "splits",
        outputs=outputs,
        models=outputs / "models",
        predictions=outputs / "predictions",
        metrics=outputs / "metrics",
        importances=outputs / "importances",
        tables=outputs / "tables",
        figures=outputs / "figures",
        logs=outputs / "logs",
    ).ensure()
