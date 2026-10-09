"""The seven pipeline stages, each runnable on its own."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.download import download_dataset
from src.data.preprocess import build_processed_dataset, load_dataset_metadata, save_dataset
from src.evaluation import aggregate, export
from src.pipeline import report
from src.pipeline.runner import ExperimentData, run_all
from src.splitting.splitter import load_split, make_split, save_split, split_summary
from src.utils.logger import get_logger
from src.utils.paths import ProjectPaths
from src.utils.timer import timed_stage
from src.visualization import (
    bars, confusion, curves, dashboard, distributions, heatmaps, importance, lines, radar, ranking,
    timing,
)
from src.visualization.style import build_context

_log = get_logger(__name__)

STAGES = ("download", "preprocess", "split", "train_eval", "aggregate", "visualize", "report")
TEMPLATE = Path(__file__).parent / "templates" / "report.html.j2"
LIBRARIES = ("numpy", "pandas", "pyarrow", "scikit-learn", "xgboost", "matplotlib", "seaborn")


class StageError(RuntimeError):
    """Raised when a stage cannot run because its inputs are missing."""


@dataclass
class RunOptions:
    """Command-line options shared by all stages."""

    models: list[str]
    splits: list[str]
    sample: float | None = None
    force: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _require(path: Path, stage: str) -> None:
    """Raise a helpful error if an input of a stage is missing.

    Args:
        path: Required file or folder.
        stage: Stage that produces it.

    Raises:
        StageError: If the path does not exist.
    """
    if not path.exists():
        raise StageError(f"Missing {path}. Run the `{stage}` stage first.")


@timed_stage("download")
def stage_download(config: dict[str, Any], paths: ProjectPaths, options: RunOptions) -> None:
    """Fetch the raw Kaggle data if needed.

    Args:
        config: Parsed configuration dictionary.
        paths: Project paths.
        options: CLI options (unused; kept for a uniform stage signature).

    Raises:
        StageError: If the data is still missing afterwards.
    """
    if not download_dataset(paths.raw, config["kaggle"]["competition"], paths.root / "config"):
        raise StageError("Raw data is not available. Follow the manual download steps above.")


@timed_stage("preprocess")
def run_preprocess(config: dict[str, Any], paths: ProjectPaths, options: RunOptions) -> None:
    """Build ``data/processed/dataset.parquet`` unless an identical one exists.

    Args:
        config: Parsed configuration dictionary.
        paths: Project paths.
        options: CLI options (``sample`` and ``force`` are used).
    """
    _require(paths.raw / "train.csv", "download")
    if paths.dataset.is_file() and not options.force:
        existing = load_dataset_metadata(paths.dataset).get("sample_fraction")
        if existing == options.sample:
            _log.info("Processed dataset already up to date, skipping (use --force to rebuild).")
            return
        _log.info("Sample fraction changed (%s -> %s); rebuilding.", existing, options.sample)
    frame, meta = build_processed_dataset(paths.raw, config, options.sample)
    save_dataset(frame, meta, paths.dataset)


def dataset_signature(paths: ProjectPaths, config: dict[str, Any]) -> str:
    """Identify the processed dataset and split mode, to detect stale splits and runs.

    Args:
        paths: Project paths.
        config: Parsed configuration dictionary.

    Returns:
        A short signature string.
    """
    meta = load_dataset_metadata(paths.dataset)
    return f"rows={meta['n_rows']}|sample={meta['sample_fraction']}|mode={config['split']['mode']}" \
           f"|seed={config['seed']}"


@timed_stage("split")
def run_split(config: dict[str, Any], paths: ProjectPaths, options: RunOptions) -> None:
    """Create and save one index file per configured split.

    Args:
        config: Parsed configuration dictionary.
        paths: Project paths.
        options: CLI options (``force`` is used).
    """
    _require(paths.dataset, "preprocess")
    data_cfg = config["data"]
    signature = dataset_signature(paths, config)
    columns = list({data_cfg["target"], data_cfg["id_column"], data_cfg["time_column"]})
    frame = pd.read_parquet(paths.dataset, columns=columns)
    y = frame[data_cfg["target"]].to_numpy()
    summaries = []
    for name, ratio in config["split"]["ratios"].items():
        path = paths.splits / f"{name}.npz"
        if path.is_file() and not options.force and load_split(path)[2] == signature:
            train_idx, test_idx, _ = load_split(path)
        else:
            train_idx, test_idx = make_split(frame, float(ratio), config["split"]["mode"],
                                             config["seed"], data_cfg)
            save_split(path, train_idx, test_idx, signature)
        summaries.append(split_summary(name, y, train_idx, test_idx))
    pd.DataFrame(summaries).to_csv(paths.splits / "split_summary.csv", index=False)


def load_experiment_data(config: dict[str, Any], paths: ProjectPaths) -> ExperimentData:
    """Read the feature matrix and labels of the processed dataset.

    Args:
        config: Parsed configuration dictionary.
        paths: Project paths.

    Returns:
        The dataset as an ``ExperimentData`` bundle.
    """
    meta = load_dataset_metadata(paths.dataset)
    features = meta["feature_columns"]
    target = config["data"]["target"]
    frame = pd.read_parquet(paths.dataset, columns=[*features, target])
    y = frame.pop(target).to_numpy().astype(np.int8)
    _log.info("Feature matrix: %s x %d (~%.0f MB)", f"{len(frame):,}", len(features),
              frame.memory_usage(deep=True).sum() / 1e6)
    return ExperimentData(X=frame, y=y, feature_columns=features)


@timed_stage("train_eval")
def run_experiments(config: dict[str, Any], paths: ProjectPaths, options: RunOptions) -> None:
    """Train and evaluate every selected (model, split) pair.

    Args:
        config: Parsed configuration dictionary.
        paths: Project paths.
        options: CLI options (``models``, ``splits``, ``force``).

    Raises:
        StageError: If any run failed.
    """
    _require(paths.dataset, "preprocess")
    for split in options.splits:
        _require(paths.splits / f"{split}.npz", "split")
    write_run_info(config, paths, options)
    data = load_experiment_data(config, paths)
    runs = [(split, model) for split in options.splits for model in options.models]
    failures = run_all(data, config, paths, runs, options.force)
    if failures:
        raise StageError(f"{len(failures)} run(s) failed: {', '.join(failures)}. See the log.")


def write_run_info(config: dict[str, Any], paths: ProjectPaths, options: RunOptions) -> None:
    """Record library versions and the effective settings in ``outputs/run_info.json``.

    Args:
        config: Parsed configuration dictionary.
        paths: Project paths.
        options: CLI options.
    """
    versions = {}
    for library in LIBRARIES:
        try:
            versions[library] = metadata.version(library)
        except metadata.PackageNotFoundError:
            versions[library] = "not installed"
    info = {
        "python": sys.version.split()[0], "platform": platform.platform(), "libraries": versions,
        "seed": config["seed"], "split_mode": config["split"]["mode"], "sample_fraction": options.sample,
        "models": options.models, "splits": options.splits,
    }
    paths.run_info.write_text(json.dumps(info, indent=2), encoding="utf-8")


def _master(config: dict[str, Any], paths: ProjectPaths) -> pd.DataFrame:
    """Load the master table in config order.

    Args:
        config: Parsed configuration dictionary.
        paths: Project paths.

    Returns:
        The master results table.
    """
    return aggregate.load_master(paths.metrics, list(config["models"]), list(config["split"]["ratios"]))


@timed_stage("aggregate")
def run_aggregate(config: dict[str, Any], paths: ProjectPaths, options: RunOptions) -> None:
    """Build and export every comparison table.

    Args:
        config: Parsed configuration dictionary.
        paths: Project paths.
        options: CLI options (unused; kept for a uniform stage signature).
    """
    master = _master(config, paths)
    summary_path = paths.splits / "split_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.is_file() else None
    tables = aggregate.build_tables(master, summary)
    export.write_all(tables, paths.tables)
    export.print_compact(tables["master_results"])


@timed_stage("visualize")
def run_visualize(config: dict[str, Any], paths: ProjectPaths, options: RunOptions) -> None:
    """Render every figure from the tables and saved predictions.

    Args:
        config: Parsed configuration dictionary.
        paths: Project paths.
        options: CLI options (unused; kept for a uniform stage signature).
    """
    _require(paths.tables / "ranking_table.csv", "aggregate")
    master = _master(config, paths)
    ranks = pd.read_csv(paths.tables / "ranking_table.csv")
    models = [m for m in config["models"] if m in set(master["model"].astype(str))]
    splits = [s for s in config["split"]["ratios"] if s in set(master["split"].astype(str))]
    ctx = build_context(config, paths.figures, models, splits)
    largest = max(splits, key=lambda s: ctx.train_pct[s])
    dashboard.plot_dashboard(master, ranks, ctx)
    heatmaps.plot_metric_heatmaps(master, ctx)
    bars.plot_grouped_bars(master, ctx)
    lines.plot_learning_lines(master, ctx)
    radar.plot_radars(master, ctx)
    curves.plot_curve_grid(ctx, paths.predictions, "roc")
    curves.plot_curve_grid(ctx, paths.predictions, "pr")
    curves.plot_curve_overlay(ctx, paths.predictions, largest)
    confusion.plot_confusion_grid(ctx, paths.predictions)
    importance.plot_importance_grid(ctx, paths.importances)
    timing.plot_efficiency(master, ctx)
    distributions.plot_score_distributions(ctx, paths.predictions, largest, config["seed"])
    ranking.plot_ranking(ranks, ctx)
    dashboard.write_interactive_dashboard(master, ctx, paths.figures / "interactive_dashboard.html")
    _log.info("Saved %d figure files to %s", len(ctx.saved), paths.figures)


@timed_stage("report")
def run_report(config: dict[str, Any], paths: ProjectPaths, options: RunOptions) -> None:
    """Write ``outputs/REPORT.html``.

    Args:
        config: Parsed configuration dictionary.
        paths: Project paths.
        options: CLI options (unused; kept for a uniform stage signature).
    """
    _require(paths.tables / "master_results.csv", "aggregate")
    _require(paths.figures / "00_dashboard.png", "visualize")
    tables = aggregate.build_tables(_master(config, paths), pd.read_csv(paths.splits / "split_summary.csv"))
    context = report.build_context(tables, config, load_dataset_metadata(paths.dataset), paths.figures)
    report.render_report(context, TEMPLATE, paths.report)
    _log.info("Report written to %s (%.1f MB)", paths.report, paths.report.stat().st_size / 1e6)


STAGE_FUNCTIONS = {
    "download": stage_download,
    "preprocess": run_preprocess,
    "split": run_split,
    "train_eval": run_experiments,
    "aggregate": run_aggregate,
    "visualize": run_visualize,
    "report": run_report,
}
