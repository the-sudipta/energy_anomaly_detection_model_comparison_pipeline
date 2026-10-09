"""End-to-end smoke test on a tiny synthetic dataset (no network needed)."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline import stages
from src.pipeline.stages import RunOptions
from src.utils.config_loader import load_config
from src.utils.logger import setup_logging
from src.utils.paths import build_paths

N_BUILDINGS = 10
N_HOURS = 200
ANOMALY_RATE = 0.03


def make_synthetic_raw(raw_dir: Path, seed: int = 0) -> None:
    """Write LEAD-shaped train, metadata and weather CSVs (~2,000 rows, ~3% anomalies).

    Args:
        raw_dir: Destination folder.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    times = pd.date_range("2016-01-01", periods=N_HOURS, freq="h")
    buildings = np.repeat(np.arange(N_BUILDINGS), N_HOURS)
    stamps = pd.DatetimeIndex(np.tile(times, N_BUILDINGS))
    base = 50 + 10 * np.sin(2 * np.pi * stamps.hour / 24) + rng.normal(0, 3, len(stamps))
    anomaly = (rng.random(len(stamps)) < ANOMALY_RATE).astype(int)
    reading = np.where(anomaly == 1, base * rng.choice([0.0, 3.0], len(stamps)), base)
    reading[rng.random(len(stamps)) < 0.02] = np.nan
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"building_id": buildings, "timestamp": stamps, "meter_reading": reading,
                  "anomaly": anomaly}).to_csv(raw_dir / "train.csv", index=False)
    pd.DataFrame({
        "site_id": np.arange(N_BUILDINGS) % 3, "building_id": np.arange(N_BUILDINGS),
        "primary_use": rng.choice(["Education", "Office", None], N_BUILDINGS),
        "square_feet": rng.integers(1_000, 90_000, N_BUILDINGS),
        "year_built": rng.choice([1970.0, 1999.0, np.nan], N_BUILDINGS),
    }).to_csv(raw_dir / "building_metadata.csv", index=False)
    sites = np.repeat(np.arange(3), N_HOURS)
    pd.DataFrame({"site_id": sites, "timestamp": np.tile(times, 3),
                  "air_temperature": rng.normal(15, 5, len(sites)),
                  "wind_speed": rng.gamma(2, 2, len(sites))}).to_csv(raw_dir / "weather_train.csv", index=False)


def small_config() -> dict:
    """Return the project config with fast model settings for testing.

    Returns:
        A modified copy of the configuration.
    """
    config = copy.deepcopy(load_config())
    config["n_jobs"] = 1
    config["models"]["random_forest"]["n_estimators"] = 20
    config["models"]["isolation_forest"]["n_estimators"] = 20
    config["models"]["xgboost"]["n_estimators"] = 20
    config["plots"]["dpi"] = 60
    config["plots"]["formats"] = ["png"]
    return config


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, object]:
    """Run split, train_eval, aggregate and part of visualize in a temporary project.

    Args:
        tmp_path_factory: pytest temporary directory factory.

    Returns:
        ``(config, paths)`` after the pipeline ran.
    """
    root = tmp_path_factory.mktemp("project")
    config = small_config()
    paths = build_paths(config, root=root)
    setup_logging(paths.logs)
    make_synthetic_raw(paths.raw)
    options = RunOptions(models=list(config["models"]), splits=list(config["split"]["ratios"]))
    for stage in ("preprocess", "split", "train_eval", "aggregate"):
        stages.STAGE_FUNCTIONS[stage](config, paths, options)
    return config, paths


def test_all_runs_have_metrics(project: tuple) -> None:
    """Every (model, split) pair produces metrics, predictions and a model file."""
    config, paths = project
    expected = len(config["models"]) * len(config["split"]["ratios"])
    assert len(list(paths.metrics.glob("*.json"))) == expected
    assert len(list(paths.predictions.glob("*.npz"))) == expected
    assert len(list(paths.models.glob("*.joblib"))) == expected
    master = pd.read_csv(paths.tables / "master_results.csv")
    assert len(master) == expected
    assert master["f1"].between(0, 1).all()


def test_tables_exist_in_all_formats(project: tuple) -> None:
    """Each comparison table is exported as CSV, HTML and Markdown, plus one workbook."""
    _, paths = project
    for name in ("master_results", "pivot_f1", "ranking_table", "best_model_per_split",
                 "train_size_effect", "tuned_threshold_results", "data_split_summary"):
        for suffix in ("csv", "html", "md"):
            assert (paths.tables / f"{name}.{suffix}").is_file(), f"{name}.{suffix} missing"
    assert (paths.tables / "results_tables.xlsx").is_file()


def test_one_visualization(project: tuple) -> None:
    """The heatmap grid renders from the aggregated results."""
    from src.visualization import heatmaps
    from src.visualization.style import build_context

    config, paths = project
    master = pd.read_csv(paths.tables / "master_results.csv")
    ctx = build_context(config, paths.figures, list(config["models"]), list(config["split"]["ratios"]))
    heatmaps.plot_metric_heatmaps(master, ctx)
    assert (paths.figures / "01_metric_heatmaps.png").is_file()


def test_isolation_forest_is_unsupervised_by_contract(project: tuple) -> None:
    """Isolation Forest records ``supervised = False`` and no log loss."""
    _, paths = project
    master = pd.read_csv(paths.tables / "master_results.csv")
    row = master[master["model"] == "isolation_forest"].iloc[0]
    assert not bool(row["supervised"])
    assert pd.isna(row["log_loss"])
