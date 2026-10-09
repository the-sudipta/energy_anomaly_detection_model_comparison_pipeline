"""Build comparison tables across all runs and write them in several formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import LOWER_IS_BETTER
from src.utils.logger import get_logger

_log = get_logger(__name__)

PIVOT_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                 "mcc", "balanced_accuracy", "fit_seconds")
RANK_METRICS = ("precision", "recall", "f1", "pr_auc", "roc_auc", "mcc", "balanced_accuracy")
TUNED_COLUMNS = ("threshold_tuned", "precision_tuned", "recall_tuned", "f1_tuned",
                 "mcc_tuned", "balanced_accuracy_tuned")


def load_master(metrics_dir: Path, model_order: list[str], split_order: list[str]) -> pd.DataFrame:
    """Collect every per-run metrics JSON into one long table.

    Args:
        metrics_dir: Folder of ``<split>__<model>.json`` files.
        model_order: Preferred model ordering.
        split_order: Preferred split ordering.

    Returns:
        The master results table, one row per run, sorted by split then model.

    Raises:
        FileNotFoundError: If no metrics files exist.
    """
    files = sorted(metrics_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No metrics in {metrics_dir}. Run the `train_eval` stage first.")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    master = pd.DataFrame(rows)
    master = master[master["model"].isin(model_order) & master["split"].isin(split_order)]
    master["model"] = pd.Categorical(master["model"], categories=model_order, ordered=True)
    master["split"] = pd.Categorical(master["split"], categories=split_order, ordered=True)
    return master.sort_values(["split", "model"]).reset_index(drop=True)


def build_tables(master: pd.DataFrame, split_summary: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    """Derive every comparison table from the master table.

    Args:
        master: Output of :func:`load_master`.
        split_summary: Per-split sizes and anomaly rates, if available.

    Returns:
        An ordered mapping of table name to DataFrame.
    """
    tables: dict[str, pd.DataFrame] = {"master_results": master.astype({"model": str, "split": str})}
    for metric in PIVOT_METRICS:
        tables[f"pivot_{metric}"] = pivot(master, metric)
    tables["ranking_table"] = ranking_table(master)
    tables["best_model_per_split"] = best_model_per_split(master)
    tables["train_size_effect"] = train_size_effect(master)
    tables["tuned_threshold_results"] = tuned_threshold_results(master)
    if split_summary is not None:
        tables["data_split_summary"] = split_summary
    return tables


def pivot(master: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot one metric into a models x splits grid.

    Args:
        master: Master results table.
        metric: Metric column.

    Returns:
        DataFrame indexed by model with one column per split.
    """
    grid = master.pivot_table(index="model", columns="split", values=metric, observed=True)
    grid.index = grid.index.astype(str)
    grid.columns = grid.columns.astype(str)
    return grid


def ranking_table(master: pd.DataFrame) -> pd.DataFrame:
    """Rank models within each split for every ranking metric (1 = best).

    Args:
        master: Master results table.

    Returns:
        One row per (split, model) with ``rank_<metric>`` columns and ``mean_rank``.
    """
    ranked = master[["split", "model"]].astype(str).copy()
    for metric in RANK_METRICS:
        ascending = metric in LOWER_IS_BETTER
        ranked[f"rank_{metric}"] = master.groupby("split", observed=True)[metric].rank(
            ascending=ascending, method="min"
        )
    rank_columns = [f"rank_{metric}" for metric in RANK_METRICS]
    ranked["mean_rank"] = ranked[rank_columns].mean(axis=1)
    return ranked


def best_model_per_split(master: pd.DataFrame) -> pd.DataFrame:
    """Name the winning model per split by F1, PR-AUC and MCC.

    Args:
        master: Master results table.

    Returns:
        One row per split with the winner and its value for each criterion.
    """
    rows = []
    for split, group in master.groupby("split", observed=True):
        row: dict[str, Any] = {"split": str(split)}
        for metric in ("f1", "pr_auc", "mcc"):
            values = group[metric].fillna(-np.inf)
            best = group.loc[values.idxmax()]
            row[f"best_by_{metric}"] = str(best["model"])
            row[f"{metric}_value"] = float(best[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def train_size_effect(master: pd.DataFrame) -> pd.DataFrame:
    """Measure how F1 and PR-AUC change from the smallest to the largest train split.

    Args:
        master: Master results table (must contain ``train_fraction``).

    Returns:
        One row per model with start/end values and deltas.
    """
    rows = []
    for model, group in master.groupby("model", observed=True):
        ordered = group.sort_values("train_fraction")
        first, last = ordered.iloc[0], ordered.iloc[-1]
        row: dict[str, Any] = {"model": str(model), "from_split": str(first["split"]),
                               "to_split": str(last["split"])}
        for metric in ("f1", "pr_auc"):
            row[f"{metric}_from"] = float(first[metric])
            row[f"{metric}_to"] = float(last[metric])
            row[f"{metric}_delta"] = float(last[metric] - first[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def tuned_threshold_results(master: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side default vs train-tuned threshold metrics.

    Args:
        master: Master results table.

    Returns:
        One row per run with default and tuned precision/recall/F1/MCC.
    """
    columns = ["split", "model", "precision", "recall", "f1", "mcc", *TUNED_COLUMNS]
    present = [column for column in columns if column in master.columns]
    return master[present].astype({"split": str, "model": str}).reset_index(drop=True)


def best_mask(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mark the best numeric value in every column of a pivot table.

    Args:
        frame: Pivot table (models x splits).
        metric: Metric name, deciding whether lower or higher is better.

    Returns:
        Boolean DataFrame of the same shape.
    """
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    target = numeric.min() if metric in LOWER_IS_BETTER else numeric.max()
    return numeric.eq(target, axis=1)
