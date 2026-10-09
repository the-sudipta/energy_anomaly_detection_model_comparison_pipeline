"""Small-multiple heatmaps: one metric per panel, models x splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from src.visualization.style import PlotContext, grid_frame, headline, new_figure, nice_metric, save_figure

HEATMAP_METRICS = ("f1", "pr_auc", "mcc", "recall", "precision", "roc_auc", "balanced_accuracy", "accuracy")


def draw_heatmap(ax: Axes, master: pd.DataFrame, metric: str, ctx: PlotContext, colorbar: bool) -> None:
    """Draw one annotated models x splits heatmap with the best cell per split outlined.

    Args:
        ax: Target axes.
        master: Master results table.
        metric: Metric to show.
        ctx: Plot context.
        colorbar: Whether to attach a colour bar.
    """
    frame = grid_frame(master, metric, ctx)
    sns.heatmap(
        frame, ax=ax, cmap=ctx.cmap, vmin=_vmin(metric), vmax=1.0, annot=True, fmt=".3f",
        linewidths=1.5, linecolor="white", cbar=colorbar, annot_kws={"fontsize": 9.5},
        cbar_kws={"shrink": 0.8} if colorbar else None,
        yticklabels=[ctx.name(m) for m in frame.index],
        xticklabels=[ctx.split_label(s) for s in frame.columns],
    )
    for col, split in enumerate(frame.columns):
        column = frame[split].to_numpy()
        if np.all(np.isnan(column)):
            continue
        row = int(np.nanargmax(column))
        ax.add_patch(_outline(col, row))
    ax.set_title(nice_metric(metric))
    ax.set_xlabel("Train / test split")
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)


def _outline(col: int, row: int) -> object:
    """Return a rectangle highlighting one heatmap cell.

    Args:
        col: Column position.
        row: Row position.

    Returns:
        A matplotlib Rectangle patch.
    """
    from matplotlib.patches import Rectangle

    return Rectangle((col + 0.04, row + 0.04), 0.92, 0.92, fill=False, ec="#F2B33D", lw=2.2)


def _vmin(metric: str) -> float:
    """Lower colour limit: MCC can be negative, ROC-AUC is centred on 0.5."""
    return {"mcc": -0.2, "roc_auc": 0.4}.get(metric, 0.0)


def plot_metric_heatmaps(master: pd.DataFrame, ctx: PlotContext) -> None:
    """Save the 2x4 metric heatmap grid.

    Args:
        master: Master results table.
        ctx: Plot context.
    """
    fig = new_figure(22, 9.5)
    axes = fig.subplots(2, 4, sharey=True)
    for index, (ax, metric) in enumerate(zip(axes.ravel(), HEATMAP_METRICS)):
        draw_heatmap(ax, master, metric, ctx, colorbar=index % 4 == 3)
    headline(fig, "Every model on every split, metric by metric",
             "Cells show test-set scores at the default threshold; the gold outline marks the best "
             "model in each split. Accuracy is inflated by the rare anomaly class.")
    save_figure(fig, ctx, "01_metric_heatmaps")
