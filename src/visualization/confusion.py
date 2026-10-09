"""Confusion-matrix grid: row-normalised colours with raw counts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from sklearn.metrics import confusion_matrix

from src.visualization.curves import load_predictions
from src.visualization.style import PlotContext, headline, new_figure, save_figure

CLASS_LABELS = ("Normal", "Anomaly")


def draw_confusion(ax: Axes, y_true: np.ndarray, y_pred: np.ndarray, cmap: str) -> None:
    """Draw one confusion matrix with row-percent colour and count + percent text.

    Args:
        ax: Target axes.
        y_true: True labels.
        y_pred: Predicted labels.
        cmap: Colormap name.
    """
    counts = confusion_matrix(y_true, y_pred, labels=[0, 1])
    totals = counts.sum(axis=1, keepdims=True)
    share = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    for (row, col), value in np.ndenumerate(counts):
        light = share[row, col] > 0.55
        ax.text(col, row, f"{100 * share[row, col]:.1f}%\n{value:,}", ha="center", va="center",
                fontsize=9.5, color="white" if light else "#1D2733", fontweight="bold")
    ax.set_xticks([0, 1], CLASS_LABELS)
    ax.set_yticks([0, 1], CLASS_LABELS)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_confusion_grid(ctx: PlotContext, pred_dir: Path) -> None:
    """Save the models x splits confusion-matrix grid.

    Args:
        ctx: Plot context.
        pred_dir: Predictions folder.
    """
    fig = new_figure(3.6 * len(ctx.splits), 3.4 * len(ctx.models))
    axes = fig.subplots(len(ctx.models), len(ctx.splits), squeeze=False)
    for row, model in enumerate(ctx.models):
        for col, split in enumerate(ctx.splits):
            ax = axes[row, col]
            run = load_predictions(pred_dir, split, model)
            if run is None:
                ax.set_axis_off()
                continue
            draw_confusion(ax, run["y_true"], run["y_pred"], "Blues")
            ax.set_title(f"{ctx.name(model)} | {ctx.split_label(split)}", fontsize=10.5,
                         color=ctx.color(model))
            ax.set_xlabel("Predicted" if row == len(ctx.models) - 1 else "")
            ax.set_ylabel("Actual" if col == 0 else "")
    headline(fig, "Where each model is right and wrong",
             "Colour and percentage are normalised per actual class (row); the second line is "
             "the raw count of test readings.")
    save_figure(fig, ctx, "08_confusion_grid")
