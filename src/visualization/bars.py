"""Grouped multi-metric bar charts, one panel per split."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from src.visualization.style import (
    PlotContext, finite, headline, new_figure, nice_metric, save_figure,
)

BAR_METRICS = ("precision", "recall", "f1", "pr_auc", "mcc")
METRIC_ALPHAS = (0.35, 0.5, 0.7, 0.85, 1.0)


def draw_grouped_bars(ax: Axes, master: pd.DataFrame, split: str, ctx: PlotContext) -> None:
    """Draw models on the x-axis with one bar per metric, shaded by metric.

    Args:
        ax: Target axes.
        master: Master results table.
        split: Split to show.
        ctx: Plot context.
    """
    subset = master[master["split"].astype(str) == split]
    subset = subset.set_index(subset["model"].astype(str))
    width = 0.8 / len(BAR_METRICS)
    x = np.arange(len(ctx.models))
    for position, (metric, alpha) in enumerate(zip(BAR_METRICS, METRIC_ALPHAS)):
        values = finite([subset[metric].get(m, np.nan) for m in ctx.models])
        offsets = x - 0.4 + width * (position + 0.5)
        ax.bar(offsets, values, width * 0.92, color=[ctx.color(m) for m in ctx.models],
               alpha=alpha, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x, [ctx.name(m).replace(" ", "\n") for m in ctx.models])
    ax.set_ylim(min(0.0, ax.get_ylim()[0]), 1.05)
    ax.axhline(0, color="#9AA3AD", lw=0.8)
    ax.set_title(f"Split {ctx.split_label(split)}")
    ax.grid(axis="x", visible=False)


def _metric_key(fig: object) -> None:
    """Add a legend explaining that shade encodes the metric.

    Args:
        fig: Target figure.
    """
    from matplotlib.patches import Patch

    handles = [Patch(facecolor="#4A5560", alpha=a) for a in METRIC_ALPHAS]
    fig.legend(handles, [nice_metric(m) for m in BAR_METRICS], loc="outside upper right",
               ncol=len(BAR_METRICS), title="Bar order and shade (light to dark)")


def plot_grouped_bars(master: pd.DataFrame, ctx: PlotContext) -> None:
    """Save the five-panel grouped bar chart.

    Args:
        master: Master results table.
        ctx: Plot context.
    """
    fig = new_figure(24, 6.5)
    axes = fig.subplots(1, len(ctx.splits), sharey=True, squeeze=False)[0]
    for ax, split in zip(axes, ctx.splits):
        draw_grouped_bars(ax, master, split, ctx)
    axes[0].set_ylabel("Score on test portion")
    _metric_key(fig)
    headline(fig, "Precision, recall, F1, PR-AUC and MCC side by side",
             "Colour identifies the model; within each model, bars run precision, recall, F1, "
             "PR-AUC, MCC from light to dark.")
    save_figure(fig, ctx, "02_grouped_metric_bars")
