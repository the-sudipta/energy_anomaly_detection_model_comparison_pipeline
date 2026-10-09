"""Radar charts: all models overlaid on six metrics, one radar per split."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from src.visualization.style import (
    MUTED, PlotContext, finite, headline, model_legend, new_figure, nice_metric, save_figure,
)

RADAR_METRICS = ("precision", "recall", "f1", "pr_auc", "mcc", "balanced_accuracy")


def draw_radar(ax: Axes, master: pd.DataFrame, split: str, ctx: PlotContext) -> None:
    """Draw one polar radar with every model's profile for a split.

    Args:
        ax: Polar axes.
        master: Master results table.
        split: Split to show.
        ctx: Plot context.
    """
    subset = master[master["split"].astype(str) == split]
    subset = subset.set_index(subset["model"].astype(str))
    angles = np.linspace(0, 2 * np.pi, len(RADAR_METRICS), endpoint=False).tolist()
    closed = angles + angles[:1]
    for model in ctx.models:
        if model not in subset.index:
            continue
        values = finite(subset.loc[model, list(RADAR_METRICS)].to_numpy()).clip(0, 1).tolist()
        values += values[:1]
        ax.plot(closed, values, color=ctx.color(model), lw=2)
        ax.fill(closed, values, color=ctx.color(model), alpha=0.10)
    ax.set_xticks(angles, [nice_metric(m) for m in RADAR_METRICS], fontsize=9.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0], ["", "0.5", "", "1.0"], color=MUTED, fontsize=8)
    ax.spines["polar"].set_color("#D5DADF")
    ax.set_title(f"Split {ctx.split_label(split)}", pad=18)


def plot_radars(master: pd.DataFrame, ctx: PlotContext) -> None:
    """Save the five-panel radar figure.

    Args:
        master: Master results table.
        ctx: Plot context.
    """
    fig = new_figure(24, 6.2)
    axes = fig.subplots(1, len(ctx.splits), subplot_kw={"projection": "polar"}, squeeze=False)[0]
    for ax, split in zip(axes, ctx.splits):
        draw_radar(ax, master, split, ctx)
    model_legend(fig, ctx, loc="outside lower center")
    headline(fig, "Model profiles across six metrics",
             "Larger, rounder shapes mean a model is strong on every axis rather than trading "
             "recall for precision.")
    save_figure(fig, ctx, "04_radar_profiles")
