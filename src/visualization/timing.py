"""Efficiency view: accuracy-cost bubble scatter and stacked time bars."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from src.visualization.style import MUTED, PlotContext, finite, headline, model_legend, new_figure, save_figure


def draw_bubbles(ax: Axes, master: pd.DataFrame, ctx: PlotContext) -> None:
    """Scatter fit time vs F1; size = PR-AUC, colour = model, marker = split.

    Args:
        ax: Target axes.
        master: Master results table.
        ctx: Plot context.
    """
    for row in master.itertuples(index=False):
        model, split = str(row.model), str(row.split)
        size = 80 + 900 * float(finite([row.pr_auc])[0])
        ax.scatter(max(row.fit_seconds, 1e-3), row.f1, s=size, color=ctx.color(model),
                   marker=ctx.marker(split), alpha=0.75, edgecolor="white", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("Fit time (seconds, log scale)")
    ax.set_ylabel("F1 on test portion")
    ax.set_title("Quality vs training cost")
    handles = [Line2D([], [], marker=ctx.marker(s), ls="", color=MUTED, ms=9) for s in ctx.splits]
    ax.legend(handles, [ctx.split_label(s) for s in ctx.splits], title="Split (marker)",
              loc="lower right", ncol=2)


def draw_stacked_time(ax: Axes, master: pd.DataFrame, ctx: PlotContext) -> None:
    """Stacked bars of fit and predict seconds per model and split.

    Args:
        ax: Target axes.
        master: Master results table.
        ctx: Plot context.
    """
    width = 0.8 / max(len(ctx.splits), 1)
    x = np.arange(len(ctx.models))
    indexed = master.astype({"model": str, "split": str}).set_index(["model", "split"])
    for position, split in enumerate(ctx.splits):
        fit = [_get(indexed, m, split, "fit_seconds") for m in ctx.models]
        pred = [_get(indexed, m, split, "predict_seconds") for m in ctx.models]
        offsets = x - 0.4 + width * (position + 0.5)
        colors = [ctx.color(m) for m in ctx.models]
        ax.bar(offsets, fit, width * 0.9, color=colors, alpha=0.45 + 0.5 * position / len(ctx.splits))
        ax.bar(offsets, pred, width * 0.9, bottom=fit, color="#2B2F33", alpha=0.75)
    ax.set_xticks(x, [ctx.name(m) for m in ctx.models])
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel("Seconds (symlog)")
    ax.set_title("Fit (colour) + predict (dark cap) time; splits left to right = 30% to 80% train")
    ax.grid(axis="x", visible=False)


def _get(indexed: pd.DataFrame, model: str, split: str, column: str) -> float:
    """Read one cell from a (model, split)-indexed table, 0.0 if absent."""
    try:
        return float(indexed.loc[(model, split), column])
    except KeyError:
        return 0.0


def plot_efficiency(master: pd.DataFrame, ctx: PlotContext) -> None:
    """Save the two-panel efficiency figure.

    Args:
        master: Master results table.
        ctx: Plot context.
    """
    fig = new_figure(20, 7.5)
    left, right = fig.subplots(1, 2, width_ratios=[1, 1.25])
    draw_bubbles(left, master, ctx)
    draw_stacked_time(right, master, ctx)
    model_legend(fig, ctx, loc="outside upper right")
    headline(fig, "What each model costs to train and run",
             "Bubble area grows with PR-AUC. Top-left bubbles are the efficient sweet spot.")
    save_figure(fig, ctx, "10_efficiency")
