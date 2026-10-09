"""Learning-curve style lines: metric vs train size, one line per model."""

from __future__ import annotations

import pandas as pd
from matplotlib.axes import Axes

from src.visualization.style import (
    PlotContext, grid_frame, headline, model_legend, new_figure, nice_metric, save_figure,
)

LINE_METRICS = ("f1", "pr_auc", "mcc", "recall", "precision", "accuracy")


def draw_metric_lines(ax: Axes, master: pd.DataFrame, metric: str, ctx: PlotContext) -> None:
    """Plot one metric against train percentage for every model.

    Args:
        ax: Target axes.
        master: Master results table.
        metric: Metric to plot.
        ctx: Plot context.
    """
    frame = grid_frame(master, metric, ctx)
    x = [ctx.train_pct[s] for s in ctx.splits]
    for model in ctx.models:
        y = frame.loc[model].to_numpy()
        ax.plot(x, y, color=ctx.color(model), lw=2.2, marker="o", ms=7,
                markeredgecolor="white", markeredgewidth=1.2, label=ctx.name(model))
        if len(y) and pd.notna(y[-1]):
            ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=ctx.color(model))
    ax.set_xticks(x, [f"{p}%" for p in x])
    ax.set_xlim(min(x) - 4, max(x) + 8)
    ax.set_title(nice_metric(metric))
    ax.set_xlabel("Training share")


def plot_learning_lines(master: pd.DataFrame, ctx: PlotContext) -> None:
    """Save the 2x3 small-multiple line chart.

    Args:
        master: Master results table.
        ctx: Plot context.
    """
    fig = new_figure(18, 10)
    axes = fig.subplots(2, 3)
    for ax, metric in zip(axes.ravel(), LINE_METRICS):
        draw_metric_lines(ax, master, metric, ctx)
    model_legend(fig, ctx, loc="outside upper right")
    headline(fig, "How each model responds to more training data",
             "Each point is an independent stratified split; the label at the right edge is the "
             "score with the largest training share.")
    save_figure(fig, ctx, "03_train_size_lines")
