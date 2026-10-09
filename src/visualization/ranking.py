"""Bump chart of mean rank across metrics, per model per split."""

from __future__ import annotations

import pandas as pd
from matplotlib.axes import Axes

from src.visualization.style import PlotContext, headline, new_figure, save_figure


def draw_bump(ax: Axes, ranking: pd.DataFrame, ctx: PlotContext) -> None:
    """Draw mean rank (1 = best, top) across splits, one line per model.

    Args:
        ax: Target axes.
        ranking: Ranking table with ``split``, ``model`` and ``mean_rank``.
        ctx: Plot context.
    """
    grid = ranking.pivot_table(index="model", columns="split", values="mean_rank", observed=True)
    grid = grid.reindex(index=ctx.models, columns=ctx.splits)
    x = list(range(len(ctx.splits)))
    for model in ctx.models:
        y = grid.loc[model].to_numpy()
        ax.plot(x, y, color=ctx.color(model), lw=3, marker="o", ms=24,
                markeredgecolor="white", markeredgewidth=2, solid_capstyle="round")
        for xi, yi in zip(x, y):
            if pd.notna(yi):
                ax.text(xi, yi, f"{yi:.1f}", ha="center", va="center", fontsize=8.5, color="white",
                        fontweight="bold")
    _end_labels(ax, grid[ctx.splits[-1]], ctx, len(x) - 1)
    ax.set_xticks(x, [ctx.split_label(s) for s in ctx.splits])
    ax.set_xlim(-0.3, len(x) + 0.3)
    ax.set_ylim(len(ctx.models) + 0.5, 0.5)
    ax.set_yticks(range(1, len(ctx.models) + 1))
    ax.set_ylabel("Mean rank across 7 metrics (1 = best)")
    ax.set_xlabel("Train / test split")
    ax.set_title("Mean rank per split")
    ax.grid(axis="x", visible=False)


def _end_labels(ax: Axes, last: pd.Series, ctx: PlotContext, x_end: int, gap: float = 0.22) -> None:
    """Write model names right of the last point, nudged apart so they never overlap.

    Args:
        ax: Target axes.
        last: Mean rank per model at the last split.
        ctx: Plot context.
        x_end: x position of the last split.
        gap: Minimum vertical distance between labels, in rank units.
    """
    placed: list[float] = []
    for model, value in last.dropna().sort_values().items():
        y = max(value, placed[-1] + gap) if placed else value
        placed.append(y)
        ax.annotate(ctx.name(model), (x_end, value), xytext=(x_end + 0.22, y), textcoords="data",
                    va="center", color=ctx.color(model), fontweight="bold")


def plot_ranking(ranking: pd.DataFrame, ctx: PlotContext) -> None:
    """Save the bump chart.

    Args:
        ranking: Ranking table.
        ctx: Plot context.
    """
    fig = new_figure(13, 6.5)
    draw_bump(fig.subplots(), ranking, ctx)
    headline(fig, "Which model leads, and does the order change with more data?",
             "Ranks are averaged over precision, recall, F1, PR-AUC, ROC-AUC, MCC and balanced "
             "accuracy within each split.")
    save_figure(fig, ctx, "12_rank_bump_chart")
