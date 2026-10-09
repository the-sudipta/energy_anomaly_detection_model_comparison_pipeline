"""Poster-style summary figure and an optional interactive Plotly dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.visualization.bars import draw_grouped_bars
from src.visualization.heatmaps import draw_heatmap
from src.visualization.lines import draw_metric_lines
from src.visualization.radar import draw_radar
from src.visualization.ranking import draw_bump
from src.visualization.style import PlotContext, headline, model_legend, new_figure, save_figure
from src.utils.logger import get_logger

_log = get_logger(__name__)


def plot_dashboard(master: pd.DataFrame, ranking: pd.DataFrame, ctx: PlotContext) -> None:
    """Save the 20x14 inch dashboard combining heatmaps, lines, a radar, bars and ranks.

    Args:
        master: Master results table.
        ranking: Ranking table.
        ctx: Plot context.
    """
    largest = max(ctx.splits, key=lambda s: ctx.train_pct[s])
    fig = new_figure(20, 14)
    grid = fig.add_gridspec(3, 6, height_ratios=[1, 1.05, 1])
    draw_heatmap(fig.add_subplot(grid[0, 0:3]), master, "f1", ctx, colorbar=False)
    draw_heatmap(fig.add_subplot(grid[0, 3:6]), master, "pr_auc", ctx, colorbar=True)
    draw_metric_lines(fig.add_subplot(grid[1, 0:2]), master, "f1", ctx)
    draw_metric_lines(fig.add_subplot(grid[1, 2:4]), master, "mcc", ctx)
    draw_radar(fig.add_subplot(grid[1, 4:6], projection="polar"), master, largest, ctx)
    draw_grouped_bars(fig.add_subplot(grid[2, 0:3]), master, largest, ctx)
    draw_bump(fig.add_subplot(grid[2, 3:6]), ranking, ctx)
    model_legend(fig, ctx, loc="outside upper right")
    headline(fig, "Energy anomaly detection: four models, five train/test splits",
             f"Test-set results on the LEAD meter readings. Row 1: F1 and PR-AUC per run. Row 2: "
             f"response to training share and the {ctx.split_label(largest)} profile. Row 3: "
             "headline metrics and mean rank.")
    save_figure(fig, ctx, "00_dashboard")


def write_interactive_dashboard(master: pd.DataFrame, ctx: PlotContext, path: Path) -> None:
    """Write a standalone interactive Plotly dashboard (skipped if Plotly is missing).

    Args:
        master: Master results table.
        ctx: Plot context.
        path: Destination HTML file.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        _log.warning("Plotly not installed; skipping interactive dashboard.")
        return
    metrics = ["f1", "pr_auc", "mcc", "recall"]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[m.upper().replace("_", "-") for m in metrics])
    frame = master.astype({"model": str, "split": str})
    for index, metric in enumerate(metrics):
        for model in ctx.models:
            rows = frame[frame["model"] == model].sort_values("train_fraction")
            fig.add_trace(go.Scatter(
                x=[ctx.split_label(s) for s in rows["split"]], y=rows[metric], mode="lines+markers",
                name=ctx.name(model), legendgroup=model, showlegend=index == 0,
                line={"color": ctx.color(model), "width": 3}, marker={"size": 9},
            ), row=index // 2 + 1, col=index % 2 + 1)
    fig.update_layout(template="plotly_white", height=820, title="Model x split benchmark",
                      font={"family": "Inter, Segoe UI, sans-serif"}, hovermode="x unified")
    fig.write_html(path, include_plotlyjs="cdn")
    ctx.saved.append(path)
