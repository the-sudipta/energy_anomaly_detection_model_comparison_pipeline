"""Anomaly-score distributions for true normal vs true anomaly readings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import seaborn as sns

from src.visualization.curves import load_predictions
from src.visualization.style import MUTED, PlotContext, headline, new_figure, save_figure

MAX_POINTS_PER_CLASS = 50_000


def _subsample(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cap the number of points so KDEs stay fast on the full dataset.

    Args:
        values: Scores of one class.
        rng: Random generator.

    Returns:
        At most ``MAX_POINTS_PER_CLASS`` values.
    """
    if len(values) <= MAX_POINTS_PER_CLASS:
        return values
    return rng.choice(values, MAX_POINTS_PER_CLASS, replace=False)


def plot_score_distributions(ctx: PlotContext, pred_dir: Path, split: str, seed: int) -> None:
    """Save one KDE panel per model comparing scores of normal and anomalous readings.

    Args:
        ctx: Plot context.
        pred_dir: Predictions folder.
        split: Split to show.
        seed: Random seed for sub-sampling.
    """
    rng = np.random.default_rng(seed)
    fig = new_figure(5 * len(ctx.models), 4.8)
    axes = fig.subplots(1, len(ctx.models), squeeze=False)[0]
    for ax, model in zip(axes, ctx.models):
        run = load_predictions(pred_dir, split, model)
        if run is None:
            ax.set_axis_off()
            continue
        for label, color, name in ((0, "#9AA5B1", "True normal"), (1, ctx.color(model), "True anomaly")):
            scores = _subsample(run["y_score"][run["y_true"] == label].astype(float), rng)
            if len(scores) > 1 and np.ptp(scores) > 0:
                sns.kdeplot(scores, ax=ax, fill=True, color=color, alpha=0.35, lw=1.8,
                            label=name, common_norm=False, cut=0, warn_singular=False)
        ax.set_title(ctx.name(model), color=ctx.color(model))
        ax.set_xlabel("Probability of anomaly" if model != "isolation_forest" else "-score_samples")
        ax.set_ylabel("Density" if ax is axes[0] else "")
        ax.legend(loc="upper center")
        ax.tick_params(axis="x", colors=MUTED)
    headline(fig, f"Score separation on the {ctx.split_label(split)} split",
             "Less overlap between the grey (normal) and coloured (anomaly) curves means the "
             "model ranks anomalies above normal readings more reliably.")
    save_figure(fig, ctx, "11_score_distributions")
