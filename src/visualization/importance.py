"""Feature-importance grid: top features per supervised model and split."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.visualization.style import PlotContext, headline, new_figure, save_figure

TOP_N = 12


def load_importances(imp_dir: Path, split: str, model: str) -> pd.DataFrame | None:
    """Load the saved importances of one run.

    Args:
        imp_dir: Importances folder.
        split: Split name.
        model: Model name.

    Returns:
        DataFrame with ``feature`` and ``importance`` columns, or None.
    """
    path = imp_dir / f"{split}__{model}.csv"
    return pd.read_csv(path) if path.is_file() else None


def plot_importance_grid(ctx: PlotContext, imp_dir: Path) -> None:
    """Save the (models with importances) x splits grid of top-12 horizontal bars.

    Args:
        ctx: Plot context.
        imp_dir: Importances folder.
    """
    models = [m for m in ctx.models
              if any((imp_dir / f"{s}__{m}.csv").is_file() for s in ctx.splits)]
    if not models:
        return
    fig = new_figure(4.4 * len(ctx.splits), 4.6 * len(models))
    axes = fig.subplots(len(models), len(ctx.splits), squeeze=False)
    for row, model in enumerate(models):
        for col, split in enumerate(ctx.splits):
            ax = axes[row, col]
            frame = load_importances(imp_dir, split, model)
            if frame is None:
                ax.set_axis_off()
                continue
            top = frame.nlargest(TOP_N, "importance").iloc[::-1]
            total = frame["importance"].abs().sum() or 1.0
            ax.barh(top["feature"], top["importance"] / total, color=ctx.color(model), alpha=0.85)
            ax.set_title(f"{ctx.name(model)} | {ctx.split_label(split)}", fontsize=10.5)
            ax.tick_params(axis="y", labelsize=8.5)
            ax.grid(axis="y", visible=False)
            ax.set_xlabel("Share of total importance" if row == len(models) - 1 else "")
    note = ("Isolation Forest is excluded because it has no native importance "
            "(enable permutation importance in config.yaml to include it).")
    if "isolation_forest" in models:
        note = "Isolation Forest uses permutation importance (drop in PR-AUC) on a test sample."
    headline(fig, f"Top {TOP_N} features driving each model", note)
    save_figure(fig, ctx, "09_feature_importance_grid")
