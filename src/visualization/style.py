"""Global plot theme, palette and figure-saving helpers shared by every chart."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

INK = "#1D2733"
MUTED = "#5B6773"
GRID = "#E6E9ED"
BACKGROUND = "#FFFFFF"
PANEL = "#FBFBFA"
SPLIT_MARKERS = ("o", "s", "D", "^", "P", "X", "v")
METRIC_LABELS = {
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced acc.", "precision": "Precision",
    "recall": "Recall", "f1": "F1", "f2": "F2", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
    "mcc": "MCC", "specificity": "Specificity", "fit_seconds": "Fit time (s)",
}


@dataclass
class PlotContext:
    """Everything a chart needs besides its data."""

    models: list[str]
    splits: list[str]
    palette: dict[str, str]
    names: dict[str, str]
    train_pct: dict[str, int]
    out_dir: Path
    dpi: int = 300
    formats: tuple[str, ...] = ("png", "svg")
    cmap: str = "mako"
    saved: list[Path] = field(default_factory=list)

    def color(self, model: str) -> str:
        """Return the fixed colour of a model."""
        return self.palette.get(model, MUTED)

    def name(self, model: str) -> str:
        """Return the display name of a model."""
        return self.names.get(model, model.replace("_", " ").title())

    def split_label(self, split: str) -> str:
        """Return a human split label such as ``"60/40"``."""
        pct = self.train_pct[split]
        return f"{pct}/{100 - pct}"

    def marker(self, split: str) -> str:
        """Return the marker shape assigned to a split."""
        return SPLIT_MARKERS[self.splits.index(split) % len(SPLIT_MARKERS)]


def build_context(config: dict[str, Any], out_dir: Path, models: list[str], splits: list[str]) -> PlotContext:
    """Create the plot context from the config and apply the global theme.

    Args:
        config: Parsed configuration dictionary.
        out_dir: Folder receiving the figures.
        models: Models present in the results, in display order.
        splits: Splits present in the results, in display order.

    Returns:
        A ready-to-use ``PlotContext``.
    """
    plots = config["plots"]
    apply_theme(plots.get("font_family", ["DejaVu Sans"]))
    ratios = config["split"]["ratios"]
    return PlotContext(
        models=models,
        splits=splits,
        palette=dict(plots["palette"]),
        names=dict(plots.get("display_names", {})),
        train_pct={s: int(round(100 * float(ratios[s]))) for s in splits},
        out_dir=out_dir,
        dpi=int(plots.get("dpi", 300)),
        formats=tuple(plots.get("formats", ["png", "svg"])),
        cmap=str(plots.get("heatmap_cmap", "mako")),
    )


def apply_theme(fonts: list[str]) -> None:
    """Set the global matplotlib/seaborn look: light, thin spines, subtle grid.

    Args:
        fonts: Preferred font families; the first installed one is used.
    """
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in fonts if f in installed), "DejaVu Sans")
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": family, "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 10, "axes.labelsize": 10.5,
        "axes.labelcolor": MUTED, "axes.edgecolor": "#C9CED4", "axes.linewidth": 0.7,
        "axes.facecolor": BACKGROUND, "figure.facecolor": BACKGROUND, "savefig.facecolor": BACKGROUND,
        "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID,
        "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.frameon": False, "legend.fontsize": 10, "text.color": INK,
        "axes.titlecolor": INK, "figure.titlesize": 17, "figure.titleweight": "bold",
        "svg.fonttype": "none",
    })


def new_figure(width: float, height: float, **kwargs: Any) -> Figure:
    """Create a figure with constrained layout.

    Args:
        width: Width in inches.
        height: Height in inches.
        **kwargs: Extra arguments for ``plt.figure``.

    Returns:
        The new figure.
    """
    return plt.figure(figsize=(width, height), layout="constrained", **kwargs)


def headline(fig: Figure, title: str, caption: str) -> None:
    """Add a left-aligned title and a muted caption to a figure.

    Args:
        fig: Target figure.
        title: Main title.
        caption: One-sentence explanation.
    """
    fig.suptitle(title, x=0.01, ha="left", color=INK)
    fig.supxlabel(caption, x=0.01, ha="left", fontsize=10, color=MUTED, style="italic")


def save_figure(fig: Figure, ctx: PlotContext, name: str) -> list[Path]:
    """Save a figure in every configured format and close it.

    Args:
        fig: Figure to save.
        ctx: Plot context with output folder, DPI and formats.
        name: File stem.

    Returns:
        Paths of the written files.
    """
    written = []
    for fmt in ctx.formats:
        path = ctx.out_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=ctx.dpi if fmt == "png" else None, bbox_inches="tight", pad_inches=0.25)
        written.append(path)
    plt.close(fig)
    ctx.saved.extend(written)
    return written


def model_legend(fig: Figure, ctx: PlotContext, loc: str = "outside upper right", **kwargs: Any) -> None:
    """Add one consistent model legend to a figure.

    Args:
        fig: Target figure.
        ctx: Plot context.
        loc: Legend location (``outside`` variants work with constrained layout).
        **kwargs: Extra legend arguments.
    """
    handles = [plt.Line2D([], [], color=ctx.color(m), lw=3, marker="o", ms=7) for m in ctx.models]
    fig.legend(handles, [ctx.name(m) for m in ctx.models], loc=loc,
               ncol=kwargs.pop("ncol", len(ctx.models)), **kwargs)


def grid_frame(master: pd.DataFrame, metric: str, ctx: PlotContext) -> pd.DataFrame:
    """Pivot a metric into a models x splits frame in context order.

    Args:
        master: Master results table.
        metric: Metric column.
        ctx: Plot context.

    Returns:
        DataFrame indexed by model with split columns (NaN where missing).
    """
    frame = master.astype({"model": str, "split": str}).pivot_table(
        index="model", columns="split", values=metric, observed=True
    )
    return frame.reindex(index=ctx.models, columns=ctx.splits).astype(float)


def nice_metric(metric: str) -> str:
    """Return the display label of a metric."""
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def finite(values: np.ndarray) -> np.ndarray:
    """Replace NaN/inf with 0 for plotting.

    Args:
        values: Array to clean.

    Returns:
        A cleaned copy.
    """
    return np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
