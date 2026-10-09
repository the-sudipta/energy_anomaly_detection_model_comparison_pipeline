"""ROC and precision-recall curve grids, plus an overlay for one split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

from src.visualization.style import MUTED, PlotContext, headline, new_figure, save_figure

MAX_CURVE_POINTS = 400


def load_predictions(pred_dir: Path, split: str, model: str) -> dict[str, np.ndarray] | None:
    """Load the saved test predictions of one run.

    Args:
        pred_dir: Predictions folder.
        split: Split name.
        model: Model name.

    Returns:
        Dict with ``y_true``, ``y_pred``, ``y_score``; None if the run has no file.
    """
    path = pred_dir / f"{split}__{model}.npz"
    if not path.is_file():
        return None
    with np.load(path) as bundle:
        return {key: bundle[key] for key in ("y_true", "y_pred", "y_score")}


def _thin(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Down-sample a curve to keep SVG files small without changing its shape.

    Args:
        x: Curve x values.
        y: Curve y values.

    Returns:
        At most ``MAX_CURVE_POINTS`` points, endpoints included.
    """
    if len(x) <= MAX_CURVE_POINTS:
        return x, y
    keep = np.unique(np.linspace(0, len(x) - 1, MAX_CURVE_POINTS).astype(int))
    return x[keep], y[keep]


def draw_curve(ax: Axes, run: dict[str, np.ndarray], kind: str, color: str, label: str | None) -> float:
    """Draw a ROC or PR curve for one run and return its area.

    Args:
        ax: Target axes.
        run: Loaded predictions.
        kind: ``"roc"`` or ``"pr"``.
        color: Line colour.
        label: Optional legend label.

    Returns:
        ROC-AUC or average precision (NaN for a single-class test set).
    """
    y_true, y_score = run["y_true"], run["y_score"]
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if kind == "roc":
        fpr, tpr, _ = roc_curve(y_true, y_score)
        x, y, area = fpr, tpr, auc(fpr, tpr)
    else:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        x, y, area = recall, precision, average_precision_score(y_true, y_score)
    x, y = _thin(x, y)
    ax.plot(x, y, color=color, lw=1.8, label=label)
    ax.fill_between(x, y, alpha=0.08, color=color, step=None)
    return float(area)


def _baseline(ax: Axes, kind: str, positive_rate: float) -> None:
    """Draw the chance line: diagonal for ROC, prevalence for PR.

    Args:
        ax: Target axes.
        kind: ``"roc"`` or ``"pr"``.
        positive_rate: Anomaly share in the test portion.
    """
    if kind == "roc":
        ax.plot([0, 1], [0, 1], ls="--", lw=0.9, color=MUTED)
    else:
        ax.axhline(positive_rate, ls="--", lw=0.9, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)


def plot_curve_grid(ctx: PlotContext, pred_dir: Path, kind: str) -> None:
    """Save the models x splits grid of ROC or PR curves.

    Args:
        ctx: Plot context.
        pred_dir: Predictions folder.
        kind: ``"roc"`` or ``"pr"``.
    """
    fig = new_figure(4.2 * len(ctx.splits), 3.6 * len(ctx.models))
    axes = fig.subplots(len(ctx.models), len(ctx.splits), sharex=True, sharey=True, squeeze=False)
    area_name = "AUC" if kind == "roc" else "AP"
    for row, model in enumerate(ctx.models):
        for col, split in enumerate(ctx.splits):
            ax = axes[row, col]
            run = load_predictions(pred_dir, split, model)
            if run is None:
                ax.set_axis_off()
                continue
            area = draw_curve(ax, run, kind, ctx.color(model), None)
            _baseline(ax, kind, float(np.mean(run["y_true"])))
            ax.set_title(f"{ctx.name(model)} | {ctx.split_label(split)}   {area_name}={area:.3f}",
                         fontsize=10.5)
    _label_grid(axes, kind)
    title = "ROC curves" if kind == "roc" else "Precision-recall curves"
    headline(fig, f"{title} for every model and split",
             "Rows are models, columns are train/test splits; the dashed line is chance level.")
    save_figure(fig, ctx, "05_roc_grid" if kind == "roc" else "06_pr_grid")


def _label_grid(axes: np.ndarray, kind: str) -> None:
    """Put axis labels only on the outer panels.

    Args:
        axes: 2-D array of axes.
        kind: ``"roc"`` or ``"pr"``.
    """
    xlabel, ylabel = ("False positive rate", "True positive rate") if kind == "roc" else ("Recall", "Precision")
    for ax in axes[-1, :]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)


def plot_curve_overlay(ctx: PlotContext, pred_dir: Path, split: str) -> None:
    """Save ROC and PR overlays of all models for a single split.

    Args:
        ctx: Plot context.
        pred_dir: Predictions folder.
        split: Split to show (usually the largest training share).
    """
    fig = new_figure(14, 6)
    axes = fig.subplots(1, 2)
    for ax, kind in zip(axes, ("roc", "pr")):
        rate = 0.0
        for model in ctx.models:
            run = load_predictions(pred_dir, split, model)
            if run is None:
                continue
            rate = float(np.mean(run["y_true"]))
            area = draw_curve(ax, run, kind, ctx.color(model), None)
            ax.plot([], [], color=ctx.color(model), lw=2.5, label=f"{ctx.name(model)}  {area:.3f}")
        _baseline(ax, kind, rate)
        ax.legend(loc="lower right" if kind == "roc" else "upper right",
                  title="ROC-AUC" if kind == "roc" else "Average precision")
        ax.set_title("ROC" if kind == "roc" else "Precision-recall")
        ax.set_xlabel("False positive rate" if kind == "roc" else "Recall")
        ax.set_ylabel("True positive rate" if kind == "roc" else "Precision")
    headline(fig, f"All models on the {ctx.split_label(split)} split",
             "The precision-recall view is the more honest one for a rare anomaly class.")
    save_figure(fig, ctx, "07_curve_overlay")

