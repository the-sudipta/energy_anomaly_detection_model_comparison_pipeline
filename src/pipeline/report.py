"""Build the self-contained single-page ``REPORT.html``."""

from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Template

from src.evaluation.export import table_fragment

FIGURES = (
    ("00_dashboard", "Overview dashboard"),
    ("01_metric_heatmaps", "Metric heatmaps"),
    ("02_grouped_metric_bars", "Headline metrics per split"),
    ("03_train_size_lines", "Effect of training size"),
    ("04_radar_profiles", "Model profiles"),
    ("12_rank_bump_chart", "Ranking across splits"),
    ("07_curve_overlay", "ROC and PR overlay"),
    ("05_roc_grid", "ROC curve grid"),
    ("06_pr_grid", "Precision-recall curve grid"),
    ("08_confusion_grid", "Confusion matrices"),
    ("11_score_distributions", "Score distributions"),
    ("09_feature_importance_grid", "Feature importance"),
    ("10_efficiency", "Training and prediction cost"),
)


def key_findings(master: pd.DataFrame, best: pd.DataFrame, effect: pd.DataFrame, names: dict[str, str]) -> list[str]:
    """Write plain-language findings derived from the result tables.

    Args:
        master: Master results table.
        best: Best model per split table.
        effect: Train size effect table.
        names: Model display names.

    Returns:
        A list of short sentences.
    """
    def nm(model: str) -> str:
        return names.get(model, model)

    findings = []
    winners = best["best_by_f1"].value_counts()
    top = winners.index[0]
    findings.append(f"{nm(top)} has the highest F1 on {winners.iloc[0]} of {len(best)} splits.")
    for row in best.itertuples(index=False):
        findings.append(f"Split {row.split}: best F1 {nm(row.best_by_f1)} ({row.f1_value:.3f}), "
                        f"best PR-AUC {nm(row.best_by_pr_auc)} ({row.pr_auc_value:.3f}), "
                        f"best MCC {nm(row.best_by_mcc)} ({row.mcc_value:.3f}).")
    for row in effect.itertuples(index=False):
        direction = "rises" if row.f1_delta > 0.005 else "falls" if row.f1_delta < -0.005 else "barely moves"
        findings.append(f"{nm(row.model)}: F1 {direction} from {row.f1_from:.3f} to {row.f1_to:.3f} "
                        f"between {row.from_split} and {row.to_split} (PR-AUC change {row.pr_auc_delta:+.3f}).")
    gap = (master["accuracy"] - master["f1"]).mean()
    findings.append(f"Accuracy is on average {gap:.3f} higher than F1, which shows how the rare "
                    "anomaly class makes accuracy look better than detection really is.")
    speed = master.astype({"model": str}).groupby("model")["fit_seconds"].mean().sort_values()
    findings.append(f"Fastest to train: {nm(speed.index[0])} ({speed.iloc[0]:.1f}s on average). "
                    f"Slowest: {nm(speed.index[-1])} ({speed.iloc[-1]:.1f}s).")
    return findings


def method_notes(config: dict[str, Any], dataset_meta: dict[str, Any]) -> list[str]:
    """Describe the evaluation protocol.

    Args:
        config: Parsed configuration dictionary.
        dataset_meta: Processed dataset metadata.

    Returns:
        A list of short sentences.
    """
    sample = dataset_meta.get("sample_fraction")
    return [
        f"Split mode: {config['split']['mode']}. Every split is drawn independently from the full "
        f"labelled data with seed {config['seed']}; all models share identical row indices.",
        "Median imputation is fitted on each train portion only and applied to its test portion.",
        "Headline metrics use the default threshold (0.5 for supervised models). A best-F1 "
        "threshold tuned on the train portion is reported separately.",
        "Isolation Forest is unsupervised: labels are used only to set contamination to the train "
        "anomaly rate and for evaluation.",
        f"Anomalies make up {100 * dataset_meta['anomaly_rate']:.2f}% of readings, so accuracy is "
        "inflated; F1, PR-AUC, MCC, recall and precision are the metrics to read.",
        f"Data used: {'a stratified ' + format(sample, '.0%') + ' sample' if sample else 'all labelled rows'} "
        f"({dataset_meta['n_rows']:,} rows, {dataset_meta['n_features']} features).",
    ]


def embed_figure(figures_dir: Path, stem: str) -> str | None:
    """Return a figure as a base64 data URI, preferring SVG.

    Args:
        figures_dir: Figures folder.
        stem: File stem.

    Returns:
        The data URI, or None if the figure does not exist.
    """
    for suffix, mime in (("svg", "image/svg+xml"), ("png", "image/png")):
        path = figures_dir / f"{stem}.{suffix}"
        if path.is_file():
            return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
    return None


def render_report(context: dict[str, Any], template_path: Path, out_path: Path) -> None:
    """Render the Jinja2 template to the report file.

    Args:
        context: Template variables.
        template_path: Path of the HTML template.
        out_path: Destination file.
    """
    template = Template(template_path.read_text(encoding="utf-8"))
    out_path.write_text(template.render(**context), encoding="utf-8")


def build_context(
    tables: dict[str, pd.DataFrame], config: dict[str, Any], dataset_meta: dict[str, Any], figures_dir: Path
) -> dict[str, Any]:
    """Assemble every variable the template needs.

    Args:
        tables: Result tables keyed by name (master, pivots, best, effect, tuned, split summary).
        config: Parsed configuration dictionary.
        dataset_meta: Processed dataset metadata.
        figures_dir: Figures folder.

    Returns:
        The template context.
    """
    names = config["plots"].get("display_names", {})
    figures = [(title, uri) for stem, title in FIGURES if (uri := embed_figure(figures_dir, stem))]
    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": dataset_meta,
        "n_runs": len(tables["master_results"]),
        "split_table": table_fragment("data_split_summary", tables["data_split_summary"]),
        "headline_tables": [(m.upper().replace("_", "-"), table_fragment(f"pivot_{m}", tables[f"pivot_{m}"]))
                            for m in ("f1", "pr_auc", "mcc", "recall", "precision")],
        "tuned_table": table_fragment("tuned_threshold_results", tables["tuned_threshold_results"]),
        "findings": key_findings(tables["master_results"], tables["best_model_per_split"],
                                 tables["train_size_effect"], names),
        "notes": method_notes(config, dataset_meta),
        "figures": figures,
    }
