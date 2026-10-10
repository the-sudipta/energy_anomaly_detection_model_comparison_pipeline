"""Assemble the public website from ``index.html`` and the ``results/`` snapshot.

Run by the GitHub Pages workflow on every push (and usable locally). It reads
the committed result tables, tuning summaries, run information and logs,
writes them as one ``data.json`` and copies the page and figures next to it.
Only the Python standard library is used.

Usage::

    python tools/build_site.py --out _site
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

FIGURES = {
    "00_dashboard": ("The whole story on one poster", "F1 and PR-AUC for every run, how each model responds to more training data, the profile on the largest split and the ranking."),
    "01_metric_heatmaps": ("Every model on every split", "Eight metrics as small heatmaps; the gold outline marks the best model in each split."),
    "02_grouped_metric_bars": ("Five headline metrics side by side", "Precision, recall, F1, PR-AUC and MCC for each model, one panel per split."),
    "03_train_size_lines": ("Does more training data help?", "Each line follows one model from 30% to 80% training data."),
    "04_radar_profiles": ("Model profiles", "Bigger, rounder shapes mean a model is strong on every metric at once."),
    "12_rank_bump_chart": ("Who leads, split by split", "Mean rank across seven metrics; 1 is best."),
    "07_curve_overlay": ("ROC and precision-recall curves", "All four models on the largest training split."),
    "05_roc_grid": ("ROC curves for all 20 runs", "Rows are models, columns are splits."),
    "06_pr_grid": ("Precision-recall curves for all 20 runs", "The honest view when anomalies are rare."),
    "08_confusion_grid": ("Where each model is right and wrong", "Confusion matrices with percentages and raw counts."),
    "11_score_distributions": ("How well the scores separate", "Scores of normal readings versus true anomalies."),
    "09_feature_importance_grid": ("Which features matter", "Top 12 features for each supervised model and split."),
    "10_efficiency": ("Quality versus cost", "Training time against F1, and time per model."),
}
NUMERIC = ("accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "f2", "mcc", "roc_auc", "pr_auc",
           "fpr", "fnr", "tn", "fp", "fn", "tp", "n_train", "n_test", "fit_seconds", "predict_seconds",
           "train_anomaly_rate", "test_anomaly_rate", "train_fraction", "f1_tuned", "precision_tuned", "recall_tuned",
           "mcc_tuned", "threshold_tuned", "threshold_default")


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as a list of row dictionaries (empty if the file is missing).

    Args:
        path: CSV file.

    Returns:
        The rows.
    """
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value: str | None) -> float | None:
    """Convert a CSV cell to a number, or None when empty or not numeric.

    Args:
        value: Cell text.

    Returns:
        The number, or None.
    """
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if result != result else result


def result_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Keep the identifying and numeric columns of result rows.

    Args:
        rows: Rows of a master results table.

    Returns:
        Compact rows with numbers parsed.
    """
    compact = []
    for row in rows:
        item: dict[str, Any] = {"split": row["split"], "model": row["model"]}
        item.update({key: number(row.get(key)) for key in NUMERIC if key in row})
        if row.get("tuned_params"):
            item["tuned_params"] = row["tuned_params"]
        compact.append(item)
    return compact


def tuning_summary() -> dict[str, dict[str, Any]]:
    """Summarise every tuning result: objective, default and best score, chosen settings, all trials.

    Returns:
        ``{model: {split: summary}}``.
    """
    summary: dict[str, dict[str, Any]] = {}
    for path in sorted((RESULTS / "tuning").glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        summary.setdefault(result["model"], {})[result["split"]] = {
            "objective": result.get("objective", "PR-AUC"), "default": result["default_cv_pr_auc"],
            "best": result["cv_pr_auc"], "params": result["best"], "rows": result.get("sample_rows"),
            "folds": result.get("folds"),
            "trials": [{"score": t["cv_pr_auc"], "params": t["params"], "seconds": round(t.get("seconds", 0), 1)}
                       for t in result["trials"]],
        }
    return summary


def logs(info: dict[str, Any]) -> list[dict[str, Any]]:
    """Load the published log excerpts.

    Args:
        info: ``run_info.json`` content.

    Returns:
        One entry per log with its plan, duration line and text.
    """
    entries = []
    for entry in info.get("logs", []):
        path = RESULTS / "logs" / entry["file"]
        if path.is_file():
            entries.append({**entry, "text": path.read_text(encoding="utf-8")})
    return entries


def build(out_dir: Path) -> dict[str, Any]:
    """Write ``data.json`` and copy the page and figures into ``out_dir``.

    Args:
        out_dir: Destination folder for the site.

    Returns:
        The data written.

    Raises:
        FileNotFoundError: If the results snapshot is missing.
    """
    info_path = RESULTS / "run_info.json"
    if not info_path.is_file():
        raise FileNotFoundError("results/run_info.json not found. Run `python -m src.publish` first.")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    figures = [{"file": f"figures/{name}.webp", "title": title, "caption": caption}
               for name, (title, caption) in FIGURES.items() if (RESULTS / "figures" / f"{name}.webp").is_file()]
    data = {
        "published": info["published"], "dataset": info["dataset"], "seed": info["seed"],
        "split_mode": info["split_mode"], "split_ratios": info["split_ratios"], "python": info.get("python"),
        "platform": info.get("platform"), "libraries": info.get("libraries", {}),
        "display_names": info["display_names"], "palette": info["palette"], "base_settings": info["models"],
        "tuning_config": info.get("tuning", {}), "stage_timings": info.get("stage_timings", {}),
        "results": result_rows(read_csv(RESULTS / "tables" / "master_results.csv")),
        "baseline": result_rows(read_csv(RESULTS / "tables" / "baseline_untuned.csv")),
        "split_summary": [{k: (number(v) if k != "split" else v) for k, v in row.items()}
                          for row in read_csv(RESULTS / "tables" / "data_split_summary.csv")],
        "tuning": tuning_summary(), "figures": figures, "logs": logs(info),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.json").write_text(json.dumps(data), encoding="utf-8")
    shutil.copyfile(ROOT / "index.html", out_dir / "index.html")
    if (out_dir / "figures").is_dir():
        shutil.rmtree(out_dir / "figures")
    shutil.copytree(RESULTS / "figures", out_dir / "figures")
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")
    return data


def main() -> None:
    """Build the site and print a short summary."""
    parser = argparse.ArgumentParser(description="Build the public website from results/.")
    parser.add_argument("--out", default="_site", help="Output folder (default: _site).")
    args = parser.parse_args()
    data = build(ROOT / args.out)
    print(f"Site built in {args.out}/: {len(data['results'])} runs, {len(data['figures'])} figures, "
          f"{len(data['logs'])} logs, results published {data['published']}.")


if __name__ == "__main__":
    main()
