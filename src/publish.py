"""Copy a small, shareable snapshot of the results into ``results/`` for the website.

``outputs/`` is large and git-ignored. The public page is built from the
committed ``results/`` folder instead, which this module refreshes:

* result tables (CSV),
* figures, downscaled to WebP,
* tuning summaries and the untuned baseline,
* run information (versions, settings, dataset summary, stage timings),
* the real pipeline logs of the runs behind the results, with local paths removed.

No raw dataset rows are copied: the dataset is subject to the competition rules.

Usage::

    python -m src.publish
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.data.preprocess import load_dataset_metadata
from src.utils.config_loader import PROJECT_ROOT, load_config
from src.utils.paths import ProjectPaths, build_paths

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURE_WIDTH = 2000
TABLES = ("master_results", "data_split_summary", "best_model_per_split", "ranking_table",
          "train_size_effect", "tuned_threshold_results", "tuned_hyperparameters", "tuning_effect")
LOG_KEEP = re.compile(r"Plan:|Started:|Finished:|Done:|Loaded |Merged |Dropping |Added |Encoded |Processed dataset|"
                      r"split_\d\d_\d\d: train=|Feature matrix|best CV|\] F1=|Wrote \d+ tables|Saved \d+ figure|Report written|"
                      r"WARNING|ERROR")
MAX_LOGS = 6
MIN_LOG_SECONDS = 120  # runs shorter than this only reused cached work


def publish(paths: ProjectPaths, config: dict[str, Any], out_dir: Path = RESULTS_DIR) -> Path:
    """Refresh the shareable snapshot.

    Args:
        paths: Project paths.
        config: Parsed configuration dictionary.
        out_dir: Destination folder (committed to the repository).

    Returns:
        The destination folder.

    Raises:
        FileNotFoundError: If the pipeline has not produced results yet.
    """
    if not (paths.tables / "master_results.csv").is_file():
        raise FileNotFoundError("No results yet. Run the pipeline first (run.bat).")
    for sub in ("tables", "figures", "tuning", "logs"):
        folder = out_dir / sub
        if folder.is_dir():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
    copy_tables(paths, out_dir / "tables")
    figures = copy_figures(paths.figures, out_dir / "figures")
    copy_tuning(paths.outputs / "tuning", out_dir / "tuning")
    logs = copy_logs(paths.logs, out_dir / "logs")
    info = run_summary(paths, config, figures, logs)
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return out_dir


def copy_tables(paths: ProjectPaths, dest: Path) -> None:
    """Copy the result tables (CSV) and the untuned baseline.

    Args:
        paths: Project paths.
        dest: Destination folder.
    """
    for name in TABLES:
        source = paths.tables / f"{name}.csv"
        if source.is_file():
            shutil.copyfile(source, dest / source.name)
    baseline = paths.outputs / "baseline_untuned" / "master_results.csv"
    if baseline.is_file():
        shutil.copyfile(baseline, dest / "baseline_untuned.csv")


def copy_figures(source: Path, dest: Path) -> list[str]:
    """Save every PNG figure as a downscaled WebP.

    Args:
        source: Folder of full-resolution figures.
        dest: Destination folder.

    Returns:
        File names written, in order.
    """
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    written = []
    for path in sorted(source.glob("*.png")):
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.width > FIGURE_WIDTH:
                image = image.resize((FIGURE_WIDTH, round(image.height * FIGURE_WIDTH / image.width)), Image.LANCZOS)
            image.save(dest / f"{path.stem}.webp", "WEBP", quality=86, method=6)
        written.append(f"{path.stem}.webp")
    return written


def copy_tuning(source: Path, dest: Path) -> None:
    """Copy tuning results without their internal cache key.

    Args:
        source: ``outputs/tuning``.
        dest: Destination folder.
    """
    for path in sorted(source.glob("*.json")) if source.is_dir() else []:
        result = json.loads(path.read_text(encoding="utf-8"))
        result.pop("key", None)
        (dest / path.name).write_text(json.dumps(result, default=str), encoding="utf-8")


def sanitise(line: str) -> str:
    """Remove machine-specific absolute paths from a log line.

    Args:
        line: One log line.

    Returns:
        The line with the project root replaced by ``.`` and the home folder by ``~``.
    """
    for root, token in ((str(PROJECT_ROOT), "."), (str(Path.home()), "~")):
        line = line.replace(root, token).replace(root.replace("\\", "/"), token)
    return line


def copy_logs(source: Path, dest: Path) -> list[dict[str, Any]]:
    """Publish the key lines of the most recent completed runs.

    Args:
        source: ``outputs/logs``.
        dest: Destination folder.

    Returns:
        One summary per published log (file, plan, start, duration line).
    """
    completed = []
    for path in sorted(source.glob("pipeline_*.log")):
        match = re.search(r"\| Done: .* in (.+)$", path.read_text(encoding="utf-8", errors="replace"), re.MULTILINE)
        if match and _seconds(match.group(1).strip()) >= MIN_LOG_SECONDS:
            completed.append(path)
    summaries = []
    for path in completed[-MAX_LOGS:]:
        lines = [sanitise(line) for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                 if LOG_KEEP.search(line)]
        (dest / path.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        plan = next((line.split("Plan: ", 1)[1] for line in lines if "Plan: " in line), "full pipeline")
        done = next((line.split("| Done: ", 1)[1] for line in lines if "| Done: " in line), "")
        summaries.append({"file": path.name, "plan": plan, "start": lines[0][:19] if lines else "", "done": done,
                          "lines": len(lines)})
    return summaries


def stage_timings(log_dir: Path, names: list[str]) -> dict[str, str]:
    """Collect the longest recorded duration of every stage (the run that did the real work).

    Args:
        log_dir: ``outputs/logs``.
        names: Log file names to scan, oldest first.

    Returns:
        Mapping of stage name to its duration text.
    """
    timings: dict[str, str] = {}
    pattern = re.compile(r"Finished: (\w+) in (.+)$")
    for name in names:
        for line in (log_dir / name).read_text(encoding="utf-8", errors="replace").splitlines():
            if match := pattern.search(line):
                stage, duration = match.group(1), match.group(2).strip()
                if stage not in timings or _seconds(duration) > _seconds(timings[stage]):
                    timings[stage] = duration
    return timings


def _seconds(text: str) -> float:
    """Convert ``"3m 12s"`` or ``"4.2s"`` to seconds."""
    match = re.match(r"(?:(\d+)m )?([\d.]+)s", text)
    return (int(match.group(1) or 0) * 60 + float(match.group(2))) if match else 0.0


def run_summary(paths: ProjectPaths, config: dict[str, Any], figures: list[str],
                logs: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble everything the page needs besides the tables.

    Args:
        paths: Project paths.
        config: Parsed configuration dictionary.
        figures: Published figure file names.
        logs: Published log summaries.

    Returns:
        A JSON-serialisable summary.
    """
    meta = load_dataset_metadata(paths.dataset)
    info = json.loads(paths.run_info.read_text(encoding="utf-8")) if paths.run_info.is_file() else {}
    return {
        "published": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": {"n_rows": meta["n_rows"], "n_anomalies": meta["n_anomalies"], "anomaly_rate": meta["anomaly_rate"],
                    "n_features": meta["n_features"], "feature_columns": meta["feature_columns"],
                    "sample_fraction": meta.get("sample_fraction")},
        "python": info.get("python"), "platform": info.get("platform"), "libraries": info.get("libraries", {}),
        "seed": config["seed"], "split_mode": config["split"]["mode"], "split_ratios": config["split"]["ratios"],
        "models": config["models"], "display_names": config["plots"].get("display_names", {}),
        "palette": config["plots"]["palette"], "tuning": {k: v for k, v in config.get("tuning", {}).items()},
        "figures": figures, "logs": logs, "stage_timings": stage_timings(paths.logs, [log["file"] for log in logs]),
    }


def main() -> int:
    """Refresh ``results/`` and report what was written.

    Returns:
        Process exit code.
    """
    config = load_config()
    try:
        out_dir = publish(build_paths(config), config)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    files = [p for p in out_dir.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files) / 1e6
    print(f"Published {len(files)} files ({size:.1f} MB) to {out_dir.relative_to(PROJECT_ROOT)}/.")
    print("Commit and push that folder to update the website.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
