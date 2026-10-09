"""Score one meter reading with all four trained models.

The reading is identified by building and hour; its building, weather and
calendar features are taken from the processed dataset, and the meter reading
can be replaced by any value to ask "what if this building had used X kWh?".

Usage::

    python -m src.predict                                  # interactive prompts
    python -m src.predict --building 107 --time "2016-03-05 14:00" --reading 0
    python -m src.predict --random                         # a random real test reading
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.data.preprocess import load_dataset_metadata
from src.splitting.splitter import load_split
from src.utils.config_loader import load_config
from src.utils.paths import ProjectPaths, build_paths

DEFAULT_SPLIT = "split_80_20"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Score one meter reading with every trained model.")
    parser.add_argument("--building", type=int, help="Building id (1-1353 in LEAD).")
    parser.add_argument("--time", help="Hour of the reading, e.g. '2016-03-05 14:00'.")
    parser.add_argument("--reading", type=float, help="Meter reading in kWh; omit to keep the recorded value.")
    parser.add_argument("--random", action="store_true", help="Use a random real reading from the test portion.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help=f"Which split's models to use (default {DEFAULT_SPLIT}).")
    return parser.parse_args(argv)


def find_row(paths: ProjectPaths, data_cfg: dict[str, Any], building: int, stamp: datetime) -> pd.DataFrame:
    """Load the processed row of one building at one hour.

    Args:
        paths: Project paths.
        data_cfg: The ``data`` config section.
        building: Building id.
        stamp: Hour of the reading.

    Returns:
        A one-row DataFrame.

    Raises:
        LookupError: If the building/hour is not in the labelled data.
    """
    table = pq.read_table(paths.dataset, filters=[(data_cfg["id_column"], "=", building)])
    times = table.column(data_cfg["time_column"])
    mask = pc.equal(times, pd.Timestamp(stamp).to_datetime64().astype("datetime64[us]"))
    row = table.filter(mask).to_pandas()
    if row.empty:
        known = len(table)
        raise LookupError(f"No labelled reading for building {building} at {stamp:%Y-%m-%d %H:%M}"
                          f" ({known} hours found for this building; data covers 2016, hourly).")
    return row.head(1)


def random_row(paths: ProjectPaths, split: str) -> pd.DataFrame:
    """Pick a random real reading from a split's test portion.

    Args:
        paths: Project paths.
        split: Split name.

    Returns:
        A one-row DataFrame.
    """
    _, test_idx, _ = load_split(paths.splits / f"{split}.npz")
    position = int(np.random.default_rng().choice(test_idx))
    parquet = pq.ParquetFile(paths.dataset)
    offset = 0
    for batch in parquet.iter_batches(batch_size=65_536):
        if position < offset + batch.num_rows:
            return batch.slice(position - offset, 1).to_pandas()
        offset += batch.num_rows
    raise LookupError("Random row out of range.")


def apply_reading(row: pd.DataFrame, reading: float | None, data_cfg: dict[str, Any]) -> bool:
    """Replace the meter reading (and its log feature) when the user gave a value.

    Args:
        row: The one-row DataFrame, modified in place.
        reading: New reading in kWh, or None to keep the recorded value.
        data_cfg: The ``data`` config section.

    Returns:
        True if the reading was changed.
    """
    if reading is None:
        return False
    row[data_cfg["reading"]] = np.float32(reading)
    if "log_meter_reading" in row.columns:
        row["log_meter_reading"] = np.float32(np.log1p(max(reading, 0.0)))
    return True


def score_all(paths: ProjectPaths, config: dict[str, Any], split: str, row: pd.DataFrame,
              features: list[str]) -> list[dict[str, Any]]:
    """Run the reading through every trained model of a split.

    Args:
        paths: Project paths.
        config: Parsed configuration dictionary.
        split: Split name.
        row: The one-row DataFrame.
        features: Model feature columns.

    Returns:
        One result per model.
    """
    names = config["plots"].get("display_names", {})
    results = []
    for model in config["models"]:
        bundle_path = paths.models / f"{split}__{model}.joblib"
        if not bundle_path.is_file():
            results.append({"model": names.get(model, model), "missing": True})
            continue
        bundle = joblib.load(bundle_path)
        X = bundle["imputer"].transform(row[features]).astype(np.float32)
        wrapper = bundle["model"]
        score = float(wrapper.predict_score(X)[0])
        results.append({"model": names.get(model, model), "score": score, "threshold": float(wrapper.threshold),
                        "anomaly": bool(wrapper.predict(X)[0]), "supervised": wrapper.supervised})
    return results


def prompt(args: argparse.Namespace) -> argparse.Namespace:
    """Ask for any missing input interactively.

    Args:
        args: Parsed arguments, filled in place.

    Returns:
        The completed arguments.
    """
    print("\nTest one meter reading against all four trained models.")
    print("Press Enter at the first question to use a random real test reading.\n")
    building = input("  Building id (1-1353): ").strip()
    if not building:
        args.random = True
        return args
    args.building = int(building)
    args.time = input("  Date and hour (e.g. 2016-03-05 14:00): ").strip()
    reading = input("  Meter reading in kWh (Enter = keep the recorded value): ").strip()
    args.reading = float(reading) if reading else None
    return args


def report(row: pd.DataFrame, results: list[dict[str, Any]], changed: bool, data_cfg: dict[str, Any],
           split: str) -> None:
    """Print the reading and every model's verdict.

    Args:
        row: The scored row.
        results: Output of :func:`score_all`.
        changed: Whether the reading was replaced by the user.
        data_cfg: The ``data`` config section.
        split: Split whose models were used.
    """
    r = row.iloc[0]
    print(f"\nBuilding {int(r[data_cfg['id_column']])} at {pd.Timestamp(r[data_cfg['time_column']]):%Y-%m-%d %H:%M}"
          f"  |  reading {float(r[data_cfg['reading']]):.2f} kWh{' (your value)' if changed else ''}")
    if not changed:
        print(f"Recorded label: {'ANOMALY' if int(r[data_cfg['target']]) else 'normal'}")
    print(f"Models trained on {split.replace('split_', '').replace('_', '/')} split:\n")
    print(f"  {'Model':<18}{'Score':>9}{'Threshold':>11}   Verdict")
    print("  " + "-" * 50)
    for res in results:
        if res.get("missing"):
            print(f"  {res['model']:<18}{'not trained yet':>20}")
            continue
        verdict = "ANOMALY" if res["anomaly"] else "normal"
        note = "" if res["supervised"] else "  (unsupervised)"
        print(f"  {res['model']:<18}{res['score']:>9.3f}{res['threshold']:>11.3f}   {verdict}{note}")
    votes = sum(res.get("anomaly", False) for res in results)
    print(f"\n  {votes} of {len(results)} models flag this reading as an anomaly.\n")


def main(argv: list[str] | None = None) -> int:
    """Score one reading and print the verdicts.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    config = load_config()
    paths = build_paths(config)
    if not paths.dataset.is_file():
        print("No processed data yet. Run the pipeline first (run.bat).", file=sys.stderr)
        return 1
    if args.building is None and not args.random:
        args = prompt(args)
    data_cfg = config["data"]
    try:
        if args.random:
            row = random_row(paths, args.split)
        else:
            stamp = datetime.fromisoformat(args.time.replace("/", "-"))
            row = find_row(paths, data_cfg, args.building, stamp)
    except (LookupError, ValueError, TypeError, FileNotFoundError) as error:
        print(f"Could not find that reading: {error}", file=sys.stderr)
        return 1
    changed = apply_reading(row, args.reading, data_cfg)
    features = load_dataset_metadata(paths.dataset)["feature_columns"]
    report(row, score_all(paths, config, args.split, row, features), changed, data_cfg, args.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
