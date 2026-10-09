"""Merge the raw tables, clean them and build the processed dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import features
from src.data.load import load_raw_tables, memory_mb, validate_train_columns
from src.utils.logger import get_logger

_log = get_logger(__name__)


def build_processed_dataset(
    raw_dir: Path, config: dict[str, Any], sample_fraction: float | None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run loading, merging, cleaning, feature engineering and optional sampling.

    Args:
        raw_dir: Folder containing the raw CSV files.
        config: Parsed configuration dictionary.
        sample_fraction: Stratified fraction of rows to keep, or None for all.

    Returns:
        ``(dataset, metadata)`` where metadata lists features and summary stats.
    """
    data_cfg = config["data"]
    train, side_tables = load_raw_tables(raw_dir, data_cfg["time_column"])
    required = [data_cfg["target"], data_cfg["reading"], data_cfg["id_column"], data_cfg["time_column"]]
    validate_train_columns(train, required)
    frame = merge_feature_table(train, side_tables["features"], data_cfg)
    frame = merge_tables(frame, side_tables["metadata"], side_tables["weather"], data_cfg)
    del train, side_tables
    frame = clean(frame, data_cfg)
    frame = drop_unwanted(frame, config["features"], data_cfg)
    frame = engineer_features(frame, config)
    if sample_fraction:
        frame = stratified_sample(frame, data_cfg["target"], sample_fraction, config["seed"])
    feature_columns = features.select_feature_columns(
        frame, data_cfg, config["features"]["use_building_id"]
    )
    _log.info("Processed dataset: %s rows, %d features, ~%.0f MB in memory",
              f"{len(frame):,}", len(feature_columns), memory_mb(frame))
    return frame, describe(frame, feature_columns, data_cfg["target"], sample_fraction)


def merge_feature_table(
    train: pd.DataFrame, extra: pd.DataFrame | None, data_cfg: dict[str, Any]
) -> pd.DataFrame:
    """Left-join ``train_features.csv`` on (building id, timestamp).

    Columns already present in ``train`` (including the label) are dropped
    from the feature table first, so the label always comes from ``train.csv``.

    Args:
        train: Labelled readings.
        extra: The feature table, or None.
        data_cfg: The ``data`` config section.

    Returns:
        The merged DataFrame with the same number of rows as ``train``.
    """
    if extra is None:
        return train
    keys = [data_cfg["id_column"], data_cfg["time_column"]]
    if not all(key in extra.columns for key in keys):
        _log.warning("train_features.csv skipped: join keys %s not found.", keys)
        return train
    duplicated = [c for c in extra.columns if c in train.columns and c not in keys]
    extra = extra.drop(columns=duplicated).drop_duplicates(subset=keys)
    frame = train.merge(extra, on=keys, how="left")
    _log.info("Merged train_features.csv (+%d columns, dropped duplicates %s): %s -> %s rows",
              extra.shape[1] - len(keys), duplicated, f"{len(train):,}", f"{len(frame):,}")
    return frame


def merge_tables(
    train: pd.DataFrame,
    building_meta: pd.DataFrame | None,
    weather: pd.DataFrame | None,
    data_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Left-join building metadata and weather onto the labelled readings.

    Args:
        train: Labelled readings.
        building_meta: Building attributes, or None.
        weather: Hourly weather per site, or None.
        data_cfg: The ``data`` config section.

    Returns:
        The merged DataFrame with the same number of rows as ``train``.
    """
    id_column, time_column = data_cfg["id_column"], data_cfg["time_column"]
    rows_before = len(train)
    frame = train
    if building_meta is not None and id_column in building_meta.columns:
        overlap = [c for c in building_meta.columns if c in frame.columns and c != id_column]
        building_meta = building_meta.drop(columns=overlap)
        building_meta = building_meta.drop_duplicates(subset=id_column)
        frame = frame.merge(building_meta, on=id_column, how="left")
        _log.info("Merged building metadata: %s -> %s rows", f"{rows_before:,}", f"{len(frame):,}")
    keys = ["site_id", time_column]
    if weather is not None and any(c in frame.columns for c in weather.columns if c not in keys):
        _log.info("Weather columns already present (from train_features.csv); skipping weather join.")
        weather = None
    if weather is not None and all(key in weather.columns and key in frame.columns for key in keys):
        weather = weather.drop_duplicates(subset=keys)
        frame = frame.merge(weather, on=keys, how="left")
        _log.info("Merged weather: %s -> %s rows", f"{rows_before:,}", f"{len(frame):,}")
    elif weather is not None:
        _log.warning("Weather table skipped: join keys %s not present in both tables.", keys)
    return frame


def clean(frame: pd.DataFrame, data_cfg: dict[str, Any]) -> pd.DataFrame:
    """Drop rows that cannot be used and coerce the target to 0/1.

    Rows with a missing label or timestamp are removed. Missing readings are
    kept as NaN and imputed later inside each split.

    Args:
        frame: Merged DataFrame.
        data_cfg: The ``data`` config section.

    Returns:
        The cleaned DataFrame with a fresh RangeIndex.
    """
    target, time_column = data_cfg["target"], data_cfg["time_column"]
    before = len(frame)
    frame = frame.dropna(subset=[target, time_column])
    frame[target] = (frame[target].astype(np.float32) > 0).astype(np.int8)
    if before != len(frame):
        _log.info("Dropped %s rows without a label or timestamp.", f"{before - len(frame):,}")
    return frame.reset_index(drop=True)


def drop_unwanted(frame: pd.DataFrame, feat_cfg: dict[str, Any], data_cfg: dict[str, Any]) -> pd.DataFrame:
    """Drop configured columns and, optionally, constant columns.

    Args:
        frame: Cleaned DataFrame.
        feat_cfg: The ``features`` config section.
        data_cfg: The ``data`` config section (its columns are never dropped).

    Returns:
        The DataFrame without the dropped columns.
    """
    protected = {data_cfg["target"], data_cfg["reading"], data_cfg["id_column"], data_cfg["time_column"]}
    listed = [c for c in feat_cfg.get("drop_columns") or [] if c in frame.columns and c not in protected]
    constant = []
    if feat_cfg.get("drop_constant", True):
        constant = [c for c in frame.columns
                    if c not in protected and c not in listed and frame[c].nunique(dropna=False) <= 1]
    if listed or constant:
        _log.info("Dropping configured columns %s and constant columns %s", listed, constant)
    return frame.drop(columns=listed + constant)


def engineer_features(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Apply time, reading, optional rolling, and categorical encoding steps.

    Args:
        frame: Cleaned DataFrame.
        config: Parsed configuration dictionary.

    Returns:
        The DataFrame with engineered features.
    """
    data_cfg, feat_cfg = config["data"], config["features"]
    frame = features.add_time_features(frame, data_cfg["time_column"])
    frame = features.add_reading_features(frame, data_cfg["reading"])
    if feat_cfg.get("rolling_features"):
        frame = features.add_rolling_features(
            frame, data_cfg["id_column"], data_cfg["time_column"], data_cfg["reading"],
            int(feat_cfg.get("rolling_window_hours", 24)),
        )
    encoded = features.encode_categoricals(frame, exclude={data_cfg["time_column"]})
    if encoded:
        _log.info("Encoded categorical columns as integer codes: %s", encoded)
    return frame


def stratified_sample(frame: pd.DataFrame, target: str, fraction: float, seed: int) -> pd.DataFrame:
    """Keep a stratified random fraction of rows.

    Args:
        frame: Full DataFrame.
        target: Label column used for stratification.
        fraction: Fraction of rows to keep, in (0, 1].
        seed: Random seed.

    Returns:
        The sampled DataFrame with a fresh RangeIndex.
    """
    if fraction >= 1:
        return frame
    keep, _ = train_test_split(
        np.arange(len(frame)), train_size=fraction, stratify=frame[target], random_state=seed
    )
    _log.info("Stratified sample: kept %s of %s rows (%.1f%%).",
              f"{len(keep):,}", f"{len(frame):,}", 100 * fraction)
    return frame.iloc[np.sort(keep)].reset_index(drop=True)


def describe(
    frame: pd.DataFrame, feature_columns: list[str], target: str, sample_fraction: float | None
) -> dict[str, Any]:
    """Summarise the processed dataset for later stages and the report.

    Args:
        frame: Processed DataFrame.
        feature_columns: Model input columns.
        target: Label column.
        sample_fraction: Fraction used, or None.

    Returns:
        A JSON-serialisable metadata dictionary.
    """
    return {
        "n_rows": int(len(frame)),
        "n_anomalies": int(frame[target].sum()),
        "anomaly_rate": float(frame[target].mean()),
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "sample_fraction": sample_fraction,
        "missing_share": {c: float(frame[c].isna().mean()) for c in feature_columns},
    }


def save_dataset(frame: pd.DataFrame, metadata: dict[str, Any], dataset_path: Path) -> None:
    """Write the dataset to Parquet and its metadata to a sibling JSON file.

    Args:
        frame: Processed DataFrame.
        metadata: Output of :func:`describe`.
        dataset_path: Destination ``.parquet`` path.
    """
    frame.to_parquet(dataset_path, index=False)
    meta_path = dataset_path.with_name(dataset_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _log.info("Saved %s (%.1f MB on disk)", dataset_path.name, dataset_path.stat().st_size / 1e6)


def load_dataset_metadata(dataset_path: Path) -> dict[str, Any]:
    """Read the metadata JSON written next to the dataset.

    Args:
        dataset_path: Path of the processed ``.parquet`` file.

    Returns:
        The metadata dictionary.

    Raises:
        FileNotFoundError: If the metadata file does not exist.
    """
    meta_path = dataset_path.with_name(dataset_path.stem + "_meta.json")
    if not meta_path.is_file():
        raise FileNotFoundError(f"{meta_path} not found. Run the `preprocess` stage first.")
    return json.loads(meta_path.read_text(encoding="utf-8"))
