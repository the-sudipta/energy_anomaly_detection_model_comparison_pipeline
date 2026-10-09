"""Read the raw CSV files with memory-efficient dtypes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

_log = get_logger(__name__)

TRAIN_FILE = "train.csv"
METADATA_FILE = "building_metadata.csv"
WEATHER_FILE = "weather_train.csv"
FEATURES_FILE = "train_features.csv"
CATEGORY_MAX_UNIQUE_RATIO = 0.05


class DataValidationError(ValueError):
    """Raised when the raw data does not have the columns the pipeline needs."""


def read_csv_compact(path: Path, time_column: str | None = None) -> pd.DataFrame:
    """Read a CSV and downcast its columns to the smallest safe dtypes.

    Args:
        path: CSV file to read.
        time_column: Optional column parsed as datetime.

    Returns:
        The downcast DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing raw file: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if time_column and time_column in frame.columns:
        frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce")
    return downcast(frame, exclude={time_column} if time_column else set())


def downcast(frame: pd.DataFrame, exclude: set[str] | None = None) -> pd.DataFrame:
    """Downcast numeric columns and convert low-cardinality text to ``category``.

    Args:
        frame: DataFrame to shrink in place.
        exclude: Column names left untouched.

    Returns:
        The same DataFrame with smaller dtypes.
    """
    skip = exclude or set()
    for column in frame.columns:
        if column in skip:
            continue
        series = frame[column]
        if pd.api.types.is_float_dtype(series):
            frame[column] = series.astype(np.float32)
        elif pd.api.types.is_integer_dtype(series):
            frame[column] = pd.to_numeric(series, downcast="integer")
        elif pd.api.types.is_bool_dtype(series):
            frame[column] = series.astype(np.int8)
        elif _looks_categorical(series):
            frame[column] = series.astype("category")
    return frame


def _looks_categorical(series: pd.Series) -> bool:
    """Decide whether a text column should be stored as ``category``.

    Args:
        series: Column to inspect.

    Returns:
        True for object/string columns with few unique values.
    """
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    return series.nunique(dropna=True) <= max(50, CATEGORY_MAX_UNIQUE_RATIO * len(series))


def load_raw_tables(
    raw_dir: Path, time_column: str
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame | None]]:
    """Load the labelled training table and the optional side tables.

    The Kaggle release ships ``train_features.csv`` (building, weather and
    engineered ASHRAE features); the LEAD1.0 release ships separate
    ``building_metadata.csv`` and ``weather_train.csv``. Whatever exists is used.

    Args:
        raw_dir: Folder containing the raw CSV files.
        time_column: Name of the timestamp column.

    Returns:
        ``(train, side_tables)`` where side_tables maps ``"features"``,
        ``"metadata"`` and ``"weather"`` to a DataFrame or None.
    """
    train = read_csv_compact(raw_dir / TRAIN_FILE, time_column)
    _log.info("Loaded %s: %s rows, columns=%s", TRAIN_FILE, f"{len(train):,}", list(train.columns))
    side_tables = {
        "features": _load_optional(raw_dir / FEATURES_FILE, time_column),
        "metadata": _load_optional(raw_dir / METADATA_FILE, None),
        "weather": _load_optional(raw_dir / WEATHER_FILE, time_column),
    }
    return train, side_tables


def _load_optional(path: Path, time_column: str | None) -> pd.DataFrame | None:
    """Load a side table if it exists, logging a warning otherwise.

    Args:
        path: CSV file to read.
        time_column: Optional column parsed as datetime.

    Returns:
        The DataFrame, or None when the file is missing.
    """
    if not path.is_file():
        _log.info("Optional file %s not found; continuing without it.", path.name)
        return None
    frame = read_csv_compact(path, time_column)
    _log.info("Loaded %s: %s rows, columns=%s", path.name, f"{len(frame):,}", list(frame.columns))
    return frame


def validate_train_columns(train: pd.DataFrame, required: list[str]) -> None:
    """Fail with a clear message if a required column is absent.

    Args:
        train: The labelled training table.
        required: Column names that must exist.

    Raises:
        DataValidationError: If any required column is missing.
    """
    missing = [column for column in required if column not in train.columns]
    if missing:
        raise DataValidationError(
            f"train.csv is missing required column(s) {missing}. "
            f"Found columns: {list(train.columns)}. Check data.* names in config.yaml."
        )


def memory_mb(frame: pd.DataFrame) -> float:
    """Return the deep memory footprint of a DataFrame in megabytes.

    Args:
        frame: DataFrame to measure.

    Returns:
        Memory usage in MB.
    """
    return float(frame.memory_usage(deep=True).sum()) / 1e6
