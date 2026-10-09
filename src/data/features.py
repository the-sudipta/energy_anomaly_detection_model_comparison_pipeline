"""Feature engineering and the per-split imputation pipeline.

Imputation design: the processed dataset keeps numeric gaps as NaN. Median
imputation is a scikit-learn ``ColumnTransformer`` that the experiment runner
fits on the train portion of each split only, so no test statistic ever leaks
into training. Categorical gaps become an explicit ``"missing"`` category before
encoding; that is a constant, so it needs no fitting and cannot leak.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

MISSING_CATEGORY = "missing"
CODE_SUFFIX = "_code"


def add_time_features(frame: pd.DataFrame, time_column: str) -> pd.DataFrame:
    """Add calendar and cyclical time features derived from the timestamp.

    Args:
        frame: DataFrame with a datetime column.
        time_column: Name of that column.

    Returns:
        The DataFrame with hour, day_of_week, month, day_of_year, is_weekend,
        hour_sin/cos and dow_sin/cos columns added.
    """
    stamp = frame[time_column].dt
    frame["hour"] = stamp.hour.astype(np.int8)
    frame["day_of_week"] = stamp.dayofweek.astype(np.int8)
    frame["month"] = stamp.month.astype(np.int8)
    frame["day_of_year"] = stamp.dayofyear.astype(np.int16)
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(np.int8)
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24).astype(np.float32)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24).astype(np.float32)
    frame["dow_sin"] = np.sin(2 * np.pi * frame["day_of_week"] / 7).astype(np.float32)
    frame["dow_cos"] = np.cos(2 * np.pi * frame["day_of_week"] / 7).astype(np.float32)
    return frame


def add_reading_features(frame: pd.DataFrame, reading_column: str) -> pd.DataFrame:
    """Add ``log_meter_reading = log1p(max(reading, 0))``.

    Args:
        frame: DataFrame with the meter reading column.
        reading_column: Name of that column.

    Returns:
        The DataFrame with the log feature added (NaN readings stay NaN).
    """
    reading = frame[reading_column].astype(np.float32)
    frame["log_meter_reading"] = np.log1p(reading.clip(lower=0)).astype(np.float32)
    return frame


def add_rolling_features(
    frame: pd.DataFrame, id_column: str, time_column: str, reading_column: str, window: int
) -> pd.DataFrame:
    """Add per-building causal rolling mean/std of the reading and the reading ratio.

    The window only looks backwards in time (past and current hour) and uses
    readings, never labels, so it does not leak the target.

    Args:
        frame: DataFrame with building id, timestamp and reading.
        id_column: Building identifier column.
        time_column: Timestamp column.
        reading_column: Meter reading column.
        window: Window length in rows (hours).

    Returns:
        The DataFrame with roll_mean, roll_std and reading_to_roll_mean added,
        in its original row order.
    """
    order = frame.sort_values([id_column, time_column]).index
    grouped = frame.loc[order].groupby(id_column, observed=True)[reading_column]
    rolling = grouped.rolling(window=window, min_periods=1)
    frame.loc[order, "roll_mean"] = rolling.mean().to_numpy(dtype=np.float32)
    frame.loc[order, "roll_std"] = rolling.std().to_numpy(dtype=np.float32)
    ratio = frame[reading_column] / (frame["roll_mean"].abs() + 1e-6)
    frame["reading_to_roll_mean"] = ratio.astype(np.float32)
    return frame


def encode_categoricals(frame: pd.DataFrame, exclude: set[str]) -> list[str]:
    """Replace every categorical/text column by integer codes with a ``missing`` level.

    Args:
        frame: DataFrame modified in place.
        exclude: Columns never encoded (e.g. the timestamp).

    Returns:
        Names of the original categorical columns that were encoded.
    """
    encoded = []
    for column in list(frame.columns):
        series = frame[column]
        is_text = pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
        if column in exclude or not (is_text or isinstance(series.dtype, pd.CategoricalDtype)):
            continue
        filled = series.astype("string").fillna(MISSING_CATEGORY)
        frame[column + CODE_SUFFIX] = pd.Categorical(filled).codes.astype(np.int16)
        frame.drop(columns=column, inplace=True)
        encoded.append(column)
    return encoded


def select_feature_columns(frame: pd.DataFrame, data_cfg: dict[str, Any], use_id: bool) -> list[str]:
    """Pick the model input columns: every numeric column except target, time and (optionally) id.

    Args:
        frame: Processed DataFrame.
        data_cfg: The ``data`` config section.
        use_id: Whether the building id is allowed as a feature.

    Returns:
        Ordered list of feature column names.
    """
    excluded = {data_cfg["target"], data_cfg["time_column"], "row_id"}
    if not use_id:
        excluded.add(data_cfg["id_column"])
    return [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]


def build_imputer(feature_columns: list[str]) -> Pipeline:
    """Build the median imputer that is fitted on each split's train portion only.

    Args:
        feature_columns: Columns to impute, in model order.

    Returns:
        An unfitted scikit-learn ``Pipeline`` producing a float32 array.
    """
    transformer = ColumnTransformer(
        [("median", SimpleImputer(strategy="median", keep_empty_features=True), feature_columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("impute", transformer)])
