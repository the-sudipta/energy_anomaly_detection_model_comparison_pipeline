"""Create independent train/test splits and persist their row indices."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.utils.logger import get_logger

_log = get_logger(__name__)


def make_split(
    frame: pd.DataFrame, train_frac: float, mode: str, seed: int, data_cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Split row positions into train and test according to ``mode``.

    Args:
        frame: Processed dataset.
        train_frac: Fraction of rows (or buildings, for ``by_building``) used for training.
        mode: ``stratified_random``, ``by_building`` or ``temporal``.
        seed: Random seed shared by every split.
        data_cfg: The ``data`` config section.

    Returns:
        ``(train_idx, test_idx)`` as sorted int64 position arrays.

    Raises:
        ValueError: If ``mode`` is unknown.
    """
    positions = np.arange(len(frame))
    if mode == "stratified_random":
        train_idx, test_idx = train_test_split(
            positions, train_size=train_frac, stratify=frame[data_cfg["target"]], random_state=seed
        )
    elif mode == "by_building":
        splitter = GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
        train_idx, test_idx = next(splitter.split(positions, groups=frame[data_cfg["id_column"]]))
    elif mode == "temporal":
        order = np.argsort(frame[data_cfg["time_column"]].to_numpy(), kind="stable")
        cut = int(round(train_frac * len(order)))
        train_idx, test_idx = order[:cut], order[cut:]
    else:
        raise ValueError(f"Unknown split mode '{mode}'.")
    return np.sort(train_idx).astype(np.int64), np.sort(test_idx).astype(np.int64)


def save_split(path: Path, train_idx: np.ndarray, test_idx: np.ndarray, signature: str) -> None:
    """Persist split indices with the signature of the dataset they belong to.

    Args:
        path: Destination ``.npz`` file.
        train_idx: Train row positions.
        test_idx: Test row positions.
        signature: Dataset signature used to detect stale splits.
    """
    np.savez_compressed(path, train_idx=train_idx, test_idx=test_idx, signature=np.array(signature))


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Load split indices saved by :func:`save_split`.

    Args:
        path: ``.npz`` file to read.

    Returns:
        ``(train_idx, test_idx, signature)``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run the `split` stage first.")
    with np.load(path) as bundle:
        return bundle["train_idx"], bundle["test_idx"], str(bundle["signature"])


def split_summary(name: str, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, Any]:
    """Describe the size and class balance of one split.

    Args:
        name: Split name.
        y: Full label array.
        train_idx: Train row positions.
        test_idx: Test row positions.

    Returns:
        A dictionary with counts and anomaly rates for both portions.
    """
    y_train, y_test = y[train_idx], y[test_idx]
    summary = {
        "split": name,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "train_anomalies": int(y_train.sum()),
        "test_anomalies": int(y_test.sum()),
        "train_anomaly_rate": float(y_train.mean()) if len(y_train) else 0.0,
        "test_anomaly_rate": float(y_test.mean()) if len(y_test) else 0.0,
    }
    _log.info(
        "%s: train=%s (%s anomalies, %.2f%%) | test=%s (%s anomalies, %.2f%%)",
        name, f"{summary['n_train']:,}", f"{summary['train_anomalies']:,}",
        100 * summary["train_anomaly_rate"], f"{summary['n_test']:,}",
        f"{summary['test_anomalies']:,}", 100 * summary["test_anomaly_rate"],
    )
    return summary
