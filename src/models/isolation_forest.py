"""Isolation Forest wrapper (unsupervised).

The model never sees labels in ``fit``. Labels are used only to set
``contamination`` to the anomaly rate observed in the train portion and,
optionally, to fit on normal rows only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

from src.models.base import BaseAnomalyModel

MIN_CONTAMINATION = 1e-4
MAX_CONTAMINATION = 0.5


class IsolationForestModel(BaseAnomalyModel):
    """Isolation Forest; score is ``-score_samples`` so higher means more anomalous."""

    name = "isolation_forest"
    supervised = False

    def __init__(self, params: dict[str, Any], seed: int, n_jobs: int) -> None:
        """Store settings and pull out the wrapper-only options.

        Args:
            params: Hyperparameters from the config.
            seed: Random seed.
            n_jobs: Parallel jobs.
        """
        params = dict(params)
        self.contamination_setting = params.pop("contamination", "train_rate")
        self.fit_on_normal_only = bool(params.pop("fit_on_normal_only", False))
        super().__init__(params, seed, n_jobs)
        self.contamination = MIN_CONTAMINATION

    def fit(self, X: np.ndarray, y: np.ndarray) -> IsolationForestModel:
        """Fit on the train features without labels.

        Args:
            X: Training features.
            y: Training labels, used only for contamination and optional normal-only fitting.

        Returns:
            The fitted model.
        """
        if self.contamination_setting == "train_rate":
            rate = float(np.mean(y)) if len(y) else MIN_CONTAMINATION
        else:
            rate = float(self.contamination_setting)
        self.contamination = float(np.clip(rate, MIN_CONTAMINATION, MAX_CONTAMINATION))
        X_fit = X[y == 0] if self.fit_on_normal_only and np.any(y == 0) else X
        self.estimator = IsolationForest(
            contamination=self.contamination,
            random_state=self.seed,
            n_jobs=self.n_jobs,
            **self.params,
        ).fit(X_fit)
        self.threshold = float(-self.estimator.offset_)
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """Return ``-score_samples(X)``.

        Args:
            X: Feature matrix.

        Returns:
            Anomaly scores, higher = more anomalous.
        """
        return -self.estimator.score_samples(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Map sklearn's -1 (outlier) / 1 (inlier) to 1 (anomaly) / 0 (normal).

        Args:
            X: Feature matrix.

        Returns:
            A 1-D int8 array of 0/1 predictions.
        """
        return (self.estimator.predict(X) == -1).astype(np.int8)
