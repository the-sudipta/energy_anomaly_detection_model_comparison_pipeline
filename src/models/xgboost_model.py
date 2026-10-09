"""XGBoost classifier wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np
from xgboost import XGBClassifier

from src.models.base import BaseAnomalyModel


class XGBoostModel(BaseAnomalyModel):
    """Histogram XGBoost with ``scale_pos_weight`` computed from the train labels."""

    name = "xgboost"
    supervised = True

    def __init__(self, params: dict[str, Any], seed: int, n_jobs: int, use_gpu: bool = False) -> None:
        """Store settings.

        Args:
            params: Hyperparameters from the config.
            seed: Random seed.
            n_jobs: Parallel threads.
            use_gpu: Train on CUDA when True.
        """
        super().__init__(params, seed, n_jobs)
        self.use_gpu = use_gpu
        self.scale_pos_weight = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> XGBoostModel:
        """Fit the booster, weighting positives by ``n_neg / n_pos`` of this train portion.

        Args:
            X: Training features.
            y: Training labels.

        Returns:
            The fitted model.
        """
        n_pos = int(np.sum(y == 1))
        self.scale_pos_weight = (len(y) - n_pos) / n_pos if n_pos else 1.0
        params = dict(self.params)
        if self.use_gpu:
            params["device"] = "cuda"
        self.estimator = XGBClassifier(
            random_state=self.seed,
            n_jobs=self.n_jobs,
            scale_pos_weight=self.scale_pos_weight,
            **params,
        ).fit(X, y)
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """Return P(anomaly).

        Args:
            X: Feature matrix.

        Returns:
            Probability of class 1.
        """
        proba = self.estimator.predict_proba(X)
        return proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(X), dtype=np.float64)
