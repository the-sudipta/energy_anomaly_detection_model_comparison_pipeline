"""Common interface shared by every anomaly model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseAnomalyModel(ABC):
    """Wrapper giving every estimator the same fit/predict/score API.

    Subclasses set ``name``, ``supervised`` and build ``self.estimator``.
    ``predict_score`` always returns a continuous score where higher means
    more anomalous; ``predict`` returns 0 (normal) or 1 (anomaly).
    """

    name: str = "base"
    supervised: bool = True

    def __init__(self, params: dict[str, Any], seed: int, n_jobs: int) -> None:
        """Store settings; the estimator is built lazily in :meth:`fit`.

        Args:
            params: Model hyperparameters from the config.
            seed: Random seed.
            n_jobs: Parallel jobs for estimators that support it.
        """
        self.params = dict(params)
        self.seed = seed
        self.n_jobs = n_jobs
        self.estimator: Any = None
        self.threshold = 0.5

    def set_feature_names(self, names: list[str]) -> None:
        """Receive the feature names before fitting (used by models that select columns).

        Args:
            names: Feature names in matrix order.
        """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseAnomalyModel:
        """Fit the model on training data.

        Args:
            X: Feature matrix.
            y: Binary labels (unsupervised models may only use them for settings).

        Returns:
            The fitted model.
        """

    @abstractmethod
    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """Return a continuous anomaly score (higher = more anomalous).

        Args:
            X: Feature matrix.

        Returns:
            A 1-D float array.
        """

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard labels using the model's default threshold.

        Args:
            X: Feature matrix.

        Returns:
            A 1-D int8 array of 0/1 predictions.
        """
        return (self.predict_score(X) >= self.threshold).astype(np.int8)

    @property
    def feature_importances_(self) -> np.ndarray | None:
        """Native feature importances, or None when the estimator has none."""
        return getattr(self.estimator, "feature_importances_", None)

    @property
    def outputs_probability(self) -> bool:
        """Whether ``predict_score`` returns a calibrated-ish probability in [0, 1]."""
        return self.supervised
