"""Random Forest classifier wrapper."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.base import BaseAnomalyModel


class RandomForestModel(BaseAnomalyModel):
    """Balanced-subsample Random Forest; score is the anomaly-class probability."""

    name = "random_forest"
    supervised = True

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestModel:
        """Fit the forest.

        Args:
            X: Training features.
            y: Training labels.

        Returns:
            The fitted model.
        """
        self.estimator = RandomForestClassifier(
            random_state=self.seed, n_jobs=self.n_jobs, **self.params
        ).fit(X, y)
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """Return P(anomaly).

        Args:
            X: Feature matrix.

        Returns:
            Probability of class 1.
        """
        return _positive_proba(self.estimator, X)


def _positive_proba(estimator: object, X: np.ndarray) -> np.ndarray:
    """Return the probability of class 1, or zeros if training saw only class 0.

    Args:
        estimator: A fitted scikit-learn style classifier.
        X: Feature matrix.

    Returns:
        A 1-D float array.
    """
    proba = estimator.predict_proba(X)
    classes = list(estimator.classes_)
    if 1 not in classes:
        return np.zeros(len(X), dtype=np.float64)
    return proba[:, classes.index(1)]
