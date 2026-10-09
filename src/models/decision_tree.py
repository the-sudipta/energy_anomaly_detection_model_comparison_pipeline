"""Decision Tree classifier wrapper."""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from src.models.base import BaseAnomalyModel
from src.models.random_forest import _positive_proba


class DecisionTreeModel(BaseAnomalyModel):
    """Class-balanced Decision Tree; score is the anomaly-class leaf probability."""

    name = "decision_tree"
    supervised = True

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeModel:
        """Fit the tree.

        Args:
            X: Training features.
            y: Training labels.

        Returns:
            The fitted model.
        """
        self.estimator = DecisionTreeClassifier(random_state=self.seed, **self.params).fit(X, y)
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """Return P(anomaly).

        Args:
            X: Feature matrix.

        Returns:
            Probability of class 1.
        """
        return _positive_proba(self.estimator, X)
