"""Leakage-safe hyperparameter search for one (split, model) pair.

The search only ever touches the split's train portion: a stratified
subsample of it is cut into stratified folds, every candidate is fitted on
the training folds (imputer included) and scored by PR-AUC on the held-out
fold. The configured defaults are always evaluated as the first candidate, so
the chosen setting is never worse than the baseline on cross-validation.

For Isolation Forest the labels are used only to score candidates; ``fit``
itself never sees them.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split

from src.data.features import build_imputer
from src.models.registry import build_model
from src.utils.logger import get_logger

_log = get_logger(__name__)


def candidates(model: str, tuning_cfg: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """List the settings to try: the defaults first, then random draws from the space.

    Args:
        model: Model name.
        tuning_cfg: The ``tuning`` config section.
        seed: Random seed.

    Returns:
        Parameter overrides, one dict per candidate.
    """
    space = tuning_cfg["spaces"].get(model) or {}
    n_iter = int(tuning_cfg["n_iter"].get(model, 10))
    drawn = list(ParameterSampler(space, n_iter=n_iter, random_state=seed)) if space else []
    unique, seen = [{}], set()
    for params in drawn:
        key = json.dumps(params, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            unique.append(params)
    return unique


def objective_name(model: str, tuning_cfg: dict[str, Any]) -> str:
    """Name of the tuning objective of a model: "PR-AUC" (default) or "F1".

    Args:
        model: Model name.
        tuning_cfg: The ``tuning`` config section.

    Returns:
        The objective label used in logs and results.
    """
    return "F1" if (tuning_cfg.get("objective") or {}).get(model) == "f1" else "PR-AUC"


def tune(X: Any, y: np.ndarray, feature_columns: list[str], train_idx: np.ndarray, model: str,
         config: dict[str, Any], label: str) -> dict[str, Any]:
    """Search the best hyperparameters for one model on one split's train portion.

    Args:
        X: Full feature frame (rows are selected by position).
        y: Full label array.
        feature_columns: Feature names in matrix order.
        train_idx: Row positions of this split's train portion.
        model: Model name.
        config: Parsed configuration dictionary.
        label: Prefix for log lines, e.g. ``"split_30_70 | xgboost"``.

    Returns:
        ``{"best": params, "cv_pr_auc": score, "default_cv_pr_auc": score, "trials": [...]}``.
    """
    cfg, seed = config["tuning"], config["seed"]
    size = min(int(cfg["sample_size"].get(model, 150_000)), len(train_idx))
    rows = train_idx if size >= len(train_idx) else train_test_split(
        train_idx, train_size=size, stratify=y[train_idx], random_state=seed)[0]
    folds = list(StratifiedKFold(int(cfg.get("folds", 3)), shuffle=True, random_state=seed).split(rows, y[rows]))
    trials = []
    for number, params in enumerate(candidates(model, cfg, seed), start=1):
        start = time.perf_counter()
        score = _cv_score(X, y, feature_columns, rows, folds, model, params, config)
        trials.append({"params": params, "cv_pr_auc": score, "seconds": time.perf_counter() - start})
        _log.info("[tune %s] trial %d/%d %s=%.4f %s", label, number, int(cfg["n_iter"].get(model, 10)) + 1,
                  objective_name(model, cfg), score, params or "(defaults)")
    best = max(trials, key=lambda t: t["cv_pr_auc"])
    return {"best": best["params"], "objective": objective_name(model, cfg),
            "cv_pr_auc": best["cv_pr_auc"], "default_cv_pr_auc": trials[0]["cv_pr_auc"],
            "sample_rows": int(len(rows)), "folds": len(folds), "trials": trials}


def _cv_score(X: Any, y: np.ndarray, feature_columns: list[str], rows: np.ndarray,
              folds: list[tuple[np.ndarray, np.ndarray]], model: str, params: dict[str, Any],
              config: dict[str, Any]) -> float:
    """Mean held-out PR-AUC of one candidate across the folds.

    Args:
        X: Full feature frame.
        y: Full label array.
        feature_columns: Feature names.
        rows: Row positions of the tuning subsample.
        folds: ``(train, validation)`` positions within ``rows``.
        model: Model name.
        params: Candidate overrides.
        config: Parsed configuration dictionary.

    Returns:
        Mean average precision (0.0 if a fold fails).
    """
    scores = []
    for fit_pos, val_pos in folds:
        fit_rows, val_rows = rows[fit_pos], rows[val_pos]
        imputer = build_imputer(feature_columns)
        X_fit = imputer.fit_transform(X.iloc[fit_rows]).astype(np.float32)
        X_val = imputer.transform(X.iloc[val_rows]).astype(np.float32)
        estimator = build_model(model, config, overrides=params)
        estimator.set_feature_names(feature_columns)
        try:
            estimator.fit(X_fit, y[fit_rows])
            if objective_name(model, config["tuning"]) == "F1":
                scores.append(f1_score(y[val_rows], estimator.predict(X_val), zero_division=0))
            else:
                scores.append(average_precision_score(y[val_rows], estimator.predict_score(X_val)))
        except (ValueError, MemoryError) as error:
            _log.warning("Candidate %s failed on a fold: %s", params, error)
            scores.append(0.0)
    return float(np.mean(scores))
