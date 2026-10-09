"""Run the (model x split) experiments and save their artifacts."""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.features import build_imputer
from src.evaluation import metrics as mx
from src.models.base import BaseAnomalyModel
from src.models.registry import build_model
from src.splitting.splitter import load_split
from src.utils.logger import get_logger
from src.utils.paths import ProjectPaths

_log = get_logger(__name__)


@dataclass
class ExperimentData:
    """The processed dataset held in memory as arrays."""

    X: pd.DataFrame
    y: np.ndarray
    feature_columns: list[str]


def run_id(split: str, model: str) -> str:
    """Return the file stem used for every artifact of one run.

    Args:
        split: Split name.
        model: Model name.

    Returns:
        ``"<split>__<model>"``.
    """
    return f"{split}__{model}"


def run_all(
    data: ExperimentData,
    config: dict[str, Any],
    paths: ProjectPaths,
    runs: list[tuple[str, str]],
    force: bool,
) -> list[str]:
    """Execute every requested run, skipping finished ones unless ``force``.

    Args:
        data: Dataset arrays.
        config: Parsed configuration dictionary.
        paths: Project paths.
        runs: ``(split, model)`` pairs to execute.
        force: Re-run even if metrics already exist.

    Returns:
        Identifiers of runs that failed.
    """
    failures: list[str] = []
    for split, model in tqdm(runs, desc="Experiments", unit="run"):
        identifier = run_id(split, model)
        train_idx, test_idx, signature = load_split(paths.splits / f"{split}.npz")
        tuned = config.get("_tuned", {}).get(split, {}).get(model, {})
        signature = f"{signature}|params={json.dumps(tuned, sort_keys=True, default=str)}"
        if not force and _is_done(paths, identifier, signature):
            _log.info("[%s | %s] already done, skipping (use --force to re-run).", split, model)
            continue
        try:
            result = run_one(data, config, paths, split, model, (train_idx, test_idx))
            result["signature"] = signature
            result["tuned_params"] = tuned
            _save_metrics(paths, identifier, result)
            _log.info("[%s | %s] F1=%.3f PR-AUC=%.3f MCC=%.3f fit=%.1fs", split, model,
                      result["f1"], result["pr_auc"], result["mcc"], result["fit_seconds"])
        except Exception:  # one failing run must not stop the others
            _log.error("[%s | %s] failed:\n%s", split, model, traceback.format_exc())
            failures.append(identifier)
    return failures


def run_one(
    data: ExperimentData,
    config: dict[str, Any],
    paths: ProjectPaths,
    split: str,
    model_name: str,
    indices: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Fit preprocessing and one model on a train portion and evaluate it on the test portion.

    Args:
        data: Dataset arrays.
        config: Parsed configuration dictionary.
        paths: Project paths.
        split: Split name.
        model_name: Model name.
        indices: ``(train_idx, test_idx)`` row positions.

    Returns:
        The metrics dictionary for this run.
    """
    train_idx, test_idx = indices
    imputer = build_imputer(data.feature_columns)
    X_train = imputer.fit_transform(data.X.iloc[train_idx]).astype(np.float32)
    X_test = imputer.transform(data.X.iloc[test_idx]).astype(np.float32)
    y_train, y_test = data.y[train_idx], data.y[test_idx]
    model = build_model(model_name, config, split)
    model.set_feature_names(data.feature_columns)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    y_score = model.predict_score(X_test)
    predict_seconds = time.perf_counter() - start
    threshold = mx.best_f1_threshold(y_train, model.predict_score(X_train))
    result = _assemble(model, split, config, y_test, y_pred, y_score, y_train)
    result.update(mx.tuned_metrics(y_test, y_score, threshold))
    result.update(fit_seconds=fit_seconds, predict_seconds=predict_seconds)
    _save_artifacts(paths, run_id(split, model_name), model, imputer, (y_test, y_pred, y_score))
    _save_importances(paths, run_id(split, model_name), model, data.feature_columns, config,
                      (X_test, y_test))
    return result


def _assemble(
    model: BaseAnomalyModel,
    split: str,
    config: dict[str, Any],
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, Any]:
    """Combine identifiers, data sizes and metrics into one record.

    Args:
        model: The fitted model.
        split: Split name.
        config: Parsed configuration dictionary.
        y_test: Test labels.
        y_pred: Test predictions.
        y_score: Test scores.
        y_train: Train labels.

    Returns:
        The metrics record.
    """
    record: dict[str, Any] = {
        "split": split,
        "model": model.name,
        "supervised": model.supervised,
        "train_fraction": float(config["split"]["ratios"][split]),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_anomaly_rate": float(y_train.mean()),
        "test_anomaly_rate": float(y_test.mean()),
        "threshold_default": float(model.threshold),
    }
    record.update(mx.classification_metrics(y_test, y_pred, y_score, model.outputs_probability))
    return record


def _save_artifacts(
    paths: ProjectPaths,
    identifier: str,
    model: BaseAnomalyModel,
    imputer: object,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Persist the fitted model (with its imputer) and the test predictions.

    Args:
        paths: Project paths.
        identifier: Run identifier.
        model: Fitted model wrapper.
        imputer: Fitted imputation pipeline.
        arrays: ``(y_true, y_pred, y_score)`` for the test portion.
    """
    joblib.dump({"imputer": imputer, "model": model}, paths.models / f"{identifier}.joblib", compress=3)
    y_true, y_pred, y_score = arrays
    np.savez_compressed(paths.predictions / f"{identifier}.npz",
                        y_true=y_true, y_pred=y_pred, y_score=y_score.astype(np.float32))


def _save_importances(
    paths: ProjectPaths,
    identifier: str,
    model: BaseAnomalyModel,
    feature_columns: list[str],
    config: dict[str, Any],
    test_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Save native importances, or optional permutation importance for Isolation Forest.

    Args:
        paths: Project paths.
        identifier: Run identifier.
        model: Fitted model wrapper.
        feature_columns: Feature names in matrix order.
        config: Parsed configuration dictionary.
        test_data: ``(X_test, y_test)`` used for permutation importance.
    """
    importances = model.feature_importances_
    eval_cfg = config["evaluation"]
    if importances is None and eval_cfg.get("permutation_importance_isolation_forest"):
        importances = _permutation_importance(model, test_data, config)
    if importances is None:
        return
    frame = pd.DataFrame({"feature": feature_columns, "importance": np.asarray(importances, float)})
    frame.sort_values("importance", ascending=False).to_csv(
        paths.importances / f"{identifier}.csv", index=False
    )


def _permutation_importance(
    model: BaseAnomalyModel, test_data: tuple[np.ndarray, np.ndarray], config: dict[str, Any]
) -> np.ndarray:
    """Compute permutation importance (drop in PR-AUC) on a test sample.

    Args:
        model: Fitted model wrapper.
        test_data: ``(X_test, y_test)``.
        config: Parsed configuration dictionary.

    Returns:
        Mean importance per feature.
    """
    from sklearn.inspection import permutation_importance

    X_test, y_test = test_data
    rng = np.random.default_rng(config["seed"])
    size = min(len(y_test), int(config["evaluation"].get("permutation_sample_size", 20000)))
    pick = rng.choice(len(y_test), size=size, replace=False)

    def scorer(_: object, X: np.ndarray, y: np.ndarray) -> float:
        return mx.ranking_metrics(y, model.predict_score(X), False)["pr_auc"]

    result = permutation_importance(model, X_test[pick], y_test[pick], scoring=scorer,
                                    n_repeats=3, random_state=config["seed"])
    return result.importances_mean


def _is_done(paths: ProjectPaths, identifier: str, signature: str) -> bool:
    """Check whether a run already has metrics for the current split signature.

    Args:
        paths: Project paths.
        identifier: Run identifier.
        signature: Signature of the split the run must belong to.

    Returns:
        True if the run can be skipped.
    """
    path = paths.metrics / f"{identifier}.json"
    if not path.is_file():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("signature") == signature


def _save_metrics(paths: ProjectPaths, identifier: str, result: dict[str, Any]) -> None:
    """Write one run's metrics as JSON (NaN stored as null).

    Args:
        paths: Project paths.
        identifier: Run identifier.
        result: Metrics record.
    """
    clean = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in result.items()}
    (paths.metrics / f"{identifier}.json").write_text(json.dumps(clean, indent=2), encoding="utf-8")
