"""Compact, data-driven summaries of fitted models for the live 3D view.

For a finished (split, model) run this loads the saved model bundle and a small
stratified sample of that split's real test readings, then extracts exactly
what the animations show:

* decision_tree    - the real top levels of the fitted tree and the real
                     decision paths of the sample readings.
* random_forest    - per-tree votes of nine real trees and the forest score.
* isolation_forest - PCA positions, real scores and average isolation path
                     lengths of real readings.
* xgboost          - real test log-loss after increasing numbers of rounds.

Results are cached as JSON in ``outputs/digests`` so each run is summarised once.
"""

from __future__ import annotations

import ctypes
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DIGEST_DIR = ROOT / "outputs" / "digests"
SAMPLE_PER_CLASS = 20
IF_CLOUD = 300
DT_DEPTH = 4
RF_TREES = 9
XGB_ROUNDS = (1, 5, 10, 25, 50, 100, 200, 400)
HEAVY_MODEL_BYTES = 50_000_000
MIN_FREE_BYTES = 600_000_000


def free_memory_bytes() -> int | None:
    """Return available physical memory on Windows, or None elsewhere.

    Returns:
        Free bytes, or None when unknown.
    """
    if sys.platform != "win32":
        return None

    class Status(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = Status()
    status.dwLength = ctypes.sizeof(Status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return int(status.ullAvailPhys)


def can_build(model_path: Path, training_active: bool) -> tuple[bool, str]:
    """Decide whether building a digest now is safe for the running pipeline.

    Args:
        model_path: Saved model bundle.
        training_active: Whether the train_eval stage is still running.

    Returns:
        ``(allowed, reason_if_not)``.
    """
    if training_active and model_path.stat().st_size > HEAVY_MODEL_BYTES:
        return False, "Large model: summarised once training has finished, to keep memory free."
    free = free_memory_bytes()
    if training_active and free is not None and free < MIN_FREE_BYTES:
        return False, f"Waiting for free memory ({free / 1e9:.1f} GB free) so the pipeline is not slowed."
    return True, ""


def digest_path(run_id: str) -> Path:
    """Return the cache file of a run's digest."""
    return DIGEST_DIR / f"{run_id}.json"


def build_digest(split: str, model: str, signature: str | None) -> dict[str, Any]:
    """Build and cache the digest of one finished run.

    Args:
        split: Split name.
        model: Model name.
        signature: Split signature stored with the run's metrics, kept for cache validation.

    Returns:
        The digest dictionary.
    """
    run_id = f"{split}__{model}"
    bundle = joblib.load(ROOT / "outputs" / "models" / f"{run_id}.joblib")
    features, X, y = _sample(split, model, bundle["imputer"])
    estimator = bundle["model"].estimator
    builders = {"decision_tree": _decision_tree, "random_forest": _random_forest,
                "isolation_forest": _isolation_forest, "xgboost": _xgboost}
    digest = builders[model](estimator, bundle["model"], features, X, y)
    digest.update(run=run_id, split=split, model=model, signature=signature)
    DIGEST_DIR.mkdir(parents=True, exist_ok=True)
    digest_path(run_id).write_text(json.dumps(digest), encoding="utf-8")
    return digest


def _sample(split: str, model: str, imputer: Any) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Draw a seeded sample of real test readings and impute them like the run did.

    Isolation Forest gets a larger mostly-normal cloud plus the stratified sample.

    Args:
        split: Split name.
        model: Model name.
        imputer: The fitted imputation pipeline saved with the run.

    Returns:
        ``(feature_names, X_sample, y_sample)``.
    """
    meta = json.loads((ROOT / "data" / "processed" / "dataset_meta.json").read_text(encoding="utf-8"))
    features = meta["feature_columns"]
    with np.load(ROOT / "data" / "processed" / "splits" / f"{split}.npz") as bundle:
        test_idx = bundle["test_idx"]
    y_all = pq.read_table(ROOT / "data" / "processed" / "dataset.parquet", columns=["anomaly"])
    y_test = y_all.column("anomaly").to_numpy()[test_idx]
    rng = np.random.default_rng(42)
    pos, neg = test_idx[y_test == 1], test_idx[y_test == 0]
    pick = [rng.choice(pos, min(SAMPLE_PER_CLASS, len(pos)), replace=False),
            rng.choice(neg, min(SAMPLE_PER_CLASS, len(neg)), replace=False)]
    if model == "isolation_forest":
        pick.append(rng.choice(test_idx, min(IF_CLOUD, len(test_idx)), replace=False))
    rows = np.concatenate(pick)
    frame = _read_rows(ROOT / "data" / "processed" / "dataset.parquet", rows, [*features, "anomaly"])
    y = frame.pop("anomaly").to_numpy().astype(int)
    X = imputer.transform(frame).astype(np.float32)
    order = np.arange(len(y))
    head = order[: 2 * SAMPLE_PER_CLASS]
    rng.shuffle(head)
    order[: len(head)] = head
    return features, X[order], y[order]


def _read_rows(path: Path, rows: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Read selected rows of a Parquet file in small batches to keep memory low.

    Args:
        path: Parquet file.
        rows: Row positions to keep, in the order wanted.
        columns: Columns to read.

    Returns:
        A DataFrame with one row per requested position, in the requested order.
    """
    wanted = np.sort(np.unique(rows))
    parts, offset = [], 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65_536, columns=columns):
        end = offset + batch.num_rows
        lo, hi = np.searchsorted(wanted, [offset, end])
        if hi > lo:
            parts.append(batch.take(wanted[lo:hi] - offset).to_pandas())
        offset = end
    frame = pd.concat(parts, ignore_index=True)
    frame.index = wanted
    return frame.loc[rows].reset_index(drop=True)


def _decision_tree(est: Any, wrapper: Any, features: list[str], X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Real top levels of the tree and real root-to-leaf paths.

    Args:
        est: Fitted DecisionTreeClassifier.
        wrapper: Model wrapper.
        features: Feature names.
        X: Sample features.
        y: Sample labels.

    Returns:
        Digest with ``nodes`` (heap-indexed, depth < DT_DEPTH) and ``samples``.
    """
    tree = est.tree_
    nodes, heap = {}, {0: 0}
    queue = [(0, 0)]
    while queue:
        node, slot = queue.pop(0)
        depth = int(math.log2(slot + 1))
        leaf = tree.children_left[node] == -1
        nodes[slot] = {"slot": slot, "depth": depth, "leaf": bool(leaf or depth == DT_DEPTH - 1),
                       "feature": None if leaf else features[tree.feature[node]],
                       "threshold": None if leaf else float(tree.threshold[node]),
                       "samples": int(tree.n_node_samples[node]), "anomaly_share": _share(tree.value[node])}
        if not leaf and depth < DT_DEPTH - 1:
            for child, cs in ((tree.children_left[node], 2 * slot + 1), (tree.children_right[node], 2 * slot + 2)):
                heap[cs] = int(child)
                queue.append((int(child), cs))
    path = est.decision_path(X)
    scores = wrapper.predict_score(X)
    samples = []
    for i in range(len(y)):
        visited = set(path.indices[path.indptr[i]:path.indptr[i + 1]].tolist())
        slots = [0]
        while len(slots) < DT_DEPTH:
            last = slots[-1]
            nxt = [s for s in (2 * last + 1, 2 * last + 2) if s in heap and heap[s] in visited]
            if not nxt:
                break
            slots.append(nxt[0])
        samples.append({"path": slots, "truth": int(y[i]), "score": float(scores[i]),
                        "depth": int(len(visited) - 1)})
    return {"kind": "decision_tree", "nodes": list(nodes.values()), "samples": samples[: 2 * SAMPLE_PER_CLASS],
            "tree_depth": int(est.get_depth()), "n_leaves": int(est.get_n_leaves())}


def _share(value: np.ndarray) -> float:
    """Anomaly share of a node from sklearn's per-class value array."""
    counts = np.asarray(value).ravel()
    total = counts.sum()
    return float(counts[1] / total) if total and len(counts) > 1 else 0.0


def _random_forest(est: Any, wrapper: Any, features: list[str], X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Real votes of nine trees and the full-forest score for each sample reading.

    Args:
        est: Fitted RandomForestClassifier.
        wrapper: Model wrapper.
        features: Feature names.
        X: Sample features.
        y: Sample labels.

    Returns:
        Digest with ``trees`` (depth, leaves) and per-sample ``votes``.
    """
    trees = est.estimators_[:RF_TREES]
    votes = np.stack([t.predict_proba(X)[:, 1] for t in trees], axis=1)
    forest = wrapper.predict_score(X)
    top = np.argsort(est.feature_importances_)[::-1][:5]
    return {"kind": "random_forest", "n_trees": len(est.estimators_),
            "trees": [{"depth": int(t.get_depth()), "leaves": int(t.get_n_leaves())} for t in trees],
            "samples": [{"votes": [round(float(v), 3) for v in votes[i]], "score": float(forest[i]),
                         "truth": int(y[i])} for i in range(len(y))],
            "top_features": [{"name": features[j], "importance": float(est.feature_importances_[j])} for j in top]}


def _isolation_forest(est: Any, wrapper: Any, features: list[str], X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """PCA positions, real scores and average path lengths of real readings.

    Args:
        est: Fitted IsolationForest.
        wrapper: Model wrapper.
        features: Feature names.
        X: Sample features (stratified head plus a random cloud).
        y: Sample labels.

    Returns:
        Digest with ``points`` and the model's decision threshold.
    """
    raw = est.score_samples(X)
    psi = int(est.max_samples_)
    c = 2 * (math.log(psi - 1) + 0.5772156649) - 2 * (psi - 1) / psi
    depth = -np.log2(-raw) * c
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    _, _, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    P = Z @ vt[:3].T
    P = P / (np.abs(P).max(0) + 1e-9)
    pred = wrapper.predict(X)
    return {"kind": "isolation_forest", "threshold": float(wrapper.threshold), "max_samples": psi,
            "points": [{"p": [round(float(v), 4) for v in P[i]], "score": float(-raw[i]),
                        "depth": round(float(depth[i]), 2), "truth": int(y[i]), "pred": int(pred[i])}
                       for i in range(len(y))],
            "highlight": list(range(2 * SAMPLE_PER_CLASS))}


def _xgboost(est: Any, wrapper: Any, features: list[str], X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Real test log-loss and anomaly recall after increasing numbers of boosting rounds.

    Args:
        est: Fitted XGBClassifier.
        wrapper: Model wrapper.
        features: Feature names.
        X: Sample features.
        y: Sample labels.

    Returns:
        Digest with one entry per checkpoint round.
    """
    total = int(est.get_booster().num_boosted_rounds())
    rounds = [r for r in XGB_ROUNDS if r <= total] or [total]
    curve = []
    for r in rounds:
        proba = np.clip(est.predict_proba(X, iteration_range=(0, r))[:, 1], 1e-7, 1 - 1e-7)
        loss = float(-np.mean(y * np.log(proba) + (1 - y) * np.log(1 - proba)))
        curve.append({"round": r, "logloss": loss, "anomaly_mean": float(proba[y == 1].mean()),
                      "normal_mean": float(proba[y == 0].mean())})
    gains = est.get_booster().get_score(importance_type="gain")
    named = sorted(((features[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k, v)
                    for k, v in gains.items()), key=lambda kv: -kv[1])[:5]
    return {"kind": "xgboost", "rounds_total": total, "curve": curve,
            "top_features": [{"name": n, "gain": float(g)} for n, g in named],
            "scale_pos_weight": float(getattr(wrapper, "scale_pos_weight", 1.0))}


def load_cached(run_id: str, signature: str | None) -> dict[str, Any] | None:
    """Return a cached digest if it belongs to the same split signature.

    Args:
        run_id: Run identifier.
        signature: Current split signature of the run.

    Returns:
        The digest or None.
    """
    path = digest_path(run_id)
    if not path.is_file():
        return None
    digest = json.loads(path.read_text(encoding="utf-8"))
    return digest if digest.get("signature") == signature else None


def run_signature(run_id: str) -> str | None:
    """Read the split signature stored in a run's metrics file.

    Args:
        run_id: Run identifier.

    Returns:
        The signature, or None if the run has no metrics.
    """
    path = ROOT / "outputs" / "metrics" / f"{run_id}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("signature")

