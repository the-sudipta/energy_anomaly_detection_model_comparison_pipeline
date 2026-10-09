"""Exact, replayable traces of fitted models on real readings for the 3D view.

For a finished (split, model) run this loads the saved model bundle and a
seeded sample of that split's real test readings (half true anomalies), then
records exactly what the model computes for each reading:

* decision_tree    - every question on the reading's real root-to-leaf path
                     (feature, threshold, the reading's value, direction) and
                     the leaf score.
* random_forest    - for nine real trees: the reading's path depth, the root
                     question and the tree's vote; plus the full-forest score.
* isolation_forest - the reading's real path through one isolation tree (each
                     random cut and which sample readings survive it), its
                     average path length over all trees and the score.
* xgboost          - the reading's log-odds after 1, 2, 5, ... 400 trees.

Each reading also carries its real (imputed) feature values for hover.
Results are cached as JSON in ``outputs/digests``.
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
DIGEST_VERSION = 2
SAMPLE_PER_CLASS = 12
IF_CLOUD = 180
RF_TREES = 9
SHOW_FEATURES = 8
XGB_ROUNDS = (1, 2, 5, 10, 25, 50, 100, 200, 400)
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
    """Decide whether building a trace now is safe for the running pipeline.

    Args:
        model_path: Saved model bundle.
        training_active: Whether the train_eval stage is still running.

    Returns:
        ``(allowed, reason_if_not)``.
    """
    if training_active and model_path.stat().st_size > HEAVY_MODEL_BYTES:
        return False, "This forest is ~100 MB; it is loaded once training finishes so the pipeline keeps its memory."
    free = free_memory_bytes()
    if training_active and free is not None and free < MIN_FREE_BYTES:
        return False, f"Waiting for free memory ({free / 1e9:.1f} GB free) so the pipeline is not slowed."
    return True, ""


def digest_path(run_id: str) -> Path:
    """Return the cache file of a run's trace."""
    return DIGEST_DIR / f"{run_id}.json"


def build_digest(split: str, model: str, signature: str | None) -> dict[str, Any]:
    """Build and cache the trace of one finished run.

    Args:
        split: Split name.
        model: Model name.
        signature: Split signature stored with the run's metrics.

    Returns:
        The trace dictionary.
    """
    run_id = f"{split}__{model}"
    bundle = joblib.load(ROOT / "outputs" / "models" / f"{run_id}.joblib")
    features, X, y = _sample(split, model, bundle["imputer"])
    wrapper = bundle["model"]
    builders = {"decision_tree": _decision_tree, "random_forest": _random_forest,
                "isolation_forest": _isolation_forest, "xgboost": _xgboost}
    digest = builders[model](wrapper.estimator, wrapper, features, X, y)
    digest.update(run=run_id, split=split, model=model, signature=signature, version=DIGEST_VERSION)
    DIGEST_DIR.mkdir(parents=True, exist_ok=True)
    digest_path(run_id).write_text(json.dumps(digest), encoding="utf-8")
    return digest


def _sample(split: str, model: str, imputer: Any) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Draw a seeded sample of real test readings and impute them exactly like the run.

    The first ``2 * SAMPLE_PER_CLASS`` rows alternate anomalies and normal
    readings; Isolation Forest additionally gets a random cloud of test rows.

    Args:
        split: Split name.
        model: Model name.
        imputer: The fitted imputation pipeline saved with the run.

    Returns:
        ``(feature_names, X_sample, y_sample)``.
    """
    meta = json.loads((ROOT / "data" / "processed" / "dataset_meta.json").read_text(encoding="utf-8"))
    features = meta["feature_columns"]
    dataset = ROOT / "data" / "processed" / "dataset.parquet"
    with np.load(ROOT / "data" / "processed" / "splits" / f"{split}.npz") as bundle:
        test_idx = bundle["test_idx"]
    y_test = pq.read_table(dataset, columns=["anomaly"]).column("anomaly").to_numpy()[test_idx]
    rng = np.random.default_rng(42)
    pos = rng.choice(test_idx[y_test == 1], SAMPLE_PER_CLASS, replace=False)
    neg = rng.choice(test_idx[y_test == 0], SAMPLE_PER_CLASS, replace=False)
    head = np.empty(2 * SAMPLE_PER_CLASS, dtype=np.int64)
    head[0::2], head[1::2] = pos, neg
    rows = [head]
    if model == "isolation_forest":
        rest = np.setdiff1d(test_idx, head)
        rows.append(rng.choice(rest, IF_CLOUD, replace=False))
    rows_all = np.concatenate(rows)
    frame = _read_rows(dataset, rows_all, [*features, "anomaly"])
    y = frame.pop("anomaly").to_numpy().astype(int)
    return features, imputer.transform(frame).astype(np.float32), y


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


def _values(x: np.ndarray, columns: list[int], features: list[str]) -> dict[str, float]:
    """Real feature values of one reading for the hover panel.

    Args:
        x: One imputed feature row.
        columns: Feature positions to include.
        features: Feature names.

    Returns:
        Mapping of feature name to rounded value.
    """
    return {features[j]: round(float(x[j]), 4) for j in columns}


def _path_steps(tree: Any, node_ids: list[int], x: np.ndarray, features: list[str],
                columns: np.ndarray | None = None) -> list[dict[str, Any]]:
    """Turn a decision path into the questions asked and the answers given.

    Args:
        tree: A fitted sklearn ``tree_`` object.
        node_ids: Node ids on the path, root first.
        x: The reading's features in the tree's own column order.
        features: Feature names.
        columns: Mapping from the tree's columns to ``features`` (Isolation Forest).

    Returns:
        One step per internal node: feature, threshold, value and direction.
    """
    steps = []
    for node, nxt in zip(node_ids[:-1], node_ids[1:]):
        col = int(tree.feature[node])
        name = features[int(columns[col])] if columns is not None else features[col]
        steps.append({"f": name, "t": round(float(tree.threshold[node]), 4), "v": round(float(x[col]), 4),
                      "left": bool(nxt == tree.children_left[node]), "n": int(tree.n_node_samples[node])})
    return steps


def _node_path(est: Any, x: np.ndarray) -> list[int]:
    """Return the node ids visited by one reading, root first."""
    path = est.decision_path(x.reshape(1, -1))
    return sorted(path.indices.tolist())


def _share(value: np.ndarray) -> float:
    """Anomaly share of a node from sklearn's per-class value array."""
    counts = np.asarray(value).ravel()
    total = counts.sum()
    return float(counts[1] / total) if total and len(counts) > 1 else 0.0


def _top(importances: np.ndarray) -> list[int]:
    """Indices of the most important features."""
    return [int(j) for j in np.argsort(importances)[::-1][:SHOW_FEATURES]]


def _decision_tree(est: Any, wrapper: Any, features: list[str], X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Full real root-to-leaf question sequence for every sample reading.

    Args:
        est: Fitted DecisionTreeClassifier.
        wrapper: Model wrapper.
        features: Feature names.
        X: Sample features.
        y: Sample labels.

    Returns:
        Trace with the top three tree levels and one path per reading.
    """
    tree, show = est.tree_, _top(est.feature_importances_)
    top_nodes, frontier = [], [(0, 0)]
    while frontier:
        node, slot = frontier.pop(0)
        depth = int(math.log2(slot + 1))
        leaf = tree.children_left[node] == -1
        top_nodes.append({"slot": slot, "depth": depth, "leaf": bool(leaf),
                          "f": None if leaf else features[tree.feature[node]],
                          "t": None if leaf else round(float(tree.threshold[node]), 4),
                          "n": int(tree.n_node_samples[node]), "share": _share(tree.value[node])})
        if not leaf and depth < 2:
            frontier += [(int(tree.children_left[node]), 2 * slot + 1), (int(tree.children_right[node]), 2 * slot + 2)]
    scores = wrapper.predict_score(X)
    samples = []
    for i in range(len(y)):
        nodes = _node_path(est, X[i])
        leaf = nodes[-1]
        samples.append({"truth": int(y[i]), "score": float(scores[i]), "values": _values(X[i], show, features),
                        "steps": _path_steps(tree, nodes, X[i], features),
                        "leaf": {"n": int(tree.n_node_samples[leaf]), "share": _share(tree.value[leaf])}})
    return {"kind": "decision_tree", "top": top_nodes, "samples": samples,
            "tree_depth": int(est.get_depth()), "n_leaves": int(est.get_n_leaves())}


def _random_forest(est: Any, wrapper: Any, features: list[str], X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Per-tree real path depth, root question and vote for each sample reading.

    Args:
        est: Fitted RandomForestClassifier.
        wrapper: Model wrapper.
        features: Feature names.
        X: Sample features.
        y: Sample labels.

    Returns:
        Trace with nine trees and per-reading votes.
    """
    trees, show = est.estimators_[:RF_TREES], _top(est.feature_importances_)
    forest = wrapper.predict_score(X)
    info = [{"depth": int(t.get_depth()), "leaves": int(t.get_n_leaves()),
             "root": {"f": features[t.tree_.feature[0]], "t": round(float(t.tree_.threshold[0]), 4)}} for t in trees]
    samples = []
    for i in range(len(y)):
        per_tree = []
        for t in trees:
            nodes = _node_path(t, X[i])
            steps = _path_steps(t.tree_, nodes, X[i], features)
            per_tree.append({"vote": round(float(t.predict_proba(X[i:i + 1])[0, -1] if t.n_classes_ > 1 else 0.0), 3),
                             "depth": len(steps), "first": steps[:3]})
        samples.append({"truth": int(y[i]), "score": float(forest[i]), "values": _values(X[i], show, features),
                        "trees": per_tree})
    return {"kind": "random_forest", "n_trees": len(est.estimators_), "trees": info, "samples": samples,
            "top_features": [features[j] for j in show[:5]]}


def _isolation_forest(est: Any, wrapper: Any, features: list[str], X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Real cut-by-cut isolation of each sample reading in one isolation tree.

    For every cut on the reading's path, records which displayed readings are
    still on the same side, so the shrinking cloud is exact.

    Args:
        est: Fitted IsolationForest.
        wrapper: Model wrapper.
        features: Feature names.
        X: Sample features (stratified head followed by a random cloud).
        y: Sample labels.

    Returns:
        Trace with point positions and one isolation path per sample reading.
    """
    raw = est.score_samples(X)
    psi = int(est.max_samples_)
    c = 2 * (math.log(psi - 1) + 0.5772156649) - 2 * (psi - 1) / psi
    avg_depth = -np.log2(-raw) * c
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    _, _, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    P = Z @ vt[:3].T
    P = P / (np.abs(P).max(0) + 1e-9)
    tree, cols = est.estimators_[0], est.estimators_features_[0]
    Xt = X[:, cols]
    leaf_of_all = tree.apply(Xt)
    pred = wrapper.predict(X)
    head = 2 * SAMPLE_PER_CLASS
    samples = []
    for i in range(head):
        nodes = _node_path(tree, Xt[i])
        steps = _path_steps(tree.tree_, nodes, Xt[i], features, cols)
        alive = np.arange(len(X))
        for step, node in zip(steps, nodes[:-1]):
            col = int(tree.tree_.feature[node])
            keep = (Xt[alive, col] <= tree.tree_.threshold[node]) == step["left"]
            alive = alive[keep]
            step["alive"] = alive.tolist() if len(alive) <= 60 else len(alive)
        used = sorted({features.index(s["f"]) for s in steps} | {features.index("log_meter_reading")}
                      if "log_meter_reading" in features else {features.index(s["f"]) for s in steps})
        samples.append({"i": i, "truth": int(y[i]), "score": float(-raw[i]), "pred": int(pred[i]),
                        "avg_depth": round(float(avg_depth[i]), 2), "steps": steps,
                        "shared_leaf": int(np.sum(leaf_of_all == leaf_of_all[i]) - 1),
                        "values": _values(X[i], used[:SHOW_FEATURES], features)})
    return {"kind": "isolation_forest", "threshold": float(wrapper.threshold), "max_samples": psi,
            "n_trees": len(est.estimators_),
            "points": [{"p": [round(float(v), 4) for v in P[i]], "truth": int(y[i]), "score": round(float(-raw[i]), 4)}
                       for i in range(len(y))],
            "samples": samples}


def _xgboost(est: Any, wrapper: Any, features: list[str], X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Each reading's real log-odds after increasing numbers of boosted trees.

    Args:
        est: Fitted XGBClassifier.
        wrapper: Model wrapper.
        features: Feature names.
        X: Sample features.
        y: Sample labels.

    Returns:
        Trace with checkpoint rounds and one margin trajectory per reading.
    """
    total = int(est.get_booster().num_boosted_rounds())
    rounds = [r for r in XGB_ROUNDS if r <= total] or [total]
    margins = np.stack([est.predict(X, iteration_range=(0, r), output_margin=True) for r in rounds], axis=1)
    gains = est.get_booster().get_score(importance_type="gain")
    ranked = sorted(gains.items(), key=lambda kv: -kv[1])
    show = [int(k[1:]) for k, _ in ranked if k.startswith("f") and k[1:].isdigit()][:SHOW_FEATURES]
    named = [features[j] for j in show[:5]] if show else [k for k, _ in ranked[:5]]
    proba = 1 / (1 + np.exp(-margins[:, -1]))
    return {"kind": "xgboost", "rounds": rounds, "rounds_total": total, "learning_rate": float(est.learning_rate or 0.3),
            "scale_pos_weight": float(getattr(wrapper, "scale_pos_weight", 1.0)), "top_features": named,
            "samples": [{"truth": int(y[i]), "margins": [round(float(m), 4) for m in margins[i]],
                         "score": float(proba[i]), "values": _values(X[i], show, features) if show else {}}
                        for i in range(len(y))]}


def load_cached(run_id: str, signature: str | None) -> dict[str, Any] | None:
    """Return a cached trace if it matches the run's split and the current format.

    Args:
        run_id: Run identifier.
        signature: Current split signature of the run.

    Returns:
        The trace or None.
    """
    path = digest_path(run_id)
    if not path.is_file():
        return None
    digest = json.loads(path.read_text(encoding="utf-8"))
    ok = digest.get("signature") == signature and digest.get("version") == DIGEST_VERSION
    return digest if ok else None


def latest_cached(model: str) -> dict[str, Any] | None:
    """Return the newest cached trace of a model from any earlier run.

    Used while the current pipeline has not finished a run of that model yet.

    Args:
        model: Model name.

    Returns:
        The trace, flagged with ``previous=True``, or None.
    """
    if not DIGEST_DIR.is_dir():
        return None
    for path in sorted(DIGEST_DIR.glob(f"*__{model}.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        digest = json.loads(path.read_text(encoding="utf-8"))
        if digest.get("version") == DIGEST_VERSION:
            digest["previous"] = True
            return digest
    return None


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
