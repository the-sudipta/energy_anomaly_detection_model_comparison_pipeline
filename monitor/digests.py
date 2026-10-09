"""Exact traces of fitted models, for the step-by-step 3D explanations.

For a finished (split, model) run this loads the saved model bundle, a
balanced sample of that split's real training readings and a few real test
readings, and records what the model really did with them:

* decision_tree    - the real top levels of the tree, which training readings
                     each split sends left or right, and every question a test
                     reading answers on its way to a leaf.
* random_forest    - for six real trees: their real bootstrap draw of the
                     shown training readings (replayed from each tree's seed),
                     their top splits, and each test reading's path and vote.
* isolation_forest - the real subsample tree #1 was built from, its top cuts,
                     and the cut-by-cut isolation of each test reading.
* xgboost          - the training readings' predictions after each boosting
                     checkpoint, the first real trees from the booster dump,
                     and each test reading's per-tree leaf values.

Results are cached as JSON in ``outputs/digests``.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
# Overridable so animation work can use scratch models without touching real outputs.
DIGEST_DIR = Path(os.environ.get("MONITOR_DIGEST_DIR", ROOT / "outputs" / "digests"))
MODELS_DIR = Path(os.environ.get("MONITOR_MODELS_DIR", ROOT / "outputs" / "models"))
DIGEST_VERSION = 3
TRAIN_PER_CLASS = 120
TEST_PER_CLASS = 6
TOP_DEPTH = 4
RF_TREES = 6
XGB_TREES_SHOWN = 4
XGB_ROUNDS = (1, 2, 3, 4, 10, 25, 50, 100, 200, 400, 600, 1000)
SHOW_FEATURES = 8
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
        training_active: Whether training or tuning is running.

    Returns:
        ``(allowed, reason_if_not)``.
    """
    if training_active and model_path.stat().st_size > HEAVY_MODEL_BYTES:
        return False, "This forest is large; it is loaded once training finishes so the pipeline keeps its memory."
    free = free_memory_bytes()
    if training_active and free is not None and free < MIN_FREE_BYTES:
        return False, f"Waiting for free memory ({free / 1e9:.1f} GB free) so the pipeline is not slowed."
    return True, ""


def digest_path(run_id: str) -> Path:
    """Return the cache file of a run's trace."""
    return DIGEST_DIR / f"{run_id}.json"


# ----------------------------------------------------------------------------- data access


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


class Sample:
    """Real readings of one split, imputed exactly like the run imputed them."""

    def __init__(self, split: str, imputer: Any) -> None:
        """Load the split, the label column and the feature list.

        Args:
            split: Split name.
            imputer: Fitted imputation pipeline saved with the run.
        """
        meta = json.loads((ROOT / "data" / "processed" / "dataset_meta.json").read_text(encoding="utf-8"))
        self.features: list[str] = meta["feature_columns"]
        self.dataset = ROOT / "data" / "processed" / "dataset.parquet"
        with np.load(ROOT / "data" / "processed" / "splits" / f"{split}.npz") as bundle:
            self.train_idx, self.test_idx = bundle["train_idx"], bundle["test_idx"]
        self.y_all = pq.read_table(self.dataset, columns=["anomaly"]).column("anomaly").to_numpy()
        self.imputer = imputer
        self.rng = np.random.default_rng(42)

    def balanced(self, pool: np.ndarray, per_class: int) -> np.ndarray:
        """Pick ``per_class`` anomalies and normal readings from a pool, alternating.

        Args:
            pool: Dataset row positions to draw from.
            per_class: Readings per class.

        Returns:
            Dataset row positions.
        """
        y = self.y_all[pool]
        pos = self.rng.choice(pool[y == 1], min(per_class, int((y == 1).sum())), replace=False)
        neg = self.rng.choice(pool[y == 0], min(per_class, int((y == 0).sum())), replace=False)
        rows = np.empty(len(pos) + len(neg), dtype=np.int64)
        rows[0::2][: len(pos)], rows[1::2][: len(neg)] = pos, neg
        return rows

    def load(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return imputed features and labels of dataset rows.

        Args:
            rows: Dataset row positions.

        Returns:
            ``(X, y)``.
        """
        frame = _read_rows(self.dataset, rows, [*self.features, "anomaly"])
        y = frame.pop("anomaly").to_numpy().astype(int)
        return self.imputer.transform(frame).astype(np.float32), y

    def train_position(self, rows: np.ndarray) -> np.ndarray:
        """Position of dataset rows inside this split's (sorted) train portion."""
        return np.searchsorted(self.train_idx, rows)


def _values(x: np.ndarray, columns: list[int], features: list[str]) -> dict[str, float]:
    """Real feature values of one reading for hover panels."""
    return {features[j]: round(float(x[j]), 4) for j in columns}


def _share(value: np.ndarray) -> float:
    """Anomaly share of a node from sklearn's per-class value array."""
    counts = np.asarray(value).ravel()
    total = counts.sum()
    return float(counts[1] / total) if total and len(counts) > 1 else 0.0


def _top_features(importances: np.ndarray) -> list[int]:
    """Indices of the most important features."""
    return [int(j) for j in np.argsort(importances)[::-1][:SHOW_FEATURES]]


# ----------------------------------------------------------------------------- sklearn trees


def tree_top(tree: Any, names: list[str], depth: int, labelled: bool = True) -> tuple[list[dict], dict[int, int]]:
    """Describe the top levels of a fitted sklearn tree in heap order.

    Args:
        tree: A fitted ``tree_`` object.
        names: Name of every column the tree was fitted on.
        depth: Number of levels to keep.
        labelled: Whether node values hold class counts (False for isolation trees).

    Returns:
        ``(nodes, node_id_to_slot)``; slot 0 is the root, children of s are 2s+1 and 2s+2.
    """
    nodes, slot_of, frontier = [], {}, [(0, 0)]
    while frontier:
        node, slot = frontier.pop(0)
        level = int(math.log2(slot + 1))
        leaf = tree.children_left[node] == -1
        slot_of[node] = slot
        nodes.append({"slot": slot, "depth": level, "leaf": bool(leaf), "cut": bool(leaf or level == depth - 1),
                      "f": None if leaf else names[tree.feature[node]],
                      "t": None if leaf else round(float(tree.threshold[node]), 4),
                      "n": int(tree.n_node_samples[node]),
                      "share": _share(tree.value[node]) if labelled else None})
        if not leaf and level < depth - 1:
            frontier += [(int(tree.children_left[node]), 2 * slot + 1), (int(tree.children_right[node]), 2 * slot + 2)]
    return nodes, slot_of


def path_steps(tree: Any, X: np.ndarray, names: list[str]) -> list[list[dict[str, Any]]]:
    """Every question each reading answers on its root-to-leaf path.

    Args:
        tree: A fitted sklearn estimator exposing ``decision_path``.
        X: Readings in the estimator's own column order.
        names: Column names.

    Returns:
        Per reading: one step per internal node (node id, feature, threshold, value, left?).
    """
    inner, indicator = tree.tree_, tree.decision_path(X)
    result = []
    for i in range(X.shape[0]):
        ids = sorted(indicator.indices[indicator.indptr[i]:indicator.indptr[i + 1]].tolist())
        steps = []
        for node, nxt in zip(ids[:-1], ids[1:]):
            col = int(inner.feature[node])
            steps.append({"id": int(node), "f": names[col], "t": round(float(inner.threshold[node]), 4),
                          "v": round(float(X[i, col]), 4), "left": bool(nxt == inner.children_left[node]),
                          "n": int(inner.n_node_samples[node])})
        result.append(steps + [{"id": int(ids[-1]), "leaf": True, "n": int(inner.n_node_samples[ids[-1]])}])
    return result


def slots_on_path(steps: list[dict[str, Any]], slot_of: dict[int, int]) -> list[int]:
    """Heap slots of the path nodes that fall inside the drawn top levels."""
    return [slot_of[s["id"]] for s in steps if s["id"] in slot_of]


# ----------------------------------------------------------------------------- builders


def _decision_tree(est: Any, wrapper: Any, sample: Sample) -> dict[str, Any]:
    """Trace of the fitted decision tree on real training and test readings."""
    features = sample.features
    train_rows = sample.balanced(sample.train_idx, TRAIN_PER_CLASS)
    X_tr, y_tr = sample.load(train_rows)
    test_rows = sample.balanced(sample.test_idx, TEST_PER_CLASS)
    X_te, y_te = sample.load(test_rows)
    top, slot_of = tree_top(est.tree_, features, TOP_DEPTH)
    show = _top_features(est.feature_importances_)
    train_paths = path_steps(est, X_tr, features)
    scores = wrapper.predict_score(X_te)
    tests = []
    for i, steps in enumerate(path_steps(est, X_te, features)):
        leaf = steps[-1]["id"]
        tests.append({"truth": int(y_te[i]), "score": float(scores[i]), "pred": int(scores[i] >= wrapper.threshold),
                      "values": _values(X_te[i], show, features), "steps": steps,
                      "slots": slots_on_path(steps, slot_of),
                      "leaf": {"n": int(est.tree_.n_node_samples[leaf]), "share": _share(est.tree_.value[leaf])}})
    return {"kind": "decision_tree", "top": top, "tree_depth": int(est.get_depth()), "n_leaves": int(est.get_n_leaves()),
            "train": [{"truth": int(y_tr[i]), "slots": slots_on_path(p, slot_of),
                       "v": {f: round(float(X_tr[i, features.index(f)]), 4) for f in {n["f"] for n in top if n["f"]}}}
                      for i, p in enumerate(train_paths)],
            "tests": tests, "criterion": est.criterion, "min_samples_leaf": est.min_samples_leaf}


def _bootstrap_counts(tree: Any, n_train: int, positions: np.ndarray, max_samples: Any) -> list[int]:
    """How often each shown training reading was drawn into a tree's bootstrap sample.

    Replays scikit-learn's own draw from the tree's stored random state.

    Args:
        tree: One fitted tree of the forest.
        n_train: Rows in the forest's training set.
        positions: Positions of the shown readings in that training set.
        max_samples: The forest's ``max_samples`` setting.

    Returns:
        A count per shown reading (empty if the replay is unavailable).
    """
    try:
        from sklearn.ensemble._forest import _generate_sample_indices, _get_n_samples_bootstrap

        n_boot = _get_n_samples_bootstrap(n_train, max_samples, None)
        drawn = _generate_sample_indices(tree.random_state, n_train, n_boot, None)
        return np.bincount(drawn, minlength=n_train)[positions].astype(int).tolist()
    except (ImportError, TypeError):
        return []


def _random_forest(est: Any, wrapper: Any, sample: Sample) -> dict[str, Any]:
    """Trace of six real trees of the forest on real training and test readings."""
    features = sample.features
    train_rows = sample.balanced(sample.train_idx, TRAIN_PER_CLASS // 2)
    X_tr, y_tr = sample.load(train_rows)
    test_rows = sample.balanced(sample.test_idx, TEST_PER_CLASS)
    X_te, y_te = sample.load(test_rows)
    positions = sample.train_position(train_rows)
    trees, show = est.estimators_[:RF_TREES], _top_features(est.feature_importances_)
    forest_scores = wrapper.predict_score(X_te)
    drawn = []
    for t in trees:
        top, slot_of = tree_top(t.tree_, features, 3)
        drawn.append({"top": top, "depth": int(t.get_depth()), "leaves": int(t.get_n_leaves()),
                      "boot": _bootstrap_counts(t, len(sample.train_idx), positions, est.max_samples),
                      "slot_of": slot_of})
    tests = []
    for i in range(len(y_te)):
        per_tree = []
        for t, info in zip(trees, drawn):
            steps = path_steps(t, X_te[i:i + 1], features)[0]
            per_tree.append({"slots": slots_on_path(steps, info["slot_of"]), "depth": len(steps) - 1,
                             "vote": round(float(t.predict_proba(X_te[i:i + 1])[0, -1]), 4),
                             "first": steps[:3]})
        tests.append({"truth": int(y_te[i]), "score": float(forest_scores[i]),
                      "pred": int(forest_scores[i] >= wrapper.threshold),
                      "values": _values(X_te[i], show, features), "trees": per_tree})
    for info in drawn:
        info.pop("slot_of")
    return {"kind": "random_forest", "n_trees": len(est.estimators_), "max_features": str(est.max_features),
            "n_train_total": int(len(sample.train_idx)),
            "train": [{"truth": int(y_tr[i])} for i in range(len(y_tr))], "trees": drawn, "tests": tests}


def _isolation_forest(est: Any, wrapper: Any, sample: Sample) -> dict[str, Any]:
    """Trace of isolation tree #1: its real subsample, its cuts, and test readings' isolation."""
    features = sample.features
    view = list(range(len(features))) if wrapper.columns is None else [int(c) for c in wrapper.columns]
    tree, cols = est.estimators_[0], [view[int(c)] for c in est.estimators_features_[0]]
    names = [features[c] for c in cols]
    sub_pos = np.asarray(est.estimators_samples_[0])
    sub_rows = sample.train_idx[np.sort(sub_pos)][:600]
    X_sub, y_sub = sample.load(sub_rows)
    test_rows = sample.balanced(sample.test_idx, TEST_PER_CLASS)
    X_te, y_te = sample.load(test_rows)
    top, slot_of = tree_top(tree.tree_, names, TOP_DEPTH, labelled=False)
    sub_paths = path_steps(tree, X_sub[:, cols], names)
    raw = est.score_samples(X_te[:, view])
    psi = int(est.max_samples_)
    c = 2 * (math.log(psi - 1) + 0.5772156649) - 2 * (psi - 1) / psi
    avg_depth = -np.log2(-raw) * c
    used_feats = sorted({n["f"] for n in top if n["f"]})
    tests = []
    for i, steps in enumerate(path_steps(tree, X_te[:, cols], names)):
        tests.append({"truth": int(y_te[i]), "score": float(-raw[i]), "pred": int(-raw[i] >= wrapper.threshold),
                      "avg_depth": round(float(avg_depth[i]), 3), "steps": steps, "slots": slots_on_path(steps, slot_of),
                      "values": {f: round(float(X_te[i, features.index(f)]), 4) for f in used_feats[:SHOW_FEATURES]}})
    return {"kind": "isolation_forest", "n_trees": len(est.estimators_), "max_samples": psi, "c_n": round(c, 4),
            "threshold": float(wrapper.threshold), "feature_set": wrapper.feature_set, "n_features_used": len(cols),
            "top": top,
            "subsample": [{"truth": int(y_sub[i]), "slots": slots_on_path(p, slot_of), "depth": len(p) - 1,
                           "v": {f: round(float(X_sub[i, features.index(f)]), 4) for f in used_feats}}
                          for i, p in enumerate(sub_paths)],
            "tests": tests}


def _xgb_tree(dump: dict[str, Any], features: list[str], depth: int) -> list[dict[str, Any]]:
    """Top levels of one booster tree from its JSON dump, in heap order."""
    nodes, frontier = [], [(dump, 0)]
    while frontier:
        node, slot = frontier.pop(0)
        level = int(math.log2(slot + 1))
        if "leaf" in node:
            nodes.append({"slot": slot, "depth": level, "leaf": True, "value": round(float(node["leaf"]), 5), "id": node["nodeid"]})
            continue
        split = node["split"]
        name = features[int(split[1:])] if split.startswith("f") and split[1:].isdigit() else split
        nodes.append({"slot": slot, "depth": level, "leaf": False, "f": name, "t": round(float(node["split_condition"]), 4),
                      "id": node["nodeid"], "cut": level == depth - 1})
        if level < depth - 1:
            kids = {c["nodeid"]: c for c in node["children"]}
            frontier += [(kids[node["yes"]], 2 * slot + 1), (kids[node["no"]], 2 * slot + 2)]
    return nodes


def _leaf_values(dump: dict[str, Any]) -> dict[int, float]:
    """Map every leaf id of a dumped tree to its value."""
    values, stack = {}, [dump]
    while stack:
        node = stack.pop()
        if "leaf" in node:
            values[int(node["nodeid"])] = float(node["leaf"])
        else:
            stack += node["children"]
    return values


def _xgboost(est: Any, wrapper: Any, sample: Sample) -> dict[str, Any]:
    """Trace of the booster: training predictions per round, real first trees, per-tree leaf values."""
    import xgboost as xgb

    features = sample.features
    booster = est.get_booster()
    total = int(booster.num_boosted_rounds())
    rounds = [r for r in XGB_ROUNDS if r <= total] + ([total] if total not in XGB_ROUNDS else [])
    train_rows = sample.balanced(sample.train_idx, 60)
    X_tr, y_tr = sample.load(train_rows)
    test_rows = sample.balanced(sample.test_idx, TEST_PER_CLASS)
    X_te, y_te = sample.load(test_rows)
    staged_tr = np.stack([est.predict_proba(X_tr, iteration_range=(0, r))[:, 1] for r in rounds], axis=1)
    margins_te = np.stack([est.predict(X_te, iteration_range=(0, r), output_margin=True) for r in rounds], axis=1)
    dumps = [json.loads(d) for d in booster.get_dump(dump_format="json")[:XGB_TREES_SHOWN]]
    leaves = booster.predict(xgb.DMatrix(X_te), pred_leaf=True, iteration_range=(0, XGB_TREES_SHOWN))
    leaf_vals = [_leaf_values(d) for d in dumps]
    first_leaf = [leaf_vals[0][int(leaves[i][0])] for i in range(len(y_te))]
    base = float(np.mean(margins_te[:, 0] - np.array(first_leaf)))
    gains = booster.get_score(importance_type="gain")
    show = [int(k[1:]) for k, _ in sorted(gains.items(), key=lambda kv: -kv[1]) if k[1:].isdigit()][:SHOW_FEATURES]
    proba = 1 / (1 + np.exp(-margins_te[:, -1]))
    tests = []
    for i in range(len(y_te)):
        per_tree = [{"leaf": int(leaves[i][k]), "value": round(leaf_vals[k][int(leaves[i][k])], 5)} for k in range(len(dumps))]
        tests.append({"truth": int(y_te[i]), "score": float(proba[i]), "pred": int(proba[i] >= 0.5),
                      "margins": [round(float(m), 4) for m in margins_te[i]], "trees": per_tree,
                      "values": _values(X_te[i], show, features)})
    return {"kind": "xgboost", "rounds": rounds, "rounds_total": total, "base_margin": round(base, 5),
            "learning_rate": float(est.learning_rate or 0.3), "max_depth": int(est.max_depth or 6),
            "scale_pos_weight": float(getattr(wrapper, "scale_pos_weight", 1.0)),
            "trees": [_xgb_tree(d, features, 3) for d in dumps],
            "train": [{"truth": int(y_tr[i]), "p": [round(float(v), 4) for v in staged_tr[i]]} for i in range(len(y_tr))],
            "tests": tests}


BUILDERS = {"decision_tree": _decision_tree, "random_forest": _random_forest,
            "isolation_forest": _isolation_forest, "xgboost": _xgboost}


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
    bundle = joblib.load(MODELS_DIR / f"{run_id}.joblib")
    sample = Sample(split, bundle["imputer"])
    digest = BUILDERS[model](bundle["model"].estimator, bundle["model"], sample)
    digest.update(run=run_id, split=split, model=model, signature=signature, version=DIGEST_VERSION,
                  params=bundle["model"].params)
    DIGEST_DIR.mkdir(parents=True, exist_ok=True)
    digest_path(run_id).write_text(json.dumps(digest, default=str), encoding="utf-8")
    return digest


def load_cached(run_id: str, signature: str | None) -> dict[str, Any] | None:
    """Return a cached trace if it matches the run's split and the current format."""
    path = digest_path(run_id)
    if not path.is_file():
        return None
    digest = json.loads(path.read_text(encoding="utf-8"))
    ok = digest.get("signature") == signature and digest.get("version") == DIGEST_VERSION
    return digest if ok else None


def latest_cached(model: str) -> dict[str, Any] | None:
    """Newest cached trace of a model from any earlier run, flagged ``previous``."""
    if not DIGEST_DIR.is_dir():
        return None
    for path in sorted(DIGEST_DIR.glob(f"*__{model}.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        digest = json.loads(path.read_text(encoding="utf-8"))
        if digest.get("version") == DIGEST_VERSION:
            digest["previous"] = True
            return digest
    return None


def run_signature(run_id: str) -> str | None:
    """Read the split signature stored in a run's metrics file."""
    path = ROOT / "outputs" / "metrics" / f"{run_id}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("signature")
