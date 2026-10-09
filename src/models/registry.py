"""Map model names to wrapper classes."""

from __future__ import annotations

from typing import Any

from src.models.base import BaseAnomalyModel
from src.models.decision_tree import DecisionTreeModel
from src.models.isolation_forest import IsolationForestModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel

MODEL_CLASSES: dict[str, type[BaseAnomalyModel]] = {
    "random_forest": RandomForestModel,
    "isolation_forest": IsolationForestModel,
    "decision_tree": DecisionTreeModel,
    "xgboost": XGBoostModel,
}


def build_model(name: str, config: dict[str, Any], split: str | None = None,
                overrides: dict[str, Any] | None = None) -> BaseAnomalyModel:
    """Instantiate a model wrapper from its name and the config.

    Hyperparameters come from ``config["models"][name]``, updated with the
    tuned values for ``split`` (if the tune stage saved any) and then with
    explicit ``overrides`` (used during the search itself).

    Args:
        name: Model name, a key of ``MODEL_CLASSES``.
        config: Parsed configuration dictionary.
        split: Split whose tuned hyperparameters should be applied.
        overrides: Hyperparameters that take precedence over everything else.

    Returns:
        An unfitted model wrapper.

    Raises:
        KeyError: If the model name is unknown.
    """
    if name not in MODEL_CLASSES:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_CLASSES)}")
    params = dict(config["models"].get(name) or {})
    if split:
        params.update(config.get("_tuned", {}).get(split, {}).get(name, {}))
    params.update(overrides or {})
    seed, n_jobs = config["seed"], config.get("n_jobs", -1)
    if name == "xgboost":
        return XGBoostModel(params, seed, n_jobs, use_gpu=bool(config.get("use_gpu", False)))
    return MODEL_CLASSES[name](params, seed, n_jobs)
