"""Load and validate ``config/config.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

KNOWN_MODELS = ("random_forest", "isolation_forest", "decision_tree", "xgboost")
SPLIT_MODES = ("stratified_random", "by_building", "temporal")
REQUIRED_SECTIONS = ("paths", "data", "features", "split", "models", "evaluation", "plots")


class ConfigError(ValueError):
    """Raised when the configuration file is missing or invalid."""


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Read the YAML config and validate it.

    Args:
        path: Optional path to a config file. Defaults to ``config/config.yaml``.

    Returns:
        The parsed configuration dictionary.

    Raises:
        ConfigError: If the file is missing or any setting is invalid.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Check that the configuration has every required, well-formed setting.

    Args:
        config: Parsed configuration dictionary.

    Raises:
        ConfigError: On the first problem found, with a message naming the setting.
    """
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ConfigError(f"Config is missing sections: {', '.join(missing)}")
    if not isinstance(config.get("seed"), int):
        raise ConfigError("'seed' must be an integer.")
    _validate_split(config["split"])
    _validate_models(config["models"], config["plots"])
    fraction = config.get("sample_fraction")
    if fraction is not None and not 0 < float(fraction) <= 1:
        raise ConfigError("'sample_fraction' must be in (0, 1] or null.")


def _validate_split(split: dict[str, Any]) -> None:
    """Validate the ``split`` section.

    Args:
        split: The ``split`` config section.

    Raises:
        ConfigError: If the mode or ratios are invalid.
    """
    if split.get("mode") not in SPLIT_MODES:
        raise ConfigError(f"'split.mode' must be one of {SPLIT_MODES}.")
    ratios = split.get("ratios") or {}
    if not ratios:
        raise ConfigError("'split.ratios' must define at least one split.")
    for name, ratio in ratios.items():
        if not 0 < float(ratio) < 1:
            raise ConfigError(f"Split '{name}' ratio must be strictly between 0 and 1.")


def _validate_models(models: dict[str, Any], plots: dict[str, Any]) -> None:
    """Validate the ``models`` section and the per-model palette.

    Args:
        models: The ``models`` config section.
        plots: The ``plots`` config section.

    Raises:
        ConfigError: If a model is unknown or has no palette colour.
    """
    unknown = [name for name in models if name not in KNOWN_MODELS]
    if unknown:
        raise ConfigError(f"Unknown models in config: {unknown}. Known: {KNOWN_MODELS}.")
    palette = plots.get("palette") or {}
    no_colour = [name for name in models if name not in palette]
    if no_colour:
        raise ConfigError(f"'plots.palette' has no colour for: {no_colour}.")
    contamination = models.get("isolation_forest", {}).get("contamination", "train_rate")
    if contamination != "train_rate" and not 0 < float(contamination) <= 0.5:
        raise ConfigError("'isolation_forest.contamination' must be 'train_rate' or in (0, 0.5].")
