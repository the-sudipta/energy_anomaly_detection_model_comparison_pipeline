"""Command-line entry point: ``python -m src.main --stage <stage ...>``."""

from __future__ import annotations

import argparse
import sys
import time

from src.pipeline.stages import STAGE_FUNCTIONS, STAGES, RunOptions, StageError
from src.utils.config_loader import ConfigError, load_config
from src.utils.logger import setup_logging
from src.utils.paths import build_paths
from src.utils.timer import format_seconds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Energy anomaly detection: model x split benchmark")
    parser.add_argument("--stage", nargs="+", default=["all"], choices=["all", *STAGES],
                        help="Stage(s) to run, in pipeline order. Default: all.")
    parser.add_argument("--models", nargs="+", help="Only run these models (train_eval).")
    parser.add_argument("--splits", nargs="+", help="Only run these splits (train_eval).")
    parser.add_argument("--sample", type=float, help="Stratified fraction of rows, e.g. 0.05.")
    parser.add_argument("--force", action="store_true", help="Redo work even if outputs exist.")
    parser.add_argument("--config", help="Path to an alternative config.yaml.")
    return parser.parse_args(argv)


def resolve_options(args: argparse.Namespace, config: dict) -> RunOptions:
    """Combine CLI arguments with config defaults and validate model/split names.

    Args:
        args: Parsed CLI arguments.
        config: Parsed configuration dictionary.

    Returns:
        The effective run options.

    Raises:
        ConfigError: If an unknown model or split is requested.
    """
    models = args.models or list(config["models"])
    splits = args.splits or list(config["split"]["ratios"])
    unknown = [m for m in models if m not in config["models"]]
    unknown += [s for s in splits if s not in config["split"]["ratios"]]
    if unknown:
        raise ConfigError(f"Unknown model/split name(s): {unknown}. Check config.yaml.")
    sample = args.sample if args.sample is not None else config.get("sample_fraction")
    if sample is not None and not 0 < sample <= 1:
        raise ConfigError("--sample must be in (0, 1].")
    return RunOptions(models=models, splits=splits, sample=None if sample == 1 else sample, force=args.force)


def main(argv: list[str] | None = None) -> int:
    """Run the requested stages in pipeline order.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code (0 on success).
    """
    args = parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    paths = build_paths(config)
    log = setup_logging(paths.logs)
    stages = list(STAGES) if "all" in args.stage else [s for s in STAGES if s in args.stage]
    start = time.perf_counter()
    try:
        options = resolve_options(args, config)
        for stage in stages:
            STAGE_FUNCTIONS[stage](config, paths, options)
    except (StageError, ConfigError, FileNotFoundError, ValueError) as error:
        log.error("%s", error)
        log.debug("Traceback:", exc_info=True)
        return 1
    log.info("Done: %s in %s", ", ".join(stages), format_seconds(time.perf_counter() - start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
