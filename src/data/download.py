"""Download the Kaggle competition files into ``data/raw`` when they are missing."""

from __future__ import annotations

import os
import time
import zipfile
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

_log = get_logger(__name__)

MAX_ATTEMPTS = 3
REQUIRED_FILE = "train.csv"


class DownloadError(RuntimeError):
    """Raised when the dataset cannot be obtained automatically."""


def download_dataset(raw_dir: Path, competition: str, config_dir: Path) -> bool:
    """Download and extract the competition files unless ``train.csv`` already exists.

    Args:
        raw_dir: Destination folder for the raw CSV files.
        competition: Kaggle competition slug.
        config_dir: Project ``config`` folder, searched for a fallback ``kaggle.json``.

    Returns:
        True if the raw data is present afterwards, False if the user must
        download it manually (instructions are logged).
    """
    if (raw_dir / REQUIRED_FILE).is_file():
        _log.info("Raw data already present in %s, skipping download.", raw_dir)
        log_file_listing(raw_dir)
        return True
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        api = _authenticate(config_dir)
        _download_with_retries(api, competition, raw_dir)
        _extract_archives(raw_dir)
    except DownloadError as error:
        _log.error("%s", error)
        _log.error(manual_instructions(raw_dir, competition))
        return False
    if not (raw_dir / REQUIRED_FILE).is_file():
        _log.error("Download finished but %s is missing in %s.", REQUIRED_FILE, raw_dir)
        _log.error(manual_instructions(raw_dir, competition))
        return False
    log_file_listing(raw_dir)
    return True


def _authenticate(config_dir: Path) -> Any:
    """Authenticate against the Kaggle API using the documented credential order.

    Order: ``KAGGLE_USERNAME``/``KAGGLE_KEY`` or ``KAGGLE_API_TOKEN`` environment
    variables, then ``~/.kaggle/access_token``, then ``~/.kaggle/kaggle.json``,
    then ``config/kaggle.json``.

    Args:
        config_dir: Project ``config`` folder.

    Returns:
        An authenticated ``KaggleApi`` instance.

    Raises:
        DownloadError: If no credentials are found or authentication fails.
    """
    has_env = bool(os.environ.get("KAGGLE_API_TOKEN")) or (
        bool(os.environ.get("KAGGLE_USERNAME")) and bool(os.environ.get("KAGGLE_KEY"))
    )
    home_file = Path.home() / ".kaggle" / "kaggle.json"
    home_token = Path.home() / ".kaggle" / "access_token"
    project_file = config_dir / "kaggle.json"
    if not has_env and home_token.is_file():
        os.environ["KAGGLE_API_TOKEN"] = home_token.read_text(encoding="utf-8").strip()
        has_env = True
    if not has_env and not home_file.is_file():
        if not project_file.is_file():
            raise DownloadError(_missing_credentials_message(home_file, project_file))
        os.environ["KAGGLE_CONFIG_DIR"] = str(config_dir)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
    except Exception as error:  # the Kaggle client raises several unrelated types
        raise DownloadError(f"Kaggle authentication failed: {error}") from error
    return api


def _missing_credentials_message(home_file: Path, project_file: Path) -> str:
    """Build the message shown when no Kaggle credentials exist.

    Args:
        home_file: Expected location of the user-level ``kaggle.json``.
        project_file: Expected location of the project-level ``kaggle.json``.

    Returns:
        A multi-line, actionable message.
    """
    return (
        "No Kaggle credentials found.\n"
        "  1. Sign in at https://www.kaggle.com, open Settings > API and click "
        "'Create New Token'.\n"
        f"  2. Save the token text to {home_file.with_name('access_token')}, or put a legacy\n"
        f"     kaggle.json at {home_file} or {project_file}.\n"
        "  Alternatively set KAGGLE_USERNAME and KAGGLE_KEY (or KAGGLE_API_TOKEN)."
    )


def _download_with_retries(api: Any, competition: str, raw_dir: Path) -> None:
    """Download the competition archive, retrying network failures with backoff.

    Args:
        api: Authenticated ``KaggleApi`` instance.
        competition: Kaggle competition slug.
        raw_dir: Destination folder.

    Raises:
        DownloadError: On a 403 (rules not accepted) or after the final failed attempt.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            _log.info("Downloading '%s' (attempt %d/%d)...", competition, attempt, MAX_ATTEMPTS)
            api.competition_download_files(competition, path=str(raw_dir), quiet=False)
            return
        except Exception as error:  # HTTP and network errors come from several libraries
            if "403" in str(error) or "Forbidden" in str(error):
                raise DownloadError(_forbidden_message(competition)) from error
            if attempt == MAX_ATTEMPTS:
                raise DownloadError(f"Download failed after {MAX_ATTEMPTS} attempts: {error}") from error
            wait = 2**attempt
            _log.warning("Download failed (%s). Retrying in %ds.", error, wait)
            time.sleep(wait)


def _forbidden_message(competition: str) -> str:
    """Build the message shown for an HTTP 403 response.

    Args:
        competition: Kaggle competition slug.

    Returns:
        A message explaining that the competition rules must be accepted.
    """
    return (
        "Kaggle returned 403 Forbidden. Open "
        f"https://www.kaggle.com/competitions/{competition}/rules while signed in, "
        "click 'Join Competition' / 'I Understand and Accept', then run again."
    )


def _extract_archives(raw_dir: Path) -> None:
    """Unzip every archive in ``raw_dir`` (including nested ones) and delete the zips.

    Args:
        raw_dir: Folder containing the downloaded archive(s).

    Raises:
        DownloadError: If an archive is corrupt.
    """
    archives = list(raw_dir.glob("*.zip"))
    while archives:
        for archive in archives:
            try:
                with zipfile.ZipFile(archive) as bundle:
                    bundle.extractall(raw_dir)
            except zipfile.BadZipFile as error:
                raise DownloadError(f"Corrupt archive {archive.name}: {error}") from error
            archive.unlink()
        archives = list(raw_dir.glob("*.zip"))


def manual_instructions(raw_dir: Path, competition: str) -> str:
    """Explain how to obtain the data by hand.

    Args:
        raw_dir: Folder the CSV files must end up in.
        competition: Kaggle competition slug.

    Returns:
        Step-by-step instructions.
    """
    return (
        "Manual download:\n"
        f"  1. Open https://www.kaggle.com/competitions/{competition}/data and accept the rules.\n"
        "  2. Click 'Download All' to get the zip archive.\n"
        f"  3. Extract it so that {raw_dir / REQUIRED_FILE} exists.\n"
        "  4. Run the pipeline again; the download step will be skipped."
    )


def log_file_listing(raw_dir: Path) -> None:
    """Log every file in ``raw_dir`` with its size.

    Args:
        raw_dir: Folder to list.
    """
    for path in sorted(raw_dir.iterdir()):
        if path.is_file():
            _log.info("  %-28s %10.1f MB", path.name, path.stat().st_size / 1e6)
