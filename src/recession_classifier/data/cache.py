from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import requests


LOGGER = logging.getLogger(__name__)


def download_file(url: str, destination: Path, refresh: bool = False, timeout: int = 30) -> Path:
    """Download a file only when the cache is missing or explicitly refreshed."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not refresh:
        LOGGER.info("Using cached file: %s", destination.name)
        return destination

    LOGGER.info("Downloading %s", url)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def save_dataframe(frame: pd.DataFrame, destination: Path, index: bool = True) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=index)
    return destination


def read_dataframe(source: Path, **kwargs: object) -> pd.DataFrame:
    return pd.read_csv(source, **kwargs)


def save_json(payload: dict[str, object], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination