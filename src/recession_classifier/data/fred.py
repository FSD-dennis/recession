from __future__ import annotations

from pathlib import Path

import pandas as pd

from recession_classifier.config import FRED_CSV_URL, FRED_SERIES, SAMPLE_FILES
from recession_classifier.paths import ProjectPaths

from .cache import download_file


def _normalise_fred_frame(frame: pd.DataFrame, value_name: str, series_id: str) -> pd.DataFrame:
    date_column = next(
        column for column in frame.columns if column.strip().lower() in {"date", "observation_date"}
    )
    value_column = next(
        (
            column
            for column in frame.columns
            if column.strip().upper() in {series_id.upper(), "VALUE"}
        ),
        frame.columns[1],
    )

    normalised = frame[[date_column, value_column]].rename(
        columns={date_column: "date", value_column: value_name}
    )
    normalised["date"] = pd.to_datetime(normalised["date"], errors="coerce")
    normalised[value_name] = pd.to_numeric(normalised[value_name].replace(".", pd.NA), errors="coerce")
    normalised = normalised.dropna(subset=["date"])
    normalised["month"] = normalised["date"].dt.to_period("M").dt.to_timestamp("M")
    monthly = normalised.groupby("month", as_index=True)[value_name].last().to_frame()
    monthly.index.name = "month"
    return monthly.sort_index()


def load_fred_series(
    paths: ProjectPaths,
    key: str,
    sample: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    series_spec = FRED_SERIES[key]
    if sample:
        source_path = paths.sample_cache_dir / SAMPLE_FILES[key]
    else:
        source_path = download_file(
            FRED_CSV_URL.format(series_id=series_spec["series_id"]),
            paths.raw_cache_dir / series_spec["cache_name"],
            refresh=refresh,
        )

    raw = pd.read_csv(source_path)
    return _normalise_fred_frame(raw, key, series_spec["series_id"])


def load_yield_spread(paths: ProjectPaths, sample: bool = False, refresh: bool = False) -> pd.DataFrame:
    return load_fred_series(paths=paths, key="yield_spread", sample=sample, refresh=refresh)


def load_unemployment_rate(
    paths: ProjectPaths,
    sample: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    return load_fred_series(paths=paths, key="unemployment_rate", sample=sample, refresh=refresh)