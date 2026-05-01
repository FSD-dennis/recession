from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from recession_classifier.config import NBER_CYCLES_URL, RAW_CACHE_FILES, SAMPLE_FILES
from recession_classifier.paths import ProjectPaths

from .cache import download_file


MONTH_PATTERN = re.compile(r"([A-Za-z]+)\s+(\d{4})")


def _parse_turning_point(value: object) -> pd.Timestamp | pd.NaT:
    if pd.isna(value):
        return pd.NaT
    match = MONTH_PATTERN.search(str(value))
    if not match:
        return pd.NaT
    month_name, year = match.groups()
    return pd.Timestamp(f"{month_name} 1 {year}").to_period("M").to_timestamp("M")


def _extract_cycles_from_excel(source_path: Path) -> pd.DataFrame:
    preview = pd.read_excel(source_path, header=None)
    header_row = None
    for index, row in preview.iterrows():
        text_values = [str(value).strip().lower() for value in row.tolist()]
        if any("peak month" in value for value in text_values) and any(
            "trough month" in value for value in text_values
        ):
            header_row = index
            break

    if header_row is None:
        raise ValueError("Could not locate the Peak Month/Trough Month header in the NBER workbook.")

    extracted = pd.read_excel(source_path, header=header_row)
    peak_column = next(
        column for column in extracted.columns if "peak month" in str(column).strip().lower()
    )
    trough_column = next(
        column for column in extracted.columns if "trough month" in str(column).strip().lower()
    )
    cycles = extracted[[peak_column, trough_column]].rename(
        columns={peak_column: "peak_month", trough_column: "trough_month"}
    )
    cycles["peak_month"] = cycles["peak_month"].apply(_parse_turning_point)
    cycles["trough_month"] = cycles["trough_month"].apply(_parse_turning_point)
    cycles = cycles.dropna(subset=["peak_month", "trough_month"]).reset_index(drop=True)
    return cycles


def load_nber_cycles(paths: ProjectPaths, sample: bool = False, refresh: bool = False) -> pd.DataFrame:
    if sample:
        source_path = paths.sample_cache_dir / SAMPLE_FILES["nber_cycles"]
        cycles = pd.read_csv(source_path)
        cycles["peak_month"] = pd.to_datetime(cycles["peak_month"])
        cycles["trough_month"] = pd.to_datetime(cycles["trough_month"])
        return cycles

    source_path = download_file(
        NBER_CYCLES_URL,
        paths.raw_cache_dir / RAW_CACHE_FILES["nber_cycles"],
        refresh=refresh,
    )
    return _extract_cycles_from_excel(source_path)


def build_recession_indicator(
    cycles: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    months = pd.period_range(start=start_date, end=end_date, freq="M").to_timestamp("M")
    indicator = pd.DataFrame(index=months, data={"recession": 0}, dtype="int64")
    indicator.index.name = "month"

    for row in cycles.itertuples(index=False):
        recession_start = pd.Timestamp(row.peak_month).to_period("M").to_timestamp("M") + pd.offsets.MonthEnd(1)
        recession_end = pd.Timestamp(row.trough_month).to_period("M").to_timestamp("M")
        indicator.loc[(indicator.index >= recession_start) & (indicator.index <= recession_end), "recession"] = 1

    return indicator