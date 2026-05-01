from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from recession_classifier.config import MARKET_TICKER, RAW_CACHE_FILES, SAMPLE_FILES
from recession_classifier.paths import ProjectPaths


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [column[0] for column in frame.columns]
    return frame


def _normalise_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalised = frame.copy()
    date_column = next(column for column in normalised.columns if column.lower() == "date")
    value_column = next(
        (
            column
            for column in normalised.columns
            if column.lower().replace(" ", "_") in {"adjusted_close", "adj_close", "close"}
        ),
        normalised.columns[-1],
    )
    normalised = normalised[[date_column, value_column]].rename(
        columns={date_column: "date", value_column: "sp500_close"}
    )
    normalised["date"] = pd.to_datetime(normalised["date"], errors="coerce")
    normalised["sp500_close"] = pd.to_numeric(normalised["sp500_close"], errors="coerce")
    normalised = normalised.dropna(subset=["date"])
    normalised["month"] = normalised["date"].dt.to_period("M").dt.to_timestamp("M")
    monthly = normalised.groupby("month", as_index=True)["sp500_close"].last().to_frame()
    monthly["sp500_return"] = monthly["sp500_close"].pct_change()
    monthly.index.name = "month"
    return monthly.sort_index()


def _download_market_history(destination: Path) -> Path:
    history = yf.download(
        MARKET_TICKER,
        start="1980-01-01",
        end="2026-01-01",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    history = _flatten_columns(history)
    if history.empty:
        raise RuntimeError("Yahoo Finance returned no rows for the S&P 500 ticker.")

    close_column = "Adj Close" if "Adj Close" in history.columns else "Close"
    prepared = history.reset_index()[["Date", close_column]].rename(
        columns={"Date": "date", close_column: "adjusted_close"}
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(destination, index=False)
    return destination


def load_sp500_monthly(paths: ProjectPaths, sample: bool = False, refresh: bool = False) -> pd.DataFrame:
    if sample:
        source_path = paths.sample_cache_dir / SAMPLE_FILES["sp500"]
    else:
        source_path = paths.raw_cache_dir / RAW_CACHE_FILES["sp500"]
        if refresh or not source_path.exists():
            source_path = _download_market_history(source_path)

    raw = pd.read_csv(source_path)
    return _normalise_market_frame(raw)