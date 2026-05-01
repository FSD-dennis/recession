from __future__ import annotations

from dataclasses import asdict, dataclass


PROJECT_SEED = 42
TARGET_COLUMN = "recession"
LAG_PERIODS = 1
MARKET_TICKER = "^GSPC"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
NBER_CYCLES_URL = (
    "https://www.nber.org/sites/default/files/2023-03/BCDC_spreadsheet_for_website.xlsx"
)

FRED_SERIES = {
    "yield_spread": {"series_id": "T10Y2YM", "cache_name": "t10y2ym.csv"},
    "unemployment_rate": {"series_id": "UNRATE", "cache_name": "unrate.csv"},
}

SAMPLE_FILES = {
    "yield_spread": "t10y2ym.csv",
    "unemployment_rate": "unrate.csv",
    "sp500": "sp500.csv",
    "nber_cycles": "nber_cycles.csv",
}

RAW_CACHE_FILES = {
    "sp500": "sp500.csv",
    "nber_cycles": "nber_cycles.xlsx",
}

BASE_FEATURE_COLUMNS = (
    "yield_spread",
    "yield_spread_change",
    "unemployment_rate",
    "unemployment_change",
    "sp500_return",
)


def lagged_feature_columns() -> list[str]:
    return [f"{column}_lag{LAG_PERIODS}" for column in BASE_FEATURE_COLUMNS]


@dataclass(frozen=True)
class SplitConfig:
    train_end: str
    validation_end: str
    test_end: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class RunConfig:
    mode: str
    start_date: str
    end_date: str
    split_config: SplitConfig

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["feature_columns"] = lagged_feature_columns()
        payload["base_feature_columns"] = list(BASE_FEATURE_COLUMNS)
        payload["seed"] = PROJECT_SEED
        return payload


FULL_RUN_CONFIG = RunConfig(
    mode="full",
    start_date="1980-01-01",
    end_date="2025-12-31",
    split_config=SplitConfig(
        train_end="2006-12-31",
        validation_end="2014-12-31",
        test_end="2025-12-31",
    ),
)

SMOKE_RUN_CONFIG = RunConfig(
    mode="sample",
    start_date="1980-01-01",
    end_date="1993-12-31",
    split_config=SplitConfig(
        train_end="1987-12-31",
        validation_end="1990-12-31",
        test_end="1993-12-31",
    ),
)


def get_run_config(sample: bool = False) -> RunConfig:
    return SMOKE_RUN_CONFIG if sample else FULL_RUN_CONFIG