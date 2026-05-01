from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from recession_classifier.config import BASE_FEATURE_COLUMNS, LAG_PERIODS, TARGET_COLUMN, get_run_config, lagged_feature_columns
from recession_classifier.paths import ProjectPaths, get_default_paths

from .cache import save_dataframe
from .fred import load_unemployment_rate, load_yield_spread
from .market import load_sp500_monthly
from .nber import build_recession_indicator, load_nber_cycles


@dataclass(frozen=True)
class DatasetBuildResult:
    frame: pd.DataFrame
    dataset_path: str
    missing_summary: dict[str, int]


def _monthly_window(start_date: str, end_date: str) -> pd.DatetimeIndex:
    return pd.period_range(start=start_date, end=end_date, freq="M").to_timestamp("M")


def assemble_modeling_frame(
    yield_spread: pd.DataFrame,
    unemployment_rate: pd.DataFrame,
    market_data: pd.DataFrame,
    recession_indicator: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    months = _monthly_window(start_date, end_date)
    joined = recession_indicator.join(yield_spread, how="left")
    joined = joined.join(unemployment_rate, how="left")
    joined = joined.join(market_data, how="left")
    joined = joined.reindex(months)
    joined.index.name = "month"

    joined["yield_spread_change"] = joined["yield_spread"].diff()
    joined["unemployment_change"] = joined["unemployment_rate"].diff()

    for base_feature in BASE_FEATURE_COLUMNS:
        joined[f"{base_feature}_lag{LAG_PERIODS}"] = joined[base_feature].shift(LAG_PERIODS)

    joined[TARGET_COLUMN] = joined[TARGET_COLUMN].fillna(0).astype(int)
    joined["source_mode"] = "sample" if len(joined) < 240 else "full"
    joined["available_feature_count"] = joined[lagged_feature_columns()].notna().sum(axis=1)
    return joined


def build_modeling_dataset(
    sample: bool = False,
    refresh: bool = False,
    paths: ProjectPaths | None = None,
) -> DatasetBuildResult:
    run_config = get_run_config(sample=sample)
    resolved_paths = paths or get_default_paths()
    resolved_paths.ensure_runtime_dirs()

    yield_spread = load_yield_spread(resolved_paths, sample=sample, refresh=refresh)
    unemployment_rate = load_unemployment_rate(resolved_paths, sample=sample, refresh=refresh)
    market_data = load_sp500_monthly(resolved_paths, sample=sample, refresh=refresh)
    nber_cycles = load_nber_cycles(resolved_paths, sample=sample, refresh=refresh)
    recession_indicator = build_recession_indicator(
        nber_cycles,
        start_date=run_config.start_date,
        end_date=run_config.end_date,
    )

    modeling_frame = assemble_modeling_frame(
        yield_spread=yield_spread,
        unemployment_rate=unemployment_rate,
        market_data=market_data,
        recession_indicator=recession_indicator,
        start_date=run_config.start_date,
        end_date=run_config.end_date,
    )

    dataset_path = resolved_paths.processed_dataset_path(run_config.mode)
    save_dataframe(modeling_frame.reset_index(), dataset_path, index=False)
    missing_summary = {
        column: int(modeling_frame[column].isna().sum())
        for column in [*BASE_FEATURE_COLUMNS, *lagged_feature_columns()]
    }
    return DatasetBuildResult(
        frame=modeling_frame,
        dataset_path=str(dataset_path),
        missing_summary=missing_summary,
    )


def load_processed_dataset(paths: ProjectPaths, sample: bool = False) -> pd.DataFrame:
    dataset_path = paths.processed_dataset_path(get_run_config(sample=sample).mode)
    frame = pd.read_csv(dataset_path, parse_dates=["month"])
    return frame.set_index("month").sort_index()