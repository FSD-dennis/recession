from __future__ import annotations

import pandas as pd

from recession_classifier.data.nber import build_recession_indicator


def test_nber_indicator_starts_after_peak_and_includes_trough_month() -> None:
    cycles = pd.DataFrame(
        {
            "peak_month": pd.to_datetime(["1980-01-31"]),
            "trough_month": pd.to_datetime(["1980-07-31"]),
        }
    )

    indicator = build_recession_indicator(cycles, start_date="1980-01-01", end_date="1980-08-31")

    assert indicator.loc[pd.Timestamp("1980-01-31"), "recession"] == 0
    assert indicator.loc[pd.Timestamp("1980-02-29"), "recession"] == 1
    assert indicator.loc[pd.Timestamp("1980-07-31"), "recession"] == 1
    assert indicator.loc[pd.Timestamp("1980-08-31"), "recession"] == 0