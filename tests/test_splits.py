from __future__ import annotations

import numpy as np
import pandas as pd

from recession_classifier.config import SMOKE_RUN_CONFIG, TARGET_COLUMN, lagged_feature_columns
from recession_classifier.model import split_dataset


def test_split_dataset_preserves_strict_chronology() -> None:
    months = pd.period_range("1980-01", "1993-12", freq="M").to_timestamp("M")
    data = {column: np.linspace(0.0, 1.0, len(months)) for column in lagged_feature_columns()}
    data[TARGET_COLUMN] = np.zeros(len(months), dtype=int)
    frame = pd.DataFrame(data, index=months)

    splits = split_dataset(frame, SMOKE_RUN_CONFIG.split_config)

    assert splits.train.index.max() < splits.validation.index.min()
    assert splits.validation.index.max() < splits.test.index.min()
    assert splits.train.index.max() == pd.Timestamp("1987-12-31")
    assert splits.validation.index.max() == pd.Timestamp("1990-12-31")
    assert splits.test.index.max() == pd.Timestamp("1993-12-31")