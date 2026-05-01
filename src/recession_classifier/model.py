from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from recession_classifier.config import PROJECT_SEED, TARGET_COLUMN, SplitConfig, lagged_feature_columns


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_dataset(frame: pd.DataFrame, split_config: SplitConfig) -> SplitFrames:
    train_end = pd.Timestamp(split_config.train_end).to_period("M").to_timestamp("M")
    validation_end = pd.Timestamp(split_config.validation_end).to_period("M").to_timestamp("M")
    test_end = pd.Timestamp(split_config.test_end).to_period("M").to_timestamp("M")

    ordered = frame.sort_index()
    train = ordered.loc[ordered.index <= train_end]
    validation = ordered.loc[(ordered.index > train_end) & (ordered.index <= validation_end)]
    test = ordered.loc[(ordered.index > validation_end) & (ordered.index <= test_end)]

    if train.empty or validation.empty or test.empty:
        raise ValueError("One or more chronological splits are empty. Check the configured date boundaries.")

    return SplitFrames(train=train, validation=validation, test=test)


def build_training_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=PROJECT_SEED,
                ),
            ),
        ]
    )


def fit_model(train_frame: pd.DataFrame) -> Pipeline:
    feature_columns = lagged_feature_columns()
    model = build_training_pipeline()
    model.fit(train_frame[feature_columns], train_frame[TARGET_COLUMN].astype(int))
    return model


def save_model_bundle(
    destination: str,
    model: Pipeline,
    split_config: SplitConfig,
    mode: str,
) -> str:
    bundle = {
        "model": model,
        "feature_columns": lagged_feature_columns(),
        "split_config": split_config.as_dict(),
        "mode": mode,
        "target_column": TARGET_COLUMN,
        "seed": PROJECT_SEED,
    }
    joblib.dump(bundle, destination)
    return destination


def load_model_bundle(source: str) -> dict[str, object]:
    return joblib.load(source)