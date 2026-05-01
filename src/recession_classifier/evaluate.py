from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from recession_classifier.config import TARGET_COLUMN, lagged_feature_columns
from recession_classifier.model import SplitFrames


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, object]
    predictions: pd.DataFrame


def _safe_roc_auc(y_true: pd.Series, y_score: np.ndarray) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _compute_split_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, object]:
    positive_months = int(y_true.sum())
    confusion = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    recession_recall = None if positive_months == 0 else float(recall_score(y_true, y_pred, zero_division=0))
    return {
        "rows": int(len(y_true)),
        "positive_months": positive_months,
        "predicted_positive_months": int(y_pred.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recession_recall": recession_recall,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true, y_score),
        "confusion_matrix": confusion,
    }


def evaluate_model(model, split_frames: SplitFrames, threshold: float = 0.5) -> EvaluationResult:
    feature_columns = lagged_feature_columns()
    metrics: dict[str, object] = {}
    prediction_frames: list[pd.DataFrame] = []

    for split_name, frame in (
        ("train", split_frames.train),
        ("validation", split_frames.validation),
        ("test", split_frames.test),
    ):
        probabilities = model.predict_proba(frame[feature_columns])[:, 1]
        predicted_class = (probabilities >= threshold).astype(int)
        y_true = frame[TARGET_COLUMN].astype(int)
        metrics[split_name] = _compute_split_metrics(y_true, predicted_class, probabilities)

        prediction_frames.append(
            pd.DataFrame(
                {
                    "split": split_name,
                    "month": frame.index,
                    "actual_recession": y_true.to_numpy(),
                    "predicted_probability": probabilities,
                    "predicted_recession": predicted_class,
                }
            )
        )

    metrics["primary_metric"] = {
        "name": "test_recession_recall",
        "value": metrics["test"]["recession_recall"],
    }
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return EvaluationResult(metrics=metrics, predictions=predictions)