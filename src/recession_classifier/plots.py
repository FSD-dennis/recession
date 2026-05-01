from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_probability_timeline(predictions: pd.DataFrame, destination: Path) -> Path:
    ordered = predictions.sort_values("month").copy()
    ordered["month"] = pd.to_datetime(ordered["month"])

    figure, axis = plt.subplots(figsize=(14, 6))
    axis.plot(ordered["month"], ordered["predicted_probability"], color="#1f77b4", label="Predicted probability")
    axis.fill_between(
        ordered["month"],
        0,
        1,
        where=ordered["actual_recession"].astype(bool),
        color="#ffb703",
        alpha=0.25,
        label="Actual recession month",
    )
    axis.axhline(0.5, linestyle="--", color="#d62828", linewidth=1.2, label="0.5 threshold")
    axis.set_title("Predicted Recession Probability Over Time")
    axis.set_ylabel("Probability")
    axis.set_xlabel("Month")
    axis.set_ylim(0, 1)
    axis.legend(loc="upper right")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200)
    plt.close(figure)
    return destination


def plot_confusion_matrix(confusion: list[list[int]], destination: Path) -> Path:
    matrix = np.asarray(confusion)
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title("Test Split Confusion Matrix")
    axis.set_xticks([0, 1], labels=["Expansion", "Recession"])
    axis.set_yticks([0, 1], labels=["Expansion", "Recession"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center", color="#0b0f14")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200)
    plt.close(figure)
    return destination


def plot_coefficients(feature_names: list[str], coefficients: np.ndarray, destination: Path) -> Path:
    ordered = sorted(zip(feature_names, coefficients, strict=True), key=lambda item: item[1])
    labels = [item[0] for item in ordered]
    values = [item[1] for item in ordered]
    colors = ["#d62828" if value < 0 else "#2a9d8f" for value in values]

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.barh(labels, values, color=colors)
    axis.axvline(0.0, color="#0b0f14", linewidth=1)
    axis.set_title("Logistic Regression Coefficients")
    axis.set_xlabel("Coefficient")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200)
    plt.close(figure)
    return destination