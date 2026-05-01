from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from recession_classifier.config import BASE_FEATURE_COLUMNS, TARGET_COLUMN, get_run_config, lagged_feature_columns
from recession_classifier.data.build_dataset import DatasetBuildResult, build_modeling_dataset, load_processed_dataset
from recession_classifier.data.cache import save_dataframe, save_json
from recession_classifier.model import SplitFrames, fit_model, load_model_bundle, save_model_bundle, split_dataset
from recession_classifier.paths import ProjectPaths, get_default_paths
from recession_classifier.plots import plot_coefficients, plot_confusion_matrix, plot_probability_timeline
from recession_classifier.evaluate import EvaluationResult, evaluate_model


@dataclass(frozen=True)
class PipelineArtifacts:
    mode: str
    dataset_path: str
    model_path: str
    metrics_path: str
    predictions_path: str
    metadata_path: str
    probability_plot_path: str
    confusion_plot_path: str
    coefficient_plot_path: str


def _resolve_paths(paths: ProjectPaths | None) -> ProjectPaths:
    resolved_paths = paths or get_default_paths()
    resolved_paths.ensure_runtime_dirs()
    return resolved_paths


def fetch_data(refresh: bool = False, paths: ProjectPaths | None = None) -> DatasetBuildResult:
    return build_modeling_dataset(sample=False, refresh=refresh, paths=paths)


def ensure_dataset(
    sample: bool = False,
    refresh: bool = False,
    rebuild: bool = False,
    paths: ProjectPaths | None = None,
) -> DatasetBuildResult:
    resolved_paths = _resolve_paths(paths)
    run_config = get_run_config(sample=sample)
    dataset_path = resolved_paths.processed_dataset_path(run_config.mode)
    if dataset_path.exists() and not rebuild and not refresh:
        frame = load_processed_dataset(resolved_paths, sample=sample)
        missing_summary = {
            column: int(frame[column].isna().sum())
            for column in [*BASE_FEATURE_COLUMNS, *lagged_feature_columns()]
        }
        return DatasetBuildResult(frame=frame, dataset_path=str(dataset_path), missing_summary=missing_summary)

    return build_modeling_dataset(sample=sample, refresh=refresh, paths=resolved_paths)


def train_only(
    sample: bool = False,
    refresh: bool = False,
    rebuild_dataset: bool = False,
    retrain: bool = False,
    paths: ProjectPaths | None = None,
) -> tuple[dict[str, object], SplitFrames, DatasetBuildResult]:
    resolved_paths = _resolve_paths(paths)
    run_config = get_run_config(sample=sample)
    dataset_result = ensure_dataset(
        sample=sample,
        refresh=refresh,
        rebuild=rebuild_dataset,
        paths=resolved_paths,
    )
    bundle_path = resolved_paths.model_bundle_path(run_config.mode)

    split_frames = split_dataset(dataset_result.frame, run_config.split_config)
    if retrain or not bundle_path.exists():
        trained_model = fit_model(split_frames.train)
        save_model_bundle(str(bundle_path), trained_model, run_config.split_config, run_config.mode)

    bundle = load_model_bundle(str(bundle_path))
    return bundle, split_frames, dataset_result


def evaluate_only(
    sample: bool = False,
    refresh: bool = False,
    rebuild_dataset: bool = False,
    retrain: bool = False,
    paths: ProjectPaths | None = None,
) -> tuple[PipelineArtifacts, EvaluationResult, DatasetBuildResult]:
    resolved_paths = _resolve_paths(paths)
    run_config = get_run_config(sample=sample)
    bundle, split_frames, dataset_result = train_only(
        sample=sample,
        refresh=refresh,
        rebuild_dataset=rebuild_dataset,
        retrain=retrain,
        paths=resolved_paths,
    )

    evaluation = evaluate_model(bundle["model"], split_frames)
    metrics_path = resolved_paths.metrics_path(run_config.mode)
    predictions_path = resolved_paths.predictions_path(run_config.mode)
    metadata_path = resolved_paths.metadata_path(run_config.mode)
    probability_plot_path = resolved_paths.probability_plot_path(run_config.mode)
    confusion_plot_path = resolved_paths.confusion_plot_path(run_config.mode)
    coefficient_plot_path = resolved_paths.coefficient_plot_path(run_config.mode)

    evaluation.predictions.to_csv(predictions_path, index=False)
    save_json(evaluation.metrics, metrics_path)

    metadata = {
        "run_config": run_config.as_dict(),
        "dataset_path": dataset_result.dataset_path,
        "missing_summary": dataset_result.missing_summary,
        "rows": int(len(dataset_result.frame)),
        "target_positive_months": int(dataset_result.frame[TARGET_COLUMN].sum()),
        "feature_columns": lagged_feature_columns(),
        "artifact_paths": {
            "metrics": str(metrics_path),
            "predictions": str(predictions_path),
            "probability_plot": str(probability_plot_path),
            "confusion_plot": str(confusion_plot_path),
            "coefficient_plot": str(coefficient_plot_path),
        },
    }
    save_json(metadata, metadata_path)

    model = bundle["model"]
    logistic_step = model.named_steps["logistic_regression"]
    coefficients = logistic_step.coef_[0]
    plot_probability_timeline(evaluation.predictions, probability_plot_path)
    plot_confusion_matrix(evaluation.metrics["test"]["confusion_matrix"], confusion_plot_path)
    plot_coefficients(lagged_feature_columns(), coefficients, coefficient_plot_path)

    artifacts = PipelineArtifacts(
        mode=run_config.mode,
        dataset_path=dataset_result.dataset_path,
        model_path=str(resolved_paths.model_bundle_path(run_config.mode)),
        metrics_path=str(metrics_path),
        predictions_path=str(predictions_path),
        metadata_path=str(metadata_path),
        probability_plot_path=str(probability_plot_path),
        confusion_plot_path=str(confusion_plot_path),
        coefficient_plot_path=str(coefficient_plot_path),
    )
    return artifacts, evaluation, dataset_result


def smoke_run(paths: ProjectPaths | None = None) -> tuple[PipelineArtifacts, EvaluationResult, DatasetBuildResult]:
    return evaluate_only(sample=True, retrain=True, rebuild_dataset=True, paths=paths)


def full_run(refresh: bool = False, paths: ProjectPaths | None = None) -> tuple[PipelineArtifacts, EvaluationResult, DatasetBuildResult]:
    return evaluate_only(
        sample=False,
        refresh=refresh,
        rebuild_dataset=refresh,
        retrain=True,
        paths=paths,
    )