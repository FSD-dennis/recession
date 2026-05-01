from __future__ import annotations

import argparse
import logging

from recession_classifier.config import get_run_config
from recession_classifier.pipeline import ensure_dataset, evaluate_only, fetch_data, full_run, smoke_run, train_only


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recession classifier pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch-data", help="Download and cache the full source data.")
    fetch_parser.add_argument("--refresh", action="store_true", help="Refresh raw downloads instead of using cache.")

    build_parser = subparsers.add_parser("build-dataset", help="Build the processed dataset.")
    build_parser.add_argument("--sample", action="store_true", help="Use the bundled cached sample data.")
    build_parser.add_argument("--refresh", action="store_true", help="Refresh remote data before building.")

    train_parser = subparsers.add_parser("train", help="Train the logistic regression model.")
    train_parser.add_argument("--sample", action="store_true", help="Train from the sample dataset.")
    train_parser.add_argument("--refresh", action="store_true", help="Refresh remote data before training.")
    train_parser.add_argument("--rebuild-dataset", action="store_true", help="Rebuild the processed dataset first.")
    train_parser.add_argument("--retrain", action="store_true", help="Force retraining even if a model bundle exists.")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the trained model and save outputs.")
    evaluate_parser.add_argument("--sample", action="store_true", help="Evaluate the sample workflow.")
    evaluate_parser.add_argument("--refresh", action="store_true", help="Refresh remote data before evaluation.")
    evaluate_parser.add_argument("--rebuild-dataset", action="store_true", help="Rebuild the processed dataset first.")
    evaluate_parser.add_argument("--retrain", action="store_true", help="Retrain the model before evaluation.")

    subparsers.add_parser("smoke-run", help="Run the bundled offline cached smoke example.")

    full_parser = subparsers.add_parser("full-run", help="Run the full 1980-2025 workflow.")
    full_parser.add_argument("--refresh", action="store_true", help="Refresh all remote downloads first.")

    return parser


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    _configure_logging()
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "fetch-data":
        result = fetch_data(refresh=args.refresh)
        print(f"Processed dataset created at {result.dataset_path}")
        return

    if args.command == "build-dataset":
        result = ensure_dataset(sample=args.sample, refresh=args.refresh, rebuild=True)
        print(f"Processed dataset created at {result.dataset_path}")
        return

    if args.command == "train":
        bundle, _, dataset_result = train_only(
            sample=args.sample,
            refresh=args.refresh,
            rebuild_dataset=args.rebuild_dataset,
            retrain=args.retrain,
        )
        print(f"Model trained for mode {bundle['mode']} using dataset {dataset_result.dataset_path}")
        return

    if args.command == "evaluate":
        artifacts, evaluation, _ = evaluate_only(
            sample=args.sample,
            refresh=args.refresh,
            rebuild_dataset=args.rebuild_dataset,
            retrain=args.retrain,
        )
        print(
            "Saved evaluation artifacts: "
            f"metrics={artifacts.metrics_path}, predictions={artifacts.predictions_path}, plots={artifacts.probability_plot_path}"
        )
        print(f"Primary metric: {evaluation.metrics['primary_metric']}")
        return

    if args.command == "smoke-run":
        artifacts, evaluation, _ = smoke_run()
        print(f"Smoke run complete. Metrics saved to {artifacts.metrics_path}")
        print(f"Primary metric: {evaluation.metrics['primary_metric']}")
        return

    if args.command == "full-run":
        artifacts, evaluation, _ = full_run(refresh=args.refresh)
        print(f"Full run complete. Metrics saved to {artifacts.metrics_path}")
        print(f"Primary metric: {evaluation.metrics['primary_metric']}")
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()