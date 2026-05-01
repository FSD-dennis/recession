from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Project-relative paths used throughout the pipeline."""

    root: Path
    data_dir: Path
    cache_dir: Path
    sample_cache_dir: Path
    raw_cache_dir: Path
    processed_dir: Path
    outputs_dir: Path
    models_dir: Path
    metrics_dir: Path
    plots_dir: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> "ProjectPaths":
        resolved_root = (root or Path(__file__).resolve().parents[2]).resolve()
        data_dir = resolved_root / "data"
        cache_dir = data_dir / "cache"
        outputs_dir = resolved_root / "outputs"
        return cls(
            root=resolved_root,
            data_dir=data_dir,
            cache_dir=cache_dir,
            sample_cache_dir=cache_dir / "sample",
            raw_cache_dir=cache_dir / "raw",
            processed_dir=data_dir / "processed",
            outputs_dir=outputs_dir,
            models_dir=outputs_dir / "models",
            metrics_dir=outputs_dir / "metrics",
            plots_dir=outputs_dir / "plots",
        )

    def ensure_runtime_dirs(self) -> None:
        for directory in (
            self.sample_cache_dir,
            self.raw_cache_dir,
            self.processed_dir,
            self.models_dir,
            self.metrics_dir,
            self.plots_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def processed_dataset_path(self, mode: str) -> Path:
        return self.processed_dir / f"modeling_dataset_{mode}.csv"

    def model_bundle_path(self, mode: str) -> Path:
        return self.models_dir / f"logistic_regression_{mode}.joblib"

    def metrics_path(self, mode: str) -> Path:
        return self.metrics_dir / f"metrics_{mode}.json"

    def predictions_path(self, mode: str) -> Path:
        return self.metrics_dir / f"predictions_{mode}.csv"

    def metadata_path(self, mode: str) -> Path:
        return self.metrics_dir / f"run_metadata_{mode}.json"

    def probability_plot_path(self, mode: str) -> Path:
        return self.plots_dir / f"recession_probability_{mode}.png"

    def confusion_plot_path(self, mode: str) -> Path:
        return self.plots_dir / f"confusion_matrix_{mode}.png"

    def coefficient_plot_path(self, mode: str) -> Path:
        return self.plots_dir / f"logistic_coefficients_{mode}.png"


def get_default_paths() -> ProjectPaths:
    return ProjectPaths.from_root()