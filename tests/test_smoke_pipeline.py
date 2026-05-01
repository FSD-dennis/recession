from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from recession_classifier.pipeline import smoke_run
from recession_classifier.paths import ProjectPaths


def test_smoke_run_uses_cached_sample_and_creates_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_sample_dir = repo_root / "data" / "cache" / "sample"

    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure_runtime_dirs()

    for sample_file in source_sample_dir.glob("*.csv"):
        shutil.copy(sample_file, paths.sample_cache_dir / sample_file.name)

    artifacts, evaluation, dataset_result = smoke_run(paths=paths)

    assert Path(dataset_result.dataset_path).exists()
    assert Path(artifacts.model_path).exists()
    assert Path(artifacts.metrics_path).exists()
    assert Path(artifacts.predictions_path).exists()
    assert Path(artifacts.probability_plot_path).exists()
    assert evaluation.metrics["test"]["recession_recall"] is not None

    predictions = pd.read_csv(artifacts.predictions_path)
    assert not predictions.empty
    assert set(predictions["split"]) == {"train", "validation", "test"}