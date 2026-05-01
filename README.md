# Treasury Yield Curve Recession Classifier

This project builds a modular Python 3.12 workflow to predict U.S. recession months from the Treasury yield-curve spread, unemployment, and S&P 500 returns. It ships with a small cached smoke example first, then supports the full 1980-2025 pipeline with cache-first downloads, strict chronological splits, reproducible training, saved outputs, and plots.

## Setup

1. Create a Python 3.12 virtual environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install dependencies.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

3. Run the offline smoke example first.

```bash
recession-classifier smoke-run
```

This uses the bundled cached sample data and should complete without network access.

## Full Run

Fetch raw data, rebuild the modeling dataset, train the model, evaluate it, and save plots.

```bash
recession-classifier full-run
```

Force a fresh download instead of using cached raw files.

```bash
recession-classifier full-run --refresh
```

## Common Commands

Run the package directly without the console script.

```bash
python -m recession_classifier.cli smoke-run
python -m recession_classifier.cli full-run
```

Fetch raw data only.

```bash
recession-classifier fetch-data
```

Rebuild the processed dataset only.

```bash
recession-classifier build-dataset
```

Train and evaluate from the existing processed dataset.

```bash
recession-classifier train
recession-classifier evaluate
```

Run tests.

```bash
pytest
```

## Cache-First Behavior

- The smoke run uses bundled sample files under `data/cache/sample/`.
- The full run downloads data only when the corresponding raw cache file is missing or `--refresh` is set.
- All filesystem access uses `pathlib.Path` and project-relative directories so the code never depends on an absolute personal path.

## Data Sources

- FRED monthly yield-curve spread: `T10Y2YM`
- FRED monthly unemployment rate: `UNRATE`
- Yahoo Finance S&P 500 history: `^GSPC`
- Direct NBER recession turning points: business-cycle expansions and contractions table

## Leakage Controls

- All predictors are lagged by one month before modeling.
- Train, validation, and test splits are strictly chronological.
- Missing values are handled inside an sklearn pipeline fit on the training split only.
- Seeds are fixed for reproducibility.

## Outputs

Generated artifacts are written under project-relative folders:

- `data/processed/` for prepared modeling tables
- `outputs/models/` for trained model artifacts
- `outputs/metrics/` for metrics, predictions, and metadata
- `outputs/plots/` for charts

## Default Splits

- Full dataset train: 1980-01 to 2006-12
- Full dataset validation: 2007-01 to 2014-12
- Full dataset test: 2015-01 to 2025-12
- Smoke dataset train/validation/test boundaries are smaller and defined in code for the bundled sample.

## Project Layout

```text
recession/
├── data/
│   ├── cache/
│   │   ├── raw/
│   │   └── sample/
│   └── processed/
├── outputs/
│   ├── metrics/
│   ├── models/
│   └── plots/
├── src/
│   └── recession_classifier/
│       ├── data/
│       ├── cli.py
│       ├── config.py
│       ├── evaluate.py
│       ├── model.py
│       ├── paths.py
│       ├── pipeline.py
│       └── plots.py
└── tests/
```

## Notes

- The direct NBER source currently reflects officially dated cycles through the 2020 recession. Months after the last dated trough are treated as non-recession months unless the source is updated.
- The S&P 500 series is a price index, so monthly returns in this project are price returns rather than total returns.