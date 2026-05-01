# Recession Classifier Run Summary

Generated on 2026-05-01 from the current saved artifacts in `outputs/metrics/` and `outputs/plots/`.

## Overall Status

- The offline smoke run completed successfully using the bundled cached sample data.
- The full 1980-01 to 2025-12 run completed successfully using cached/downloaded source data.
- All validation tests passed.

## Smoke Run Summary

- Mode: `sample`
- Dataset rows: 168
- Primary metric: test recession recall = `1.00`
- Test accuracy: `0.9722`
- Test precision: `0.7500`
- Test F1: `0.8571`
- Test ROC-AUC: `1.0000`
- Test confusion matrix: `[[32, 1], [0, 3]]`

Interpretation:

- The smoke result is strong because the bundled sample data is a deterministic small example designed to verify that the pipeline runs correctly.
- This result is useful as a functional check, not as a realistic estimate of production performance.

## Full Run Summary

- Mode: `full`
- Date range: `1980-01-01` to `2025-12-31`
- Dataset rows: 552
- Features used:
  - `yield_spread_lag1`
  - `yield_spread_change_lag1`
  - `unemployment_rate_lag1`
  - `unemployment_change_lag1`
  - `sp500_return_lag1`
- Primary metric: test recession recall = `0.50`

### Split Metrics

Train:

- Rows: `324`
- Positive months: `38`
- Accuracy: `0.8333`
- Precision: `0.4000`
- Recession recall: `0.8421`
- F1: `0.5424`
- ROC-AUC: `0.9037`
- Confusion matrix: `[[238, 48], [6, 32]]`

Validation:

- Rows: `96`
- Positive months: `18`
- Accuracy: `0.8750`
- Precision: `0.6500`
- Recession recall: `0.7222`
- F1: `0.6842`
- ROC-AUC: `0.8846`
- Confusion matrix: `[[71, 7], [5, 13]]`

Test:

- Rows: `132`
- Positive months: `2`
- Predicted positive months: `21`
- Accuracy: `0.8409`
- Precision: `0.0476`
- Recession recall: `0.5000`
- F1: `0.0870`
- ROC-AUC: `0.7808`
- Confusion matrix: `[[110, 20], [1, 1]]`

### Test-Set Behavior

Actual recession months in the held-out test split:

- `2020-03-31`: predicted probability `0.2390`, predicted class `0` (missed)
- `2020-04-30`: predicted probability `0.9998`, predicted class `1` (caught)

The model therefore identified `1` of the `2` held-out recession months.

The test result also shows substantial false positives:

- `21` test months were predicted as recession.
- Only `1` of those `21` predicted recession months was actually labeled as recession.

Interpretation:

- The classifier has some signal, but the current default threshold produces too many false positives on the 2015-2025 holdout period.
- Validation performance is materially better than final test precision, so the model does not generalize cleanly to the final holdout with the present feature set and threshold.
- The headline metric requested for this task, recession-month recall, is met at `0.50` on the full held-out test split.

## Missing-Value Handling

- Missing values remain at the raw feature stage near the front edge of the sample because of differencing and lag construction.
- These are handled inside the sklearn pipeline with median imputation fit on the training split only, which preserves chronological integrity.

## Validation Status

- `pytest` status: `3 passed`
- Smoke pipeline command status: passed
- Full pipeline command status: passed

## Output Artifacts

Metrics and predictions:

- `outputs/metrics/metrics_sample.json`
- `outputs/metrics/metrics_full.json`
- `outputs/metrics/predictions_sample.csv`
- `outputs/metrics/predictions_full.csv`
- `outputs/metrics/run_metadata_sample.json`
- `outputs/metrics/run_metadata_full.json`

Plots:

- `outputs/plots/recession_probability_sample.png`
- `outputs/plots/recession_probability_full.png`
- `outputs/plots/confusion_matrix_sample.png`
- `outputs/plots/confusion_matrix_full.png`
- `outputs/plots/logistic_coefficients_sample.png`
- `outputs/plots/logistic_coefficients_full.png`

Models:

- `outputs/models/logistic_regression_sample.joblib`
- `outputs/models/logistic_regression_full.joblib`

## Practical Conclusion

The project is functioning correctly end to end. The smoke example is healthy, and the full pipeline runs reproducibly with cached data and saved artifacts. The current baseline Logistic Regression model catches half of the recession months in the held-out test window, but it does so with low precision because it flags many non-recession months as recession.