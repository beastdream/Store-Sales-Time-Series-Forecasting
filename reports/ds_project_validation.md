# Data Science Project Validation

> Scope: temporal-validation configuration and persisted forecasting artifacts. Existence and structural checks do not independently prove model quality; recorded CV metrics remain the comparison evidence.

## Execution timestamp

2026-08-15T20:00:56.128106+07:00

## Environment

- Python: `3.11.9`
- Working directory: repository root

## Passed checks

- **Temporal split configuration:** Observed rolling-origin folds: [('2017-06-12', '2017-06-13', '2017-06-28'), ('2017-06-28', '2017-06-29', '2017-07-14'), ('2017-07-14', '2017-07-15', '2017-07-30'), ('2017-07-30', '2017-07-31', '2017-08-15')].
- **Baseline artifacts:** Validated four-fold baseline structure and finite recorded metrics; minimum recorded mean RMSLE belongs to rolling_historical_median_28d. This is an artifact check, not an independent model-quality claim.
- **Current modeling reports:** Current recursive backtest, ablation, tuning, and final forecast reports exist.
- **Selected model metadata:** Chosen configuration and final metadata are complete and mutually consistent.
- **Final model artifact:** Loaded models/final_global_lightgbm.txt with 250 trees and 36 features.
- **Final submission existence:** Found reports/modeling/final_submission.csv.
- **Final submission schema and row count:** Observed columns=['id', 'sales'], rows=28,512; test rows=28,512.
- **Final submission IDs:** Submission IDs are unique and match exact original test order.
- **Final submission predictions:** Predictions are numeric, finite, non-missing, and nonnegative.

## Warnings

- None.

## Failed checks

- None.

## Not-run checks

- None.

## Interpretation boundary

- Final-horizon accuracy is unknown because competition test targets are unavailable.
- Legacy error-segmentation, specialist-routing, and interval reports are historical evidence, not diagnostics of the current recursive selected model.
- This validator never retrains, tunes, forecasts, or changes artifacts.

## Command to reproduce

```powershell
python -m src.validate_ds_project
```
