# Git Hygiene and Portability Report

Audit date: 2026-08-15

## Policy

- `.gitattributes` normalizes repository text to LF on Windows and Linux.
- PBIX, Parquet, PNG, JPG, and JPEG files are explicitly binary and are never
  line-ending normalized.
- Local competition CSVs remain under `data/raw/` but are excluded from Git.
- Generated `data/interim/`, `data/processed/`, and `data/features/` artifacts
  remain excluded, with `.gitkeep` placeholders retained.
- Experimental model outputs remain ignored. The selected configuration, final
  model, and final metadata are explicit exceptions because project validation
  and portfolio reproducibility depend on them.

## Artifact classification and decision

| Class | Files | Decision |
|---|---|---|
| Source data | Seven competition CSVs under `data/raw/` | Keep locally; do not track. `train.csv` was already untracked, and the other six were removed from the index with `git rm --cached`. |
| Reproducible generated tables | `holiday_analysis.csv`, `promotion_analysis_matched.csv`, `transactions_analysis.csv` | Keep locally; remove from index. Together they occupy about 66 MiB. |
| Regression/model evidence | Four modeling Parquet files, final submission, score tables, reports, and figures | Keep tracked because repository tests and documented analysis consume these persisted artifacts. |
| Canonical model contract | `global_lightgbm_chosen_config.json`, `final_global_lightgbm.txt`, `final_global_lightgbm_metadata.json` | Keep and track; combined size is about 1.13 MiB. |
| Irreplaceable portfolio artifact | `powerbi/store_sales_analytics.pbix` | Keep tracked as binary; do not modify or normalize. |
| Reproducible warehouse data | Parquet under `data/interim/`, `data/processed/`, and `data/features/` | Keep locally and ignored; no generated Parquet in these directories is tracked. |

## Files removed from the Git index

The following files were untracked with `git rm --cached`; no local file was
deleted:

- `data/raw/holidays_events.csv`
- `data/raw/oil.csv`
- `data/raw/sample_submission.csv`
- `data/raw/stores.csv`
- `data/raw/test.csv`
- `data/raw/transactions.csv`
- `reports/tables/holiday_analysis.csv`
- `reports/tables/promotion_analysis_matched.csv`
- `reports/tables/transactions_analysis.csv`

## Files deliberately kept in Git

- `powerbi/store_sales_analytics.pbix` (49.57 MiB): final local dashboard.
- `reports/modeling/global_lgbm_prediction_intervals.parquet` (3.44 MiB).
- `reports/modeling/recursive_global_lgbm_oof_predictions.parquet` (1.42 MiB).
- `reports/modeling/global_lgbm_tuned_oof_predictions.parquet` (1.41 MiB).
- `reports/modeling/two_stage_intermittent_oof_predictions.parquet` (0.29 MiB).
- `reports/modeling/final_submission.csv` and its validation evidence.
- The selected/final model contract listed above.

The modeling Parquet files are reproducible, but artifact tests read them
directly. Removing them without redesigning the test/reproduction contract would
make a fresh clone incomplete, so they remain tracked.

## Remaining concerns

- The PBIX is approximately 49.57 MiB. It is intentionally retained, but it is
  the largest remaining tracked file and may approach hosting limits if future
  versions grow substantially.
- Git history still contains previously committed raw/generated files. This task
  changes only the current index and intentionally does not rewrite history.
- Competition data must be placed under `data/raw/` after a fresh clone, as
  documented in the README.

No commit or history rewrite was performed.
