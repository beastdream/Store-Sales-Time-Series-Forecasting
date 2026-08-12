# Project Architecture

## Implemented end-to-end flow

~~~text
Raw competition CSV
    ↓
Cleaning and source validation
    ↓
Interim Parquet
    ↓
Dimensional warehouse and reconciliation
    ↓
Processed Parquet
    ├─→ EDA reports
    ├─→ Local Power BI dashboard
    └─→ Forecast-readiness diagnostics
              ↓
Forecast-origin-safe feature builders
              ↓
Four rolling 16-day temporal folds
              ↓
Baselines → global LightGBM → ablation → controlled tuning
              ↓
OOF error/readiness analysis
    ├─→ Intermittent-demand shadow experiments
    └─→ Split-conformal interval evaluation
              ↓
Full-history final global model
              ↓
Validated 16-day point submission
~~~

PostgreSQL is an optional deployment branch. DDL, loading, marts, and quality SQL
exist, but database runtime execution is **NOT RUN**. Power BI is implemented
locally; Power BI Service operations are not validated.

## Stage ownership and status

| Stage | Responsible code/artifact | Principal output | Status |
|---|---|---|---|
| Raw loading and audit | src/data/load_raw.py, src/data/audit.py, notebook 01 | Typed sources and audit reports | Implemented |
| Cleaning | src/data/run_cleaning.py and clean modules | Six interim Parquet tables | Implemented |
| Warehouse | src/data/run_warehouse_build.py and builders | Eight reconciled processed tables | Implemented |
| Data analysis | notebooks 03–08 | Analytical tables, figures, insights | Implemented |
| Power BI | powerbi/store_sales_analytics.pbix | Eight-page local report | Implemented locally |
| Forecast readiness | notebook 09 | 1,782-series readiness diagnostics | Implemented |
| Forecast contract | notebook 10 | Verified target, grain, horizon, availability | Implemented |
| Temporal splits and baselines | src/modeling/splits.py, baselines.py, notebook 11 | Four-fold baseline leaderboard | Implemented |
| Forecast features | src/features/, notebook 13 | In-memory causal feature frames | Implemented; persisted snapshots not implemented |
| Global model | train_global.py, predict.py, notebook 14 | Backtest, model, comparison | Implemented |
| Feature ablation | ablation.py, notebook 15 | Controlled group effects | Implemented |
| Parameter selection | tuning.py, notebook 16 | Chosen T2 config and tuned model | Implemented |
| Error segmentation | error_analysis.py, notebook 17 | OOF and segment reports | Implemented |
| Specialized models | intermittent.py, notebook 18 | Intermittent cohort comparison | Evaluated; routing shadow-only |
| Prediction intervals | uncertainty.py, notebook 19 | Temporally calibrated interval evaluation | Evaluated prototype |
| Final forecast | final_forecast.py, notebook 20 | Final model, metadata, 28,512-row submission | Implemented and validated |
| PostgreSQL deployment | src/load_to_postgres.py, sql/ | Optional database warehouse | Code present; runtime NOT RUN |

## Forecast modeling boundary

The forecasting target is daily Sales Volume at date × store × family grain. All
model comparisons use the same four rolling 16-day folds. Final selection uses
mean fold RMSLE and its variation, never the best fold or final test targets.

The selected strategy is a global LightGBM with a log1p target, 250 trees, and the
validation-selected T2 parameters. It is retrained on actual history through
2017-08-15 and forecasts the complete 2017-08-16 to 2017-08-31 test grid.

Feature construction is performed in memory. data/features/ is retained for
optional future snapshots; the architecture does not claim train_features or
test_features files.

## Leakage and availability boundaries

Training rows use ordinary shifted causal features. A validation or final horizon
uses a single forecast origin: post-origin targets are masked, seasonal lag
references resolve to observed pre-origin dates, and rolling values are frozen
from the origin snapshot.

Directly allowed:

- deterministic calendar features;
- static store and family identifiers/metadata;
- supplied competition promotion plan;
- supplied event calendar under the competition contract.

Conditionally allowed:

- sales lags and rolling statistics only through the forecast origin;
- promotion/events in production only when their plans were available at origin.

Excluded:

- unavailable current/future transactions;
- oil under the current future-aware interpolation policy;
- full-history readiness and anomaly outputs as training features.

Readiness labels may be joined after prediction for diagnosis. Their current
full-history construction does not authorize production routing.

## Modeling artifacts and consumers

| Artifact | Role |
|---|---|
| reports/modeling/baseline_scores.csv and baseline_summary.csv | Fold-level and mean/std baseline evidence |
| reports/modeling/global_lgbm_scores.csv | Untuned global fold scores |
| reports/modeling/ablation_scores.csv | Controlled feature-group experiments |
| reports/modeling/tuning_results.csv and tuning_fold_scores.csv | Parameter-selection evidence |
| models/global_lightgbm_chosen_config.json | Immutable validation-selected final configuration |
| reports/modeling/global_lgbm_tuned_oof_predictions.parquet | Row-level OOF diagnostic source |
| reports/modeling/error_analysis.md and scores_by_*.csv | Post-hoc segment analysis |
| reports/modeling/intermittent_model_scores.csv | Specialist shadow evidence |
| reports/modeling/global_lgbm_prediction_intervals.parquet | Interval evaluation artifact |
| models/final_global_lightgbm.txt | Full-history final point model |
| models/final_global_lightgbm_metadata.json | Model, validation, cutoff, and horizon metadata |
| reports/modeling/final_submission.csv | Validated competition point forecast |

Experimental, tuned, and final model files have distinct names.

## Analytical dimensional model

- DimDate: one row per calendar date through the test horizon.
- DimStore: one row per store.
- DimFamily: one row per family.
- DimStoreDate: complete date × store context and observation flags.
- FactDailySales: one observed date × store × family row.
- FactStoreTransactions: one observed date × store row; never expanded by family.
- FactOilPrice: one calendar-date row.
- BridgeStoreHoliday: holiday-only audit/detail mapping.

For the local warehouse, FactStoreTransactions contains date_key, store_key, and
transactions. Its conformed DimStoreDate relationship is the composite
(date_key, store_key), while FactDailySales retains date_store_key.

## Power BI boundary

The PBIX contains Executive Overview, Sales Trend & Seasonality, Store
Performance, Product Family Performance, Promotion Analysis, Holiday & Event
Analysis, Transactions & Oil Drivers, and Forecast Readiness & Anomalies.
Dashboard measures are descriptive, not causal. Forecast model artifacts are not
claimed as a deployed Power BI scoring service.

## Operational limitations

- No unseen final-test accuracy can be reported.
- Final point forecasts do not use the shadow intermittent router or prototype
  prediction intervals.
- Model retraining commands are batch scripts, not an orchestrated production
  scheduler.
- Model registry, drift monitoring, service deployment, PostgreSQL runtime, and
  Power BI Service refresh remain future operational work.
