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
Four rolling 16-day temporal folds (base recursive run validated)
              ↓
Baselines → global LightGBM → ablation → controlled tuning
              ↓
OOF error/readiness analysis
    ├─→ Intermittent-demand shadow experiments
    └─→ Split-conformal interval evaluation
              ↓
Full-history final global model (recursive artifact validated)
              ↓
16-day point submission (validated)
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
| Temporal splits and baselines | src/modeling/splits.py, baselines.py, notebook 11 | Four-fold baseline leaderboard | Verified unaffected by the recursive refactor |
| Forecast features | src/features/, notebook 13 | In-memory causal feature frames | Implemented; persisted snapshots not implemented |
| Global model | train_global.py, recursive.py, notebooks 14 and 21 | Recursive backtest, OOF, comparison | Base untuned four-fold rerun validated |
| Feature ablation | ablation.py, notebook 15 | M1-M6 plus M6_NO_HOLIDAY | Recursive four-fold run validated; M6_NO_HOLIDAY recommended |
| Parameter selection | tuning.py, notebook 16 | Four-candidate recursive search | Validated; T2 selected on M6_NO_HOLIDAY |
| Error segmentation | error_analysis.py, notebook 17 | OOF and segment reports | Code implemented; legacy reports require rerun |
| Specialized models | intermittent.py, notebook 18 | Intermittent cohort comparison | Code implemented; routing shadow-only and legacy results require rerun |
| Prediction intervals | uncertainty.py, notebook 19 | Temporally calibrated interval evaluation | Evaluated prototype is legacy; rerun required |
| Final forecast | final_forecast.py, notebook 20 | Final model, metadata, 28,512-row submission | Regenerated and validated recursively |
| PostgreSQL deployment | src/load_to_postgres.py, sql/ | Optional database warehouse | Code present; runtime NOT RUN |

## Forecast modeling boundary

The forecasting target is daily Sales Volume at date × store × family grain. All
model comparisons use the same four rolling 16-day folds. Final selection uses
mean fold RMSLE and its variation, never the best fold or final test targets.

The previous strategy selected a global LightGBM with a log1p target and 250
trees. That selection and final artifact predate corrected recursive semantics.
The untuned base configuration, controlled feature ablation, and parameter search
have been rerun under corrected semantics. M6_NO_HOLIDAY with T2 parameters is
the validation-selected recursive strategy.

Feature construction is performed in memory. data/features/ is retained for
optional future snapshots; the architecture does not claim train_features or
test_features files.

## Leakage and availability boundaries

Training rows use ordinary shifted causal features. A validation or final horizon
uses a single forecast origin: post-origin targets are masked, seasonal lag
references use exact calendar dates. Each prior prediction is inserted into a
private history before later lag and rolling features are recomputed.

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

Recursive backtest, ablation, tuning, final model, metadata, and submission
artifacts are current. Error-analysis, specialist, and interval artifacts still
predate the semantics correction and require ordered regeneration.

| Artifact | Role |
|---|---|
| reports/modeling/baseline_scores.csv and baseline_summary.csv | Fold-level and mean/std baseline evidence |
| reports/modeling/recursive_backtest_scores.csv | Corrected recursive untuned global fold scores |
| reports/modeling/recursive_global_lgbm_oof_predictions.parquet | Corrected recursive untuned OOF predictions |
| reports/modeling/recursive_vs_previous_strategy.md | Baseline and previous-strategy comparison |
| reports/modeling/global_lgbm_scores.csv | Untuned global fold scores |
| reports/modeling/ablation_scores.csv and ablation_summary.md | Corrected recursive feature-group evidence and recommendation |
| reports/modeling/tuning_results.csv, tuning_fold_scores.csv, and tuning_summary.md | Corrected recursive parameter-selection evidence |
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

Power BI can materialize `date_store_key = date_key * 100 + store_key` on
`FactStoreTransactions` after import to implement a single-column semantic-model
relationship to `DimStoreDate`. That helper belongs to the Power BI model only;
it is not a persisted column in the local Parquet or PostgreSQL fact.

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
