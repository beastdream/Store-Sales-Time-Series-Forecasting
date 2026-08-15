# Store Sales — Time Series Forecasting

## Project overview

This repository implements an end-to-end Store Sales project for the Corporación
Favorita competition data: data engineering, descriptive analysis, a local Power
BI report, forecast-readiness diagnostics, temporal backtesting, global
LightGBM modeling, post-hoc error analysis, uncertainty evaluation, and a final
16-day competition forecast.

**sales** means Sales Volume, not revenue or profit. The source has no price, cost,
margin, inventory, stockout, or lead-time fields. Promotion and holiday findings
are descriptive associations, not causal effects.

## Current project status

| Phase | Status | Evidence and boundary |
|---|---|---|
| Data Engineering | **Implemented and file-validated** | Raw loaders, cleaning, dimensional builders, reconciled Parquet warehouse, PostgreSQL DDL/load code. PostgreSQL runtime remains **NOT RUN**. |
| Data Analysis | **Implemented and validated** | Store/family performance, trend and seasonality, promotions, holidays, transactions/oil, and anomaly outputs under reports/. |
| Power BI | **Implemented locally** | Eight-page PBIX exists. Power BI Service publication, gateway, refresh, and production access are not validated. |
| Forecast Readiness | **Implemented and validated** | All 1,782 store-family series are classified; labels remain diagnostics and are not model features. |
| Forecast Modeling | **Recursive selection validated** | Base evaluation, M1-M6/M6_NO_HOLIDAY ablation, and controlled four-candidate tuning use the corrected recursive contract. |
| Final Forecast | **Regenerated and validated** | The 28,512-row submission and 250-tree model use M6_NO_HOLIDAY, selected T2 parameters, and recursive inference. |

## Dataset and forecasting problem

- Historical actual sales: **2013-01-01 through 2017-08-15**.
- Forecast target: **sales**.
- Forecast grain: **date × store × family**.
- Stores: **54**.
- Families: **33**.
- Series: **1,782**.
- Competition horizon: **16 days**, 2017-08-16 through 2017-08-31.
- Required predictions: **28,512**.

The original test ID and row order are preserved in the final submission.

## Implemented architecture

~~~text
Raw CSV
  → Cleaning and validation
  → Interim Parquet
  → Dimensional warehouse and processed Parquet
  → EDA reports and local Power BI dashboard
  → Forecast-readiness diagnostics
  → Forecast-origin-safe feature construction
  → Four-fold 16-day temporal backtests
  → Baselines, LightGBM, tuning, and error analysis
  → Full-history final training
  → Validated 16-day competition forecast
~~~

See [Architecture](docs/architecture.md), [Data Dictionary](docs/data_dictionary.md),
and [Data Science Implementation Roadmap](docs/data_science_roadmap.md).

## Temporal validation and model results

Model selection uses the mean metric across all four rolling 16-day folds. It does
not use the best individual fold or the final test set.

| Candidate | Fold count | Mean RMSLE | RMSLE std | Mean MAE | Mean WAPE |
|---|---:|---:|---:|---:|---:|
| **T2 tuned M6_NO_HOLIDAY — selected** | 4 | **0.401675** | **0.018557** | **63.968921** | **0.135529** |
| Untuned M6_NO_HOLIDAY | 4 | 0.406112 | 0.018907 | 66.528250 | 0.140951 |
| Rolling historical median, 28 days | 4 | 0.483829 | 0.046095 | 103.300456 | 0.218477 |

These values come directly from
[tuning_results.csv](reports/modeling/tuning_results.csv) and
[baseline_summary.csv](reports/modeling/baseline_summary.csv). The selected model
improved mean RMSLE over the untuned control by **0.004438**, exceeding the
predefined 0.001 selection threshold. Its fold RMSLE values were 0.392566,
0.385777, 0.400308, and 0.428048; the final fold remains the hardest.

## Recursive feature selection

The corrected recursive ablation found mean RMSLE changes of **-0.041538** for
lags, **-0.011840** for rolling statistics, **-0.000606** for calendar
(negligible), **-0.016595** for promotion, **-0.001126** for holiday/event, and
**-0.003037** for store/family metadata.

The direct full-model confirmation is decisive: **M6_NO_HOLIDAY** achieved mean
RMSLE **0.406112 +/- 0.018907**, versus **0.409086 +/- 0.020242** for M6, while
also improving MAE and WAPE. It is therefore the recommended 36-feature candidate
for the next stage. Oil was not tested because its current interpolation is
future-aware. See the [ablation report](reports/modeling/ablation_summary.md).

## Recursive tuned model

The selected T2_moderate_capacity configuration uses 47 leaves, depth 10,
minimum 100 rows per leaf, learning rate 0.05, feature/bagging fractions 0.9,
lambda L1 0.1, lambda L2 2.0, deterministic seeds, and 250 boosting rounds.
See the [controlled tuning report](reports/modeling/tuning_summary.md).

## Leakage controls

- Every validation horizon is forecast from one fixed origin.
- Post-origin actual sales are masked before lag and rolling construction.
- Rolling features use shifted targets; current-row sales never enter their own
  features.
- Final-test covariates are used only for inference, never model or parameter
  selection.
- Future onpromotion and calendar/event inputs are accepted under the supplied
  competition contract, with explicit production-availability caveats.
- Future transactions are unavailable and excluded.
- Full-history readiness and anomaly labels are post-hoc diagnostics only.
- Oil is gated off until causal availability and imputation are implemented.

See [Feature Leakage Audit](reports/modeling/feature_leakage_audit.md).

## Historical error analysis and readiness segmentation

The detailed segmentation artifacts below were generated with the deprecated
fixed/frozen multi-step strategy. They are retained as historical diagnostic
evidence and do not describe the current recursive M6_NO_HOLIDAY model. In that
historical run, pooled OOF RMSLE was **0.411671**, MAE was **71.497549**, and WAPE
was **0.151310**. Readiness labels were joined only after predictions:

- Intermittent demand is the worst primary class by RMSLE: **0.553603** across
  417 series.
- High volatility has RMSLE **0.544593** across 102 primary-class series.
- Holiday rows have RMSLE **0.478143**, versus **0.409899** for regular rows.
- Promotion-dependent series have low proportional error but high absolute error:
  RMSLE **0.233837**, MAE **233.014673**.

These are historical predictive diagnostics, not causal conclusions. They have
not yet been regenerated for the selected recursive model. See
[Error Analysis](reports/modeling/error_analysis.md).

## Historical specialized-model and interval experiments

Croston, SBA, TSB, and a two-stage LightGBM were evaluated on the 417-series
post-hoc intermittent cohort. Two-stage LightGBM had the best cohort mean RMSLE,
**0.541790 ± 0.058160**, versus **0.549916 ± 0.073648** for the tuned global
control, but improved only 2 of 4 folds. It remains a shadow recommendation
because the full-history readiness label is not an origin-causal production
router. The final forecast therefore uses the global model for every series.

These routing results used the same deprecated fixed/frozen global control. An
80% split-conformal P10/P90 layer was also evaluated using separate prior
calibration windows. Pooled empirical coverage was **79.8252%**, mean width
**428.216**, and mean three-quantile pinball loss **28.023029**. Coverage was weak
for intermittent demand (**64.13%**) and high volatility (**66.76%**), so the
interval layer is a historical evaluated prototype, not part of the final point
submission. Neither experiment has been rerun against the selected recursive
model.

See [Routing Analysis](reports/modeling/model_routing_analysis.md) and
[Prediction Intervals](reports/modeling/prediction_intervals.md).

## Final forecast artifacts

The canonical final model and submission were regenerated with the corrected
recursive strategy. Previous fixed-strategy artifacts are retained separately
with the `legacy_fixed_strategy` suffix.

- [Final submission](reports/modeling/final_submission.csv): exactly id,sales,
  28,512 rows, no index column, unique IDs in original test order, finite
  nonnegative predictions.
- models/final_global_lightgbm.txt: final 250-tree model trained on all 3,000,888
  historical rows through 2017-08-15.
- models/final_global_lightgbm_metadata.json: model type, features, parameters,
  cutoff, target transform, temporal validation evidence, horizon, ID checksum,
  and submission validation metadata.

Experimental artifacts use separate names and were not overwritten.

## Reproducibility

Use Python 3.11 or a compatible environment and place the competition CSVs under
data/raw/.

~~~bash
python -m pip install -r requirements.txt

# Data engineering
python notebooks/01_data_audit.py
python -m src.data.run_cleaning
python -m src.data.run_warehouse_build

# Forecast contract and in-memory feature construction audit
python notebooks/10_forecast_problem_definition.py
python notebooks/13_feature_engineering.py

# Baseline and ML temporal backtests
python notebooks/11_temporal_backtesting.py
python notebooks/14_global_lightgbm.py
python notebooks/15_feature_ablation.py
python notebooks/16_global_lightgbm_tuning.py

# Post-selection diagnostics
python notebooks/17_forecast_error_analysis.py
python notebooks/18_intermittent_demand_models.py
python notebooks/19_prediction_intervals.py

# Full-history training and final competition forecast
python notebooks/20_final_competition_forecast.py

# Automated validation
python -m src.validate_da_project
python -m src.validate_ds_project
python -m src.validate_project
python -m pytest -q
~~~

Feature builders are implemented in src/features/ and are executed in memory by
the modeling entrypoints. The project intentionally does not claim persisted
train_features.parquet or test_features.parquet artifacts.

## Repository structure

~~~text
data/raw/          Local competition CSVs
data/interim/      Cleaned generated Parquet
data/processed/    Reconciled dimensional warehouse Parquet
data/features/     Reserved for optional persisted feature snapshots
docs/              Architecture, data dictionary, Power BI, DS documentation
models/            Untuned, tuned, selected-config, and final model artifacts
notebooks/         Executable data, analysis, modeling, and final-forecast entrypoints
powerbi/           Local eight-page Power BI report
reports/modeling/  Backtest, ablation, error, uncertainty, and submission evidence
src/features/      Reusable forecast-origin-safe feature builders
src/modeling/      Splits, baselines, training, evaluation, tuning, and inference
tests/             Automated unit, leakage, artifact, and regression contracts
~~~

## Known limitations and future work

- No final competition score is claimed because final test targets are unavailable.
- Fold 4 is materially harder than the first three folds.
- The intermittent router requires an origin-causal cohort definition before
  deployment.
- Prediction intervals need better segment calibration and additional future
  origins before production use.
- Oil and current-day future transactions remain excluded.
- Persisted feature snapshots and a feature registry are not implemented.
- PostgreSQL runtime and Power BI Service operation remain unvalidated.
- Advanced models are future experiments, not required replacements for the
  validated global baseline.
