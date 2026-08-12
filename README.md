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
| Forecast Modeling | **Implemented and temporally validated** | Four rolling 16-day folds, statistical baselines, global LightGBM, ablation, controlled tuning, error analysis, intermittent-demand experiments, and prediction-interval evaluation. |
| Final Forecast | **Implemented and validated** | Final global LightGBM trained through 2017-08-15; 28,512 ordered test predictions generated for 2017-08-16 through 2017-08-31. |

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
| **T2 tuned global LightGBM — selected** | 4 | **0.410900** | **0.029071** | 71.497548 | 0.151356 |
| Untuned global LightGBM | 4 | 0.412917 | 0.028385 | 72.762005 | 0.154012 |
| Rolling historical median, 28 days | 4 | 0.483829 | 0.046095 | 103.300456 | 0.218477 |
| Seasonal naive, 7 days | 4 | 0.544832 | 0.048200 | 79.976144 | 0.169547 |
| Seasonal naive, 14 days | 4 | 0.552091 | 0.048802 | 87.070868 | 0.184478 |
| Seasonal naive, 28 days | 4 | 0.559653 | 0.046874 | 79.760265 | 0.168949 |
| Last value naive | 4 | 0.609679 | 0.034279 | 147.653929 | 0.312495 |
| Weekday historical median | 4 | 1.413779 | 0.015362 | 127.956768 | 0.270670 |

These values come directly from
[tuning_results.csv](reports/modeling/tuning_results.csv) and
[baseline_summary.csv](reports/modeling/baseline_summary.csv). The selected model
improved mean RMSLE over the untuned control by **0.002017**, exceeding the
predefined 0.001 selection threshold. Its fold RMSLE values were 0.397283,
0.394095, 0.397784, and 0.454439; the final fold is visibly harder.

The pooled row-level OOF RMSLE is **0.411671** across 114,048 predictions. This is
a separate pooled calculation used in error analysis, not the mean-fold selection
metric.

## Selected model and validated features

The competition strategy is one global LightGBM regressor with 250 boosting
rounds, a log1p(sales) target, and nonnegative clip(expm1(prediction), lower=0)
inversion. The selected configuration is T2_moderate_capacity with 47 leaves,
maximum depth 10, minimum 100 rows per leaf, learning rate 0.05, deterministic
seeds, feature/bagging fraction 0.9, and L1/L2 regularization 0.1/2.0.

Controlled cumulative ablation across the same four folds found:

- sales lags: improved mean RMSLE by 0.037428;
- rolling statistics: improved by 0.012261;
- calendar features: improved by 0.004891;
- promotion features: improved by 0.011879;
- holiday/event features: negligible degradation of 0.000798;
- store/family metadata: improved by 0.005251.

The final validated T2 configuration retains the complete M6 feature list,
including holiday/event columns. Although ablation suggested excluding the
holiday/event group, that reduced combination was never separately backtested and
therefore did not replace the validated M6 configuration. Oil was not tested
because its existing interpolation is future-aware. Current/future transactions,
readiness outputs, and anomaly outputs are excluded.

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

## Error analysis and readiness segmentation

The tuned model's pooled OOF results are RMSLE **0.411671**, MAE **71.497549**, and
WAPE **0.151310**. Readiness labels were joined only after predictions:

- Intermittent demand is the worst primary class by RMSLE: **0.553603** across
  417 series.
- High volatility has RMSLE **0.544593** across 102 primary-class series.
- Holiday rows have RMSLE **0.478143**, versus **0.409899** for regular rows.
- Promotion-dependent series have low proportional error but high absolute error:
  RMSLE **0.233837**, MAE **233.014673**.

These are predictive diagnostics, not causal conclusions. See
[Error Analysis](reports/modeling/error_analysis.md).

## Specialized models and prediction intervals

Croston, SBA, TSB, and a two-stage LightGBM were evaluated on the 417-series
post-hoc intermittent cohort. Two-stage LightGBM had the best cohort mean RMSLE,
**0.541790 ± 0.058160**, versus **0.549916 ± 0.073648** for the tuned global
control, but improved only 2 of 4 folds. It remains a shadow recommendation
because the full-history readiness label is not an origin-causal production
router. The final forecast therefore uses the global model for every series.

An 80% split-conformal P10/P90 layer was also evaluated using separate prior
calibration windows. Pooled empirical coverage was **79.8252%**, mean width
**428.216**, and mean three-quantile pinball loss **28.023029**. Coverage was weak
for intermittent demand (**64.13%**) and high volatility (**66.76%**), so the
interval layer is an evaluated prototype, not part of the final point submission.

See [Routing Analysis](reports/modeling/model_routing_analysis.md) and
[Prediction Intervals](reports/modeling/prediction_intervals.md).

## Final forecast artifacts

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
