# Data Science Implementation Roadmap

This document records what the forecasting phase actually implements, what has
been validated, and what remains future work. Metrics are copied from persisted
reports; no final-test accuracy is inferred.

## 1. Forecasting objective — implemented

- Target: Sales Volume (sales), not revenue.
- Grain: store × family × day.
- Historical actual-sales cutoff: 2017-08-15.
- Forecast horizon: 16 days, 2017-08-16 through 2017-08-31.
- Coverage: 54 stores × 33 families = 1,782 series.
- Final output: 28,512 rows with original test IDs and row order.

The contract is validated by notebook 10, src/modeling/final_forecast.py, and
artifact tests.

## 2. Temporal validation — implemented

Four rolling validation folds use complete 16-day horizons:

| Fold | Training end | Validation period |
|---:|---|---|
| 1 | 2017-06-12 | 2017-06-13 through 2017-06-28 |
| 2 | 2017-06-28 | 2017-06-29 through 2017-07-14 |
| 3 | 2017-07-14 | 2017-07-15 through 2017-07-30 |
| 4 | 2017-07-30 | 2017-07-31 through 2017-08-15 |

Every fold forecasts from one fixed origin. Selection uses the mean RMSLE across
all four folds, never a cherry-picked fold. The final competition test is not
loaded by baseline, ablation, or tuning selection entrypoints.

## 3. Baseline leaderboard — implemented

The best statistical baseline is the 28-day rolling historical median at mean
RMSLE **0.483829 ± 0.046095**. Other four-fold results are:

| Baseline | Mean RMSLE | Std |
|---|---:|---:|
| Rolling historical median, 28 days | 0.483829 | 0.046095 |
| Seasonal naive, 7 days | 0.544832 | 0.048200 |
| Seasonal naive, 14 days | 0.552091 | 0.048802 |
| Seasonal naive, 28 days | 0.559653 | 0.046874 |
| Last value naive | 0.609679 | 0.034279 |
| Weekday historical median | 1.413779 | 0.015362 |

Source: reports/modeling/baseline_summary.csv.

## 4. Feature engineering and leakage controls — implemented

Reusable feature builders cover calendar, future-known promotion, holiday/event
applicability, store metadata, family identity, causal sales lags, and shifted
rolling statistics. Feature frames are built in memory; persisted train/test
feature snapshots are not implemented.

For multi-step forecasts, post-origin actual targets are masked. Historical lag
references and origin rolling snapshots cannot read targets from inside the
forecast horizon. Behavioral tests perturb future actuals and verify feature
invariance.

Current-day future transactions, full-history readiness labels, and anomaly
outputs are forbidden model inputs. Oil remains excluded because the current
cleaning interpolation is not causal for temporal backtests. Future promotion and
event inputs are valid for the supplied competition data but require an
origin-available plan/calendar in production.

## 5. Controlled feature validation — implemented

All ablations use the same four folds and fixed LightGBM configuration:

| Added group | Mean RMSLE effect versus previous |
|---|---:|
| Sales lags | -0.041538, improved |
| Rolling statistics | -0.011840, improved |
| Calendar | -0.000606, negligible |
| Promotion | -0.016595, improved |
| Holiday/event | -0.001126, improved |
| Store/family metadata | -0.003037, improved |
| Full model without holiday/event, versus M6 | -0.002974, improved |

M6_NO_HOLIDAY is the best and most stable experiment at **0.406112 +/-
0.018907**, compared with **0.409086 +/- 0.020242** for M6. It is the recommended
36-feature candidate. Oil remains excluded because its current interpolation is
not causal for temporal backtests.

## 6. Selected recursive machine-learning model — validated

A controlled four-candidate search selected T2_moderate_capacity:

- global LightGBM regression;
- log1p target and clipped expm1 inverse;
- 250 boosting rounds;
- learning rate 0.05;
- 47 leaves, depth 10, minimum 100 rows per leaf;
- feature and bagging fractions 0.9;
- lambda L1 0.1 and lambda L2 2.0;
- fixed seeds 42.

Selection metric: mean four-fold RMSLE **0.401675 +/- 0.018557**. Mean MAE is
**63.968921** and mean WAPE is **0.135529**. Improvement over untuned M6_NO_HOLIDAY
mean RMSLE is **0.004438**, above the predefined 0.001 threshold.

Fold RMSLE is 0.392566, 0.385777, 0.400308, and 0.428048. This variation is
reported explicitly; no single fold is used as the headline selection result.

## 7. Error analysis and readiness segmentation — implemented

The persisted OOF artifact contains 114,048 predictions. Pooled metrics are RMSLE
**0.411671**, MAE **71.497549**, and WAPE **0.151310**. These pooled results are
separate from the mean-fold selection metric.

Readiness is attached after prediction only. Intermittent demand is the worst
primary class at RMSLE **0.553603**, while high volatility is **0.544593**.
Holiday rows are **0.478143** versus **0.409899** on regular rows. Detailed
store, family, promotion, holiday, readiness, and overlapping-risk tables are in
reports/modeling/error_analysis.md and scores_by_*.csv.

## 8. Specialized models — evaluated, shadow only

Croston, SBA, TSB, and two-stage LightGBM were evaluated on 417 post-hoc
intermittent-demand series. Two-stage LightGBM achieved mean RMSLE
**0.541790 ± 0.058160**, compared with **0.549916 ± 0.073648** for the tuned
global model on the same cohort. It improved only two of four folds.

This result supports a controlled shadow experiment, not production routing.
The cohort label is computed from full history; an origin-causal router has not
been implemented. The final competition forecast therefore uses the global model
for all series.

## 9. Prediction intervals — evaluated prototype

An 80% split-conformal P10/P90 method on the log1p scale uses a separate prior
16-day calibration window for each validation fold. P50 is unchanged.

- Pooled empirical coverage: **79.8252%**.
- Mean interval width: **428.216**.
- Mean three-quantile pinball loss: **28.023029**.
- Intermittent-demand coverage: **64.13%**.
- High-volatility coverage: **66.76%**.

The interval method is leakage-controlled and evaluated, but uneven segment
coverage prevents a production-ready claim. Final submission contains point
forecasts only.

## 10. Final forecast generation — regenerated and validated

The final artifact retrains the selected recursive global strategy on all 3,000,888 historical
rows through 2017-08-15. The final output covers 2017-08-16 through 2017-08-31.
Before publication, the pipeline validates exact schema, 28,512 rows, unique and
ordered IDs, complete date/store/family coverage, finite numeric predictions, and
nonnegative sales. It then reloads the serialized model and CSV.

Artifacts:

- models/global_lightgbm_chosen_config.json
- models/final_global_lightgbm.txt
- models/final_global_lightgbm_metadata.json
- reports/modeling/final_submission.csv

The canonical model, metadata, and submission now use M6_NO_HOLIDAY and the
validation-selected T2 configuration. Previous fixed-strategy artifacts are
retained with a legacy_fixed_strategy suffix. No competition score is claimed
because final-horizon target values are unavailable.

## 11. Exact reproduction commands

~~~bash
# Tests
python -m pytest -q

# Forecast contract and feature construction audit
python notebooks/10_forecast_problem_definition.py
python notebooks/13_feature_engineering.py

# Baseline backtests
python notebooks/11_temporal_backtesting.py

# Untuned model backtest and full-history artifact
python notebooks/14_global_lightgbm.py

# Controlled feature and parameter experiments
python notebooks/15_feature_ablation.py
python notebooks/16_global_lightgbm_tuning.py

# OOF diagnostics, specialists, and intervals
python notebooks/17_forecast_error_analysis.py
python notebooks/18_intermittent_demand_models.py
python notebooks/19_prediction_intervals.py

# Final full-history training and competition forecast
python notebooks/20_final_competition_forecast.py
~~~

Run the commands from the repository root after installing requirements and
placing source CSVs in data/raw/. The modeling commands are computationally
expensive and intentionally retrain their artifacts.

## 12. Remaining future work

- Implement and validate an origin-causal intermittent-demand router.
- Improve interval calibration for intermittent and high-volatility cohorts.
- Add more future origins before making production-stability claims.
- Implement causal oil availability/imputation before any oil ablation.
- Forecast transactions separately before considering future transaction inputs.
- Optionally persist versioned feature snapshots and add a feature registry.
- Compare advanced models only under the same temporal/leakage contract.
