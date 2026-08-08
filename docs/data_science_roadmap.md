# Data Science Roadmap

This document defines the next project phase. It is a plan only: no feature
pipeline, temporal split, model, backtest, prediction, or submission is implemented
by this documentation update.

## 1. Forecasting Objective

- **Target:** `sales` (Sales Volume, not revenue).
- **Grain:** store × family × day.
- **Stores:** 54.
- **Product families:** 33.
- **Potential series:** 1,782.
- **Historical actual-sales end:** 2017-08-15.
- **Forecast horizon:** 16 days, from 2017-08-16 through 2017-08-31.
- **Expected predictions:** 28,512 rows (`54 × 33 × 16`).

The future objective is to generate one nonnegative Sales Volume prediction for
each test row. No forecast-accuracy result currently exists.

## 2. Temporal Validation

A random train/test split is inappropriate because it mixes future and past rows,
breaks temporal dependence, and can make future information available to training.

The planned approach is rolling or walk-forward validation with **16-day validation
windows** to match the competition horizon. Each fold should:

1. Choose a historical forecast origin.
2. Build every feature using information available at or before that origin.
3. Train on the permitted history.
4. Forecast the next 16 calendar days.
5. Score the complete date × store × family validation grid.

Multiple origins should cover different seasonal, promotion, holiday, and demand
regimes. Fold definitions, minimum history, gap policy, and aggregation must be
fixed before model comparison. This is a design plan, not implemented code.

## 3. Baseline Models

Establish transparent baselines before machine learning:

- naive last-observation forecast;
- seasonal naive with 7-day lag;
- seasonal naive with 14-day lag;
- seasonal naive with 28-day lag;
- weekday median;
- rolling median;
- an intermittent-demand baseline where the readiness profile warrants it.

Baseline availability and zero-demand behavior must be defined at every fold.
Machine-learning models should only be accepted after demonstrating stable value
over appropriate baselines.

## 4. Feature Engineering

Planned feature groups are:

- **Calendar:** weekday, week/month/quarter, weekend, payday, month boundaries, and
  seasonal encodings.
- **Lag:** historical Sales Volume lags aligned to forecast origin.
- **Rolling:** lagged rolling mean, median, variability, zero rate, and trend.
- **Promotion:** future-known `onpromotion` values and historical promotion-demand
  interactions, with clear availability rules.
- **Holiday/event:** national, regional, and local applicability through store-date
  context; transferred/work-day semantics retained.
- **Store metadata:** city, state, type, and cluster.
- **Family metadata:** family identity and historical behavior computed causally.
- **Oil:** lagged or future covariates only after availability and leakage checks.

Potential outputs remain planned until implemented:

- `data/features/train_features.parquet`
- `data/features/test_features.parquet`

## 5. Leakage Prevention

At prediction time, every feature must use only information available at the
forecast origin.

Correct pattern:

```python
sales.shift(1).rolling(28).mean()
```

Incorrect pattern:

```python
sales.rolling(28).mean()
```

The incorrect version includes the current target row in its own feature and can
leak validation/future Sales Volume.

All aggregations, encodings, imputation parameters, thresholds, and normalization
must be fitted within each training fold. `ForecastReadiness` and `SalesAnomalies`
are outputs of full historical analysis. They must not automatically become model
features in backtests unless they are recalculated causally at every training
cutoff using only permitted history.

## 6. Transactions Limitation

Future transactions are not directly available for the 2017-08-16–2017-08-31
forecast horizon. Current-day future transaction values must not be used as direct
predictors unless transactions are forecast separately using a leakage-safe
process.

Permissible candidates include lagged historical transaction features known at the
forecast origin. Model evaluation should compare performance with and without them
and document the production dependency they introduce.

## 7. Oil Price Considerations

Two availability settings must be distinguished:

- **Competition-known future covariates:** oil values included in the supplied
  competition data may be usable under the competition rules.
- **Production-realistic availability:** a real forecast may not know oil prices
  for every future day and may require a lag, external forecast, or scenario.

Interpolation must be causal within temporal backtests. Filling a historical gap
with a later observation can leak future information even if it appears harmless
in a full-series EDA artifact. Oil feature policies should be compared by ablation
and documented separately for competition and production use.

## 8. Candidate Models

Recommended progression:

1. Baselines.
2. A global LightGBM or other gradient-boosting model across series.
3. Feature-group and availability ablation.
4. Error analysis by readiness segment.
5. Specialized intermittent-demand approaches where appropriate.
6. Prediction intervals and calibration.
7. Optional advanced time-series or deep-learning models later.

LSTM, GRU, and Transformer-style models should **not** be the initial baseline.
They add complexity before the data split, leakage controls, and transparent
baselines have been proven.

## 9. Evaluation

The primary competition-oriented metric is **RMSLE**, applied to nonnegative
predictions. Supporting analytical metrics are:

- **MAE** for absolute error;
- **WAPE** for aggregate volume-weighted error, with explicit zero-denominator
  handling.

Report metrics by:

- overall;
- store;
- family;
- readiness class;
- promotion status;
- holiday status.

Also inspect fold stability, bias, zero-demand behavior, cold-start groups, and
error concentration. Model selection must use validation folds rather than the
competition test target, which is unavailable.

## 10. Planned Outputs

The following are future artifacts and do not currently represent completed work:

```text
data/features/train_features.parquet
data/features/test_features.parquet

models/

reports/modeling/baseline_scores.csv
reports/modeling/backtest_scores.csv
reports/modeling/feature_importance.csv
reports/modeling/error_analysis.md
reports/modeling/final_submission.csv
```

When implemented, each artifact should record its data cutoff, fold definition,
feature version, model configuration, and reproducibility metadata. Until then,
`data/features/` and `models/` remain placeholders and no submission is claimed.
