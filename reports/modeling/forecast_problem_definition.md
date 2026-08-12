# Forecast Problem Definition

> Scope: this entrypoint audits only the forecasting contract and feature availability; it does not itself train a model or create predictions. Later validated pipeline stages and artifacts are documented in the project README.

## Verified forecasting contract

| Contract item | Verified value |
| --- | --- |
| Forecast target | sales |
| Forecast grain | store × family × day |
| Historical period | 2013-01-01 through 2017-08-15 |
| Final actual-sales date | 2017-08-15 |
| Test forecast period | 2017-08-16 through 2017-08-31 |
| Forecast horizon | 16 calendar days |
| Stores | 54 |
| Product families | 33 |
| Store-family series | 1,782 |
| Expected predictions | 28,512 |
| Test ID | id |

The supplied test is a complete `54 × 33 × 16` grid, giving `28,512` unique `id` rows. `sales` exists in train and is absent from test.

## Feature availability audit

| Feature | Source | Available historically? | Available for future forecast horizon? | Allowed in initial model? | Leakage risk? | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Calendar | test.date / DimDate | Yes | Yes | Yes | Low | Deterministic from the prediction date; derive without target data. |
| Store metadata | stores.csv / DimStore | Yes | Yes | Yes | Low | Static city, state, type, and cluster cover all test stores. |
| Family | train.csv, test.csv / DimFamily | Yes | Yes | Yes | Low | Product-family identity is part of the forecast grain. |
| onpromotion | train.csv and Kaggle test.csv | Yes | Yes | Yes | Low for competition; deployment caveat | Future-known in the supplied Kaggle test. Production use requires an available promotion plan. |
| Holiday / event | holidays_events.csv / DimStoreDate | Yes | Yes | Yes | Low to medium | Calendar-known events cover the horizon; preserve national/regional/local and transfer rules. |
| Oil | oil.csv / FactOilPrice | Yes | Competition-known: Yes | Conditional | Medium to high | Competition data reaches the horizon, but production-realistic future oil availability differs. Any interpolation must be causal within each fold. |
| Transactions | transactions.csv / FactStoreTransactions | Yes | No | Historical lags only | High for current-day values | Future transactions are not supplied. Do not use current-day transactions unless they are forecast separately. |
| Historical sales lags | train.sales / FactDailySales | Yes | At forecast origin only | Yes | High if not shifted | Use shift before rolling calculations; multi-step forecasts cannot read future actual sales. |
| Observation status | DimStoreDate.has_sales_observation | Yes | Known as no actual observation | Historical context only | Medium | Missing observation is not zero sales. Preserve the flag and never impute target zero automatically. |
| ForecastReadiness outputs | forecast_readiness.csv | Yes | Artifact exists | No | High in temporal backtests | Computed over the full historical window for 1,782 series; do not use automatically unless recalculated causally at every cutoff. |
| SalesAnomalies outputs | sales_anomalies.csv | Yes | Artifact exists | No | High in temporal backtests | Full-history review output (2,741 rows); do not use automatically unless recomputed causally at every cutoff. |

## Initial-model availability boundary

- Allowed directly: calendar, store metadata, family, and Kaggle-test `onpromotion`.
- Allowed with causal construction: historical sales lags/rolling features and historical transaction lags.
- Conditional: holiday/event context and oil, subject to forecast-origin and production-availability policies.
- Not allowed automatically: current-day future transactions, full-window ForecastReadiness outputs, and full-window SalesAnomalies outputs.

## Missing-date and observation-status rule

**Missing observation date != zero sales.**

The historical calendar contains `4` dates with no sales observation: 2013-12-25, 2014-12-25, 2015-12-25, 2016-12-25. Across the historical date-store grid, `216` of `91,152` store-days have `has_sales_observation = 0`. The 16-day forecast grid has `864` of `864` store-days without an actual sales observation by construction.

`has_sales_observation` must be preserved (or represented by an equivalent explicit status). A missing row must not be silently materialized as `sales = 0`; observed zero Sales Volume and absent observations have different meanings.

## Oil: competition versus production

The competition oil source contains dated information through the test end, so a competition experiment may evaluate it under explicit causal imputation rules. A production forecast may not know future oil prices at forecast origin. Production use must instead define lagged availability, an external oil forecast, or a scenario. Full-series interpolation that uses later dates is not valid inside temporal backtests.

## This entrypoint's boundaries

- This contract audit does not create predictions.
- It does not persist a final feature dataset.
- It does not train or evaluate a forecasting model.
- No test target was read or inferred.
