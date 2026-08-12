# Forecasting Feature Leakage Audit

## Scope and decision rule

This audit covers the reusable forecasting feature pipeline as implemented before
machine-learning training. `SAFE` means the feature is available at forecast time
without target-derived future information. `CONDITIONALLY SAFE` means it is allowed
only under the stated cutoff or availability contract. `UNSAFE` means it must not
enter the initial training or inference frame in its current form.

The corrected multi-step implementation uses one recursive state transition.
Sales after the forecast origin are masked once. Each predicted day is appended
to a private temporary history before the next day's calendar lags and shifted
rolling statistics are recomputed. Actual validation/test targets are never
appended. Persisted metrics and model artifacts predate this correction and remain
legacy until the complete experiment chain is rerun.

## Feature decisions

| Feature group | Availability | Risk | Decision | Notes |
|---|---|---|---|---|
| Sales lags | Historical actuals through the fold cutoff plus prior horizon predictions | Sparse row shifts or validation actuals would make training/inference inconsistent or leak | CONDITIONALLY SAFE | Training and inference use a dense calendar. Recursive inference masks future targets, then appends predictions only, so lag N means calendar date t-N. |
| Rolling statistics | Historical actuals through the fold cutoff plus prior horizon predictions | An unshifted window includes the current target; validation actuals would leak | CONDITIONALLY SAFE | Every statistic uses shifted sales. Recursive inference recomputes each later day after appending earlier predictions only. Missing observations remain null rather than zero. |
| Calendar | Deterministic from the forecast date | Low; definitions can be wrong but do not reveal sales | SAFE | Day of week, ISO week, month, quarter, year, weekend, month boundaries and payday are reused from the project date-dimension builder. |
| Promotion | Supplied for every Kaggle test row | Production promotion plans may be unavailable or revised | CONDITIONALLY SAFE | `onpromotion` and `promotion_active` are safe for the supplied competition horizon. Production use requires a promotion schedule available at the origin. Missing future promotion values raise an error. |
| Holiday/event | Supplied event calendar with national/regional/local applicability | Announcement timing or later revisions can differ in production | CONDITIONALLY SAFE | Safe for the supplied competition calendar. Production backtests must use the event calendar version available at each origin. No causal effect is inferred. |
| Store metadata | Static store table supplied before the horizon | Historical attributes could change in a different system | SAFE | `store_type`, `cluster`, `city`, and `state` are joined many-to-one and must map every store. Reassess if metadata becomes time-varying. |
| Family metadata | Family identifier supplied on train and test rows | Target-derived family summaries would leak if computed over the full sample | SAFE | The current frame retains only the supplied family identifier. No full-history family performance metric is joined. |
| Oil | Source includes dates through the competition horizon | Current `clean_oil` uses linear interpolation and `bfill`, so a later oil value can change an earlier imputed value | CONDITIONALLY SAFE | Oil remains excluded. It may be evaluated only under an explicit competition-known availability assumption and a documented fold-specific policy. For production-realistic evaluation, use only origin-available values with causal filling, lagged publication, an external forecast, or scenarios. |
| Transactions | Historical table ends before the test horizon | Current-day future transactions are unavailable; joining them would leak or make inference impossible | UNSAFE | No transaction column is joined. Only separately implemented causal historical lags or a forecast generated without future actual transactions could later be considered. Never expand store-day transactions to family grain as if they were family-specific. |
| ForecastReadiness | Full-history analytical CSV | Series statistics and thresholds use the entire historical sample and therefore cross validation cutoffs | UNSAFE | `forecast_readiness.csv`, readiness classes, risk flags, and its full-history statistics are excluded. They may be used only if recomputed independently inside every training fold. |
| SalesAnomalies | Full-history anomaly-review CSV | Weekday peer summaries and anomaly labels use observations beyond earlier validation cutoffs | UNSAFE | `sales_anomalies.csv` and derived flags/scores are excluded. The artifact remains descriptive review output, not a modeling feature source. |

## Mandatory confirmations

1. **Validation targets do not enter validation features.** Behavioral tests replace
   every post-origin target with extreme values and verify all lag/rolling horizon
   features remain identical.
2. **Rolling windows are shifted.** Every rolling source is the group-wise
   `_feature_sales.shift(1)` column; current-day sales are not in their own feature.
3. **Inference does not insert actual future targets.** The corrected strategy is
   recursive, but only model predictions are inserted into temporary history.
4. **Test rows have no target leakage.** Canonical test rows contain `sales = NaN`,
   `sales_observed = 0`, and preserve their original test IDs.
5. **Transactions do not require unavailable future values.** Transactions are absent
   from the forecasting frame and no current-day transaction feature is implemented.
6. **ForecastReadiness outputs are not training features.** No readiness artifact or
   derived column is imported by `src/features`.
7. **SalesAnomalies outputs are not training features.** No anomaly artifact, label,
   peer statistic, or score is imported by `src/features`.
8. **Oil future interpolation is not silently accepted.** A test demonstrates that
   the current linear interpolation can change an earlier missing value when a later
   value changes. Oil is excluded unless a competition-known assumption and explicit
   availability policy are documented; the existing cleaner is not causal by default.
9. **Missing sales differs from zero sales.** Missing observations retain `NaN` and
   `sales_observed = 0`; observed zero sales retain `0` and `sales_observed = 1`.

## No-go gate

Do not begin model training if `tests/test_feature_leakage.py` or any other feature
contract test fails. Do not enable oil, current/future transactions, ForecastReadiness,
or SalesAnomalies features without a new cutoff-aware implementation and explicit
tests proving their forecast-origin availability.
