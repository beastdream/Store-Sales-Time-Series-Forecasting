# Controlled Global LightGBM Feature Ablation

All M1-M6 and M6_NO_HOLIDAY experiments use the same four rolling 16-day folds, the same fixed LightGBM parameters, and 250 boosting rounds. No hyperparameter tuning or final test target is used.

M1-M6 effects use mean RMSLE relative to the preceding cumulative experiment; M6_NO_HOLIDAY is compared directly with M6. Absolute changes up to 0.001 are `NEGLIGIBLE`.

| experiment | model | added_group | fold_count | rmsle_mean | rmsle_std | mae_mean | wape_mean | comparison_experiment | delta_rmsle_vs_reference | effect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | rolling_historical_median_28d | strongest statistical baseline | 4 | 0.483829 | 0.046095 | 103.300456 | 0.218477 | - | - | REFERENCE |
| M1 | global_lightgbm | lag features | 4 | 0.442291 | 0.042088 | 69.043569 | 0.146351 | M0 | -0.041538 | IMPROVED |
| M2 | global_lightgbm | rolling features | 4 | 0.430450 | 0.044527 | 69.053561 | 0.146375 | M1 | -0.011840 | IMPROVED |
| M3 | global_lightgbm | calendar features | 4 | 0.429844 | 0.045157 | 69.463049 | 0.147199 | M2 | -0.000606 | NEGLIGIBLE |
| M4 | global_lightgbm | promotion features | 4 | 0.413249 | 0.022921 | 67.263942 | 0.142526 | M3 | -0.016595 | IMPROVED |
| M5 | global_lightgbm | holiday/event features | 4 | 0.412123 | 0.020736 | 67.224331 | 0.142422 | M4 | -0.001126 | IMPROVED |
| M6 | global_lightgbm | store/family metadata | 4 | 0.409086 | 0.020242 | 66.983594 | 0.141950 | M5 | -0.003037 | IMPROVED |
| M6_NO_HOLIDAY | global_lightgbm | full model without holiday/event features | 4 | 0.406112 | 0.018907 | 66.528250 | 0.140951 | M6 | -0.002974 | IMPROVED |

## Feature-group conclusions

- **lag features (M1): IMPROVED.** Mean RMSLE change versus M0 = -0.041538.
- **rolling features (M2): IMPROVED.** Mean RMSLE change versus M1 = -0.011840.
- **calendar features (M3): NEGLIGIBLE.** Mean RMSLE change versus M2 = -0.000606.
- **promotion features (M4): IMPROVED.** Mean RMSLE change versus M3 = -0.016595.
- **holiday/event features (M5): IMPROVED.** Mean RMSLE change versus M4 = -0.001126.
- **store/family metadata (M6): IMPROVED.** Mean RMSLE change versus M5 = -0.003037.
- **full model without holiday/event features (M6_NO_HOLIDAY): IMPROVED.** Mean RMSLE change versus M6 = -0.002974.
- **Oil features (M7): not run.** The current oil cleaner uses future-aware linear interpolation and `bfill`; no leakage-safe availability scenario has passed.

## M6 holiday removal decision

M6_NO_HOLIDAY has mean RMSLE 0.406112 versus 0.409086 for M6. It is better or equivalent, so holiday/event features should be omitted from the final point-forecast candidate.
The M5-versus-M4 increment and the full-model removal test answer different conditional questions. Here holiday/event fields help slightly before metadata, but hurt once metadata is present; the direct M6 comparison governs the full-model decision.

## Recommended feature set

Selection first establishes the best mean RMSLE band (within 0.001), then uses fold RMSLE stability and finally feature-count simplicity.
**M6_NO_HOLIDAY** is recommended with mean RMSLE 0.406112, std 0.018907, and 36 features: `store_nbr, family, store_type, cluster, city, state, day_of_week, week_of_year, month, quarter, year, is_weekend, is_month_start, is_month_end, is_payday, onpromotion, promotion_active, sales_lag_1, sales_lag_2, sales_lag_3, sales_lag_7, sales_lag_14, sales_lag_21, sales_lag_28, sales_lag_56, sales_lag_364, rolling_mean_7, rolling_mean_14, rolling_mean_28, rolling_mean_56, rolling_median_7, rolling_median_28, rolling_std_28, rolling_min_28, rolling_max_28, rolling_zero_rate_28`.
This is feature selection only; no hyperparameter tuning or final-test selection was performed. M7 remains prohibited.