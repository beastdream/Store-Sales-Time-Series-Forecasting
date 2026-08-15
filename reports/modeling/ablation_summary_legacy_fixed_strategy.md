# Controlled Global LightGBM Feature Ablation

All M1-M6 experiments use the same four rolling 16-day folds, the same fixed LightGBM parameters, and 250 boosting rounds. No hyperparameter tuning or final test target is used.

Effects use mean RMSLE relative to the immediately preceding experiment. Absolute changes below 0.001 are `negligible effect`.

| experiment | model | added_group | fold_count | rmsle_mean | rmsle_std | mae_mean | wape_mean | delta_rmsle_vs_previous | effect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | rolling_historical_median_28d | strongest statistical baseline | 4 | 0.483829 | 0.046095 | 103.300456 | 0.218477 | nan | reference |
| M1 | global_lightgbm | lag features | 4 | 0.446401 | 0.037502 | 79.569395 | 0.168454 | -0.037428 | improved |
| M2 | global_lightgbm | rolling features | 4 | 0.43414 | 0.038249 | 78.08186 | 0.165312 | -0.012261 | improved |
| M3 | global_lightgbm | calendar features | 4 | 0.429249 | 0.040467 | 74.092702 | 0.156831 | -0.004891 | improved |
| M4 | global_lightgbm | promotion features | 4 | 0.41737 | 0.029869 | 72.645957 | 0.153746 | -0.011879 | improved |
| M5 | global_lightgbm | holiday/event features | 4 | 0.418167 | 0.030218 | 72.713522 | 0.15391 | 0.000798 | negligible effect |
| M6 | global_lightgbm | store/family metadata | 4 | 0.412917 | 0.028385 | 72.762005 | 0.154012 | -0.005251 | improved |

## Feature-group conclusions

- **lag features (M1): improved.** Mean RMSLE change versus previous = -0.037428.
- **rolling features (M2): improved.** Mean RMSLE change versus previous = -0.012261.
- **calendar features (M3): improved.** Mean RMSLE change versus previous = -0.004891.
- **promotion features (M4): improved.** Mean RMSLE change versus previous = -0.011879.
- **holiday/event features (M5): negligible effect.** Mean RMSLE change versus previous = +0.000798.
- **store/family metadata (M6): improved.** Mean RMSLE change versus previous = -0.005251.
- **Oil features (M7): not run.** The current oil cleaner uses future-aware linear interpolation and `bfill`; no leakage-safe availability scenario has passed.

## Recommended feature set

**M6** is the lowest-scoring complete experiment, with mean RMSLE 0.412917. For the next model, recommend only feature groups whose incremental effect improved validation RMSLE. The resulting feature set is: `sales_lag_1, sales_lag_2, sales_lag_3, sales_lag_7, sales_lag_14, sales_lag_21, sales_lag_28, sales_lag_56, sales_lag_364, rolling_mean_7, rolling_mean_14, rolling_mean_28, rolling_mean_56, rolling_median_7, rolling_median_28, rolling_std_28, rolling_min_28, rolling_max_28, rolling_zero_rate_28, day_of_week, week_of_year, month, quarter, year, is_weekend, is_month_start, is_month_end, is_payday, onpromotion, promotion_active, store_nbr, family, store_type, cluster, city, state`.

Excluded despite appearing in the cumulative best experiment: **holiday/event features**, because its measured effect was not an improvement. This reduced combination has not itself been backtested and must pass a confirmation run before replacing the validated M6 artifact. Features added after the best experiment are not recommended without evidence. M7 remains prohibited.