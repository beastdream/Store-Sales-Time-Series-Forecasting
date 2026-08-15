# Final Recursive Forecast

- Selected configuration: **T2_moderate_capacity**.
- Feature set: **M6_NO_HOLIDAY** (36 features).
- Untuned mean RMSLE: **0.406112**.
- Selected mean RMSLE: **0.401675**; std: **0.018557**.
- Mean MAE: **63.968921**; mean WAPE: **0.135529**.
- Fold 4 RMSLE: **0.428048**; MAE: **72.745373**; WAPE: **0.155724**.
- Strongest baseline: **rolling_historical_median_28d**, mean RMSLE **0.483829**.
- Inference: recursive calendar-day forecasting; no final-test target was used for feature or parameter selection.
- Submission: **28,512 rows**, exact original test ID order, unique IDs, finite and nonnegative predictions.
- Model artifact: `models/final_global_lightgbm.txt`.
- Metadata: `models/final_global_lightgbm_metadata.json`.
