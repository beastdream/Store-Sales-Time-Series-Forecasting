# Prediction Interval Evaluation

> **Historical artifact:** these interval results were calibrated around point forecasts from the previous fixed/frozen strategy. They are retained as methodology evidence but have not been regenerated for the current recursive M6_NO_HOLIDAY model.

## Method and temporal contract

P50 is the unchanged tuned LightGBM point forecast. P10/P90 use an 80% split-conformal interval on the log1p scale. For every validation fold, the calibration residuals come from a separate 16-day horizon ending before the validation horizon and forecast from an earlier origin. No current-fold target is used to calibrate its interval. The method is global and does not use readiness labels during calibration or prediction.

The P10/P90 labels denote lower/upper bounds of a nominal 80% conformal interval; they are not independently trained conditional quantile models.

## Calibration windows

| fold | calibration_origin | calibration_start | calibration_end | validation_start | validation_end | calibration_observation_count | calibration_log_radius |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000000 | 2017-05-27 00:00:00 | 2017-05-28 00:00:00 | 2017-06-12 00:00:00 | 2017-06-13 00:00:00 | 2017-06-28 00:00:00 | 28512.000000 | 0.447292 |
| 2.000000 | 2017-06-12 00:00:00 | 2017-06-13 00:00:00 | 2017-06-28 00:00:00 | 2017-06-29 00:00:00 | 2017-07-14 00:00:00 | 28512.000000 | 0.437844 |
| 3.000000 | 2017-06-28 00:00:00 | 2017-06-29 00:00:00 | 2017-07-14 00:00:00 | 2017-07-15 00:00:00 | 2017-07-30 00:00:00 | 28512.000000 | 0.439386 |
| 4.000000 | 2017-07-14 00:00:00 | 2017-07-15 00:00:00 | 2017-07-30 00:00:00 | 2017-07-31 00:00:00 | 2017-08-15 00:00:00 | 28512.000000 | 0.433162 |

## Uncertainty calibration evidence

Nominal P10/P90 coverage is **80%**. Pooled empirical coverage is **79.83%**, with mean interval width **428.216** and mean three-quantile pinball loss **28.023029**.

### Overall by temporal fold

| segment_value | observation_count | series_count | fold_count | empirical_coverage | coverage_gap_vs_nominal | mean_interval_width | p10_pinball_loss | p50_pinball_loss | p90_pinball_loss | mean_pinball_loss | point_rmsle | point_mae | point_wape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 28512.000000 | 1782.000000 | 1.000000 | 0.805906 | 0.005906 | 432.340538 | 16.709814 | 31.183071 | 30.231994 | 26.041626 | 0.397283 | 62.366143 | 0.136651 |
| 2 | 28512.000000 | 1782.000000 | 1.000000 | 0.798892 | -0.001108 | 407.355768 | 19.865738 | 37.703939 | 26.593139 | 28.054272 | 0.394095 | 75.407878 | 0.155544 |
| 3 | 28512.000000 | 1782.000000 | 1.000000 | 0.804609 | 0.004609 | 426.681819 | 18.367204 | 31.202202 | 29.583232 | 26.384213 | 0.397784 | 62.404403 | 0.129533 |
| 4 | 28512.000000 | 1782.000000 | 1.000000 | 0.783600 | -0.016400 | 446.485904 | 17.778176 | 42.905885 | 34.151959 | 31.612007 | 0.454439 | 85.811770 | 0.183695 |

### Readiness class (post-hoc only)

| segment_value | observation_count | series_count | fold_count | empirical_coverage | coverage_gap_vs_nominal | mean_interval_width | p10_pinball_loss | p50_pinball_loss | p90_pinball_loss | mean_pinball_loss | point_rmsle | point_mae | point_wape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Intermittent demand | 26688.000000 | 417.000000 | 4.000000 | 0.641299 | -0.158701 | 16.609988 | 1.113121 | 2.596127 | 2.566468 | 2.091905 | 0.553603 | 5.192253 | 0.269230 |
| High volatility | 6528.000000 | 102.000000 | 4.000000 | 0.667586 | -0.132414 | 52.622313 | 3.721025 | 9.639685 | 7.502084 | 6.954265 | 0.544593 | 19.279371 | 0.296342 |
| Ready with caution | 22080.000000 | 345.000000 | 4.000000 | 0.726676 | -0.073324 | 290.831570 | 10.875137 | 20.665299 | 20.795830 | 17.445422 | 0.471887 | 41.330599 | 0.133462 |
| Ready | 23296.000000 | 364.000000 | 4.000000 | 0.862552 | 0.062552 | 170.964360 | 7.347887 | 14.967685 | 11.296290 | 11.203954 | 0.322599 | 29.935371 | 0.156946 |
| Insufficient history | 9216.000000 | 144.000000 | 4.000000 | 0.937283 | 0.137283 | 137.135738 | 5.889021 | 8.977148 | 8.208902 | 7.691690 | 0.217705 | 17.954295 | 0.115615 |
| Promotion dependent | 26240.000000 | 410.000000 | 4.000000 | 0.944703 | 0.144703 | 1386.516144 | 59.216797 | 116.507337 | 96.111467 | 90.611867 | 0.233837 | 233.014673 | 0.151951 |

### High-volatility and intermittent cohorts

| segment_value | observation_count | series_count | fold_count | empirical_coverage | coverage_gap_vs_nominal | mean_interval_width | p10_pinball_loss | p50_pinball_loss | p90_pinball_loss | mean_pinball_loss | point_rmsle | point_mae | point_wape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_volatility | 30016.000000 | 469.000000 | 4.000000 | 0.698561 | -0.101439 | 20.618221 | 1.539866 | 4.017171 | 3.526624 | 3.027887 | 0.522367 | 8.034341 | 0.316533 |
| intermittent | 33920.000000 | 530.000000 | 4.000000 | 0.710908 | -0.089092 | 13.201823 | 0.877655 | 2.050444 | 2.034928 | 1.654343 | 0.496065 | 4.100889 | 0.270084 |

## Point accuracy versus uncertainty calibration

Point accuracy is unchanged by construction: pooled P50 RMSLE is **0.411671**, MAE **71.497549**, and WAPE **0.151310**. These metrics evaluate central predictions. Coverage, width and pinball loss evaluate interval calibration and sharpness; a wider interval can improve coverage without improving point accuracy.

## Segment findings

- Lowest readiness coverage: **Intermittent demand** at **64.13%**.
- Widest readiness interval: **Promotion dependent**, mean width **1386.516**.
- Full-history readiness labels are diagnostic only. Segment coverage differences do not authorize using those labels as model or calibration features.

## Recommendation

Keep the validated point model unchanged. Treat these intervals as an initial global calibration layer. Before production use, require stable near-80% coverage across future origins and investigate segment-conditional or adaptive conformal calibration only when implemented from origin-available information. Do not narrow intervals merely to improve sharpness if empirical coverage deteriorates.
