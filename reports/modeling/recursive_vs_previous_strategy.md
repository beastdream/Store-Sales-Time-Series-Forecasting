# Recursive Global LightGBM versus Previous Strategy

## Evaluation contract

The base untuned global LightGBM uses the unchanged M6 feature list, 250 boosting rounds, and the repository's fixed default parameters. Every 16-day fold uses shared recursive inference. No tuning or final-test data is used. Legacy evidence is preserved separately.

Baselines were verified rather than rerun: they cut all targets after the origin, use calendar-date references, and cannot consume validation actuals. The four June-August folds contain no missing Christmas closure date.

## Verified baseline leaderboard

| model | folds | rmsle_mean | rmsle_std | mae_mean | wape_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| rolling_historical_median_28d | 4 | 0.483829 | 0.046095 | 103.300456 | 0.218477 |
| seasonal_naive_7d | 4 | 0.544832 | 0.048200 | 79.976144 | 0.169547 |
| seasonal_naive_14d | 4 | 0.552091 | 0.048802 | 87.070868 | 0.184478 |
| seasonal_naive_28d | 4 | 0.559653 | 0.046874 | 79.760265 | 0.168949 |
| last_value_naive | 4 | 0.609679 | 0.034279 | 147.653929 | 0.312495 |
| weekday_historical_median | 4 | 1.413779 | 0.015362 | 127.956768 | 0.270670 |

## Fold comparison

| fold | old_strategy_rmsle | new_recursive_rmsle | recursive_minus_previous |
| ---: | ---: | ---: | ---: |
| 1 | 0.399636 | 0.401056 | 0.001419 |
| 2 | 0.396827 | 0.390768 | -0.006058 |
| 3 | 0.399759 | 0.406736 | 0.006977 |
| 4 | 0.455446 | 0.437786 | -0.017660 |

## Recursive summary

- OOF rows: **114,048** (28,512 per fold).
- Mean RMSLE: **0.409086**; fold std: **0.020242**.
- Mean MAE: **66.983594**; std: **6.478294**.
- Mean WAPE: **0.141950**; std: **0.015686**.
- Strongest baseline: **rolling_historical_median_28d**, mean RMSLE **0.483829 +/- 0.046095**.
- Recursive model beats the strongest baseline by mean RMSLE.
- Fold 4: RMSLE **0.437786**, MAE **76.620464**, WAPE **0.164019**.
- Predictions are complete, finite, non-missing, and nonnegative.

## Methodology and recommendation

Correct recursive semantics are retained regardless of whether the metric improves. The next step is controlled feature ablation under this same recursive contract. Do not reuse the old tuning selection or tune parameters until recursive ablation evidence has been regenerated.
