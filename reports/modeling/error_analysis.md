# Tuned Global LightGBM OOF Error Analysis

## Scope and leakage boundary

These results use row-level predictions from four independently trained temporal folds. Each 16-day horizon is generated from one fixed origin. `ForecastReadiness` is joined only after prediction as a post-hoc label; none of its full-history statistics enters training.

## Evidence

### Overall

| observation_count | fold_count | rmsle | mae | wape |
| --- | --- | --- | --- | --- |
| 114048.000000 | 4.000000 | 0.411671 | 71.497549 | 0.151310 |

### Store type

| store_type | observation_count | rmsle | mae | wape |
| --- | --- | --- | --- | --- |
| D | 38016.000000 | 0.370219 | 58.326827 | 0.132309 |
| A | 19008.000000 | 0.416350 | 130.866914 | 0.138673 |
| E | 8448.000000 | 0.427974 | 64.173452 | 0.153826 |
| B | 16896.000000 | 0.431053 | 78.781385 | 0.175039 |
| C | 31680.000000 | 0.440091 | 49.749175 | 0.195428 |

### Promotion status

| promotion_active | observation_count | rmsle | mae | wape |
| --- | --- | --- | --- | --- |
| 0.000000 | 64393.000000 | 0.452418 | 9.297701 | 0.166310 |
| 1.000000 | 49655.000000 | 0.351871 | 152.158807 | 0.150236 |

### Holiday status

| is_holiday | observation_count | rmsle | mae | wape |
| --- | --- | --- | --- | --- |
| 0.000000 | 111309.000000 | 0.409899 | 70.646728 | 0.149123 |
| 1.000000 | 2739.000000 | 0.478143 | 106.073664 | 0.250903 |

### Readiness class (post-hoc only)

| readiness_class | series_count | observation_count | rmsle | mae | wape |
| --- | --- | --- | --- | --- | --- |
| Insufficient history | 144.000000 | 9216.000000 | 0.217705 | 17.954295 | 0.115615 |
| Promotion dependent | 410.000000 | 26240.000000 | 0.233837 | 233.014673 | 0.151951 |
| Ready | 364.000000 | 23296.000000 | 0.322599 | 29.935371 | 0.156946 |
| Ready with caution | 345.000000 | 22080.000000 | 0.471887 | 41.330599 | 0.133462 |
| High volatility | 102.000000 | 6528.000000 | 0.544593 | 19.279371 | 0.296342 |
| Intermittent demand | 417.000000 | 26688.000000 | 0.553603 | 5.192253 | 0.269230 |

### Overlapping readiness failure flags

| risk_flag | series_count | observation_count | rmsle | mae | wape |
| --- | --- | --- | --- | --- | --- |
| is_high_volatility | 469.000000 | 30016.000000 | 0.522367 | 8.034341 | 0.316533 |
| is_intermittent | 530.000000 | 33920.000000 | 0.496065 | 4.100889 | 0.270084 |
| is_promotion_dependent | 426.000000 | 27264.000000 | 0.233534 | 229.948737 | 0.150651 |
| is_insufficient_history | 144.000000 | 9216.000000 | 0.217705 | 17.954295 | 0.115615 |

### Best and worst stores

Best by RMSLE:

| store_nbr | store_type | rmsle | mae | wape |
| --- | --- | --- | --- | --- |
| 3.000000 | D | 0.304612 | 120.152658 | 0.109638 |
| 8.000000 | D | 0.313004 | 63.702021 | 0.096328 |
| 6.000000 | D | 0.327817 | 61.158117 | 0.116067 |
| 7.000000 | D | 0.339345 | 61.938750 | 0.104572 |
| 24.000000 | D | 0.339702 | 55.874624 | 0.104127 |

Worst by RMSLE:

| store_nbr | store_type | rmsle | mae | wape |
| --- | --- | --- | --- | --- |
| 50.000000 | A | 0.500873 | 89.054760 | 0.138546 |
| 19.000000 | C | 0.491257 | 36.403362 | 0.130592 |
| 47.000000 | A | 0.472273 | 159.666821 | 0.139196 |
| 48.000000 | A | 0.465415 | 135.687978 | 0.178730 |
| 26.000000 | D | 0.465107 | 36.224598 | 0.237862 |

### Best and worst families

Best by RMSLE:

| family | rmsle | mae | wape |
| --- | --- | --- | --- |
| BOOKS | 0.116753 | 0.049104 | 1.928452 |
| DAIRY | 0.148695 | 94.039777 | 0.106885 |
| PRODUCE | 0.153785 | 269.700321 | 0.114198 |
| BREAD/BAKERY | 0.167979 | 63.374317 | 0.115924 |
| GROCERY I | 0.172120 | 586.932031 | 0.128426 |

Worst by RMSLE:

| family | rmsle | mae | wape |
| --- | --- | --- | --- |
| SCHOOL AND OFFICE SUPPLIES | 0.767902 | 14.698872 | 0.843142 |
| LINGERIE | 0.637110 | 3.349709 | 0.490118 |
| GROCERY II | 0.605418 | 11.819490 | 0.373329 |
| CELEBRATION | 0.553621 | 5.040404 | 0.378235 |
| HARDWARE | 0.532096 | 1.057230 | 0.723235 |

## Evidence-based findings

- Overall pooled OOF RMSLE is **0.411671** across 114,048 predictions.
- The worst readiness class by RMSLE is **Intermittent demand** at **0.553603** across 417 series.
- The best readiness class by RMSLE is **Insufficient history** at **0.217705**; class names alone therefore do not establish failure.
- The highest-error overlapping risk cohort is **is_high_volatility** at RMSLE **0.522367**.
- **Promotion dependent** has low proportional error (RMSLE **0.233837**) but the largest readiness-class MAE (**233.014673**), consistent with its high sales volume; this is an absolute-error burden, not an RMSLE failure.
- **Insufficient history** is not an observed failure in these folds: RMSLE **0.217705** and WAPE **0.115615** are the lowest among readiness classes.
- Holiday rows have RMSLE **0.478143** versus **0.409899** on regular rows, but include only 2,739 observations.
- RMSLE, MAE and WAPE answer different questions: RMSLE emphasizes proportional/log-scale error, while MAE and WAPE are dominated more strongly by high-volume segments.

## Speculation and hypotheses to test

- Larger intermittent-demand errors may reflect zero/nonzero occurrence difficulty. This is a hypothesis, not a causal conclusion.
- Promotion-status gaps may reflect promotion intensity, assortment, or unmodeled timing; the post-hoc comparison does not estimate promotion effects.
- Holiday gaps may be unstable because relatively few OOF rows are holidays and event types are heterogeneous.
- High-volatility cohorts may benefit from robust objectives or uncertainty modeling, but their label was computed over full history and is diagnostic only.

## Specialized-model recommendation

Do **not** replace the global model solely from this segmentation. First run controlled, fold-identical experiments for the worst sufficiently large cohorts. A specialized model is warranted only if it improves cohort RMSLE without materially degrading pooled RMSLE, MAE or WAPE. Prioritize an intermittent-demand occurrence/size experiment and a high-volatility robust-loss or uncertainty experiment. There is no current evidence for an insufficient-history specialist. Keep ForecastReadiness labels outside training unless they are recomputed causally per fold.

Plots: `reports/figures/modeling/oof_family_rmsle.png` and `reports/figures/modeling/oof_readiness_rmsle.png`.