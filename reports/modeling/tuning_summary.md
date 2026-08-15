# Controlled Recursive LightGBM Tuning

All candidates use M6_NO_HOLIDAY, the same four rolling 16-day folds, recursive inference, 250 boosting rounds, and deterministic seeds. The final competition test is not loaded by this entrypoint.

| experiment | mean RMSLE | RMSLE std | mean MAE | mean WAPE | fold 4 RMSLE | chosen |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| T0_untuned | 0.406112 | 0.018907 | 66.528250 | 0.140951 | 0.433031 | no |
| T1_compact_regularized | 0.410609 | 0.022683 | 68.109054 | 0.144304 | 0.443372 | no |
| T2_moderate_capacity | 0.401675 | 0.018557 | 63.968921 | 0.135529 | 0.428048 | yes |
| T3_robust_subsample | 0.405947 | 0.019430 | 66.536347 | 0.140978 | 0.433671 | no |

## Selection

**T2_moderate_capacity** was selected. Improvement versus untuned is **0.004438 RMSLE**. A tuned candidate must improve mean RMSLE by at least 0.001 and may not degrade fold std by more than 0.002. Candidates within 0.0005 are resolved by stability and then parameter simplicity.

Untuned control: mean RMSLE 0.406112, std 0.018907. Selected: mean RMSLE 0.401675, std 0.018557.
