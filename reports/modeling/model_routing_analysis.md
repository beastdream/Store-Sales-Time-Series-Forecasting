# Intermittent-Demand Model Routing Analysis

> **Historical artifact:** this shadow-routing experiment used the previous fixed/frozen global-model control. It is retained as experiment history and does not replace or describe the current recursive selected model. See `tuning_summary.md` and `final_forecast_report.md` for current status.

## Evaluation contract

The evaluation cohort contains 417 series labeled `Intermittent demand`. ForecastReadiness is used only to select and score this post-hoc cohort; it is not a training feature. All strategies use the same four rolling 16-day folds. Croston, SBA and TSB use fixed alpha/beta 0.1; no smoothing tuning was performed. The two-stage model uses the already chosen LightGBM configuration without further tuning.

## Evidence

| model | fold_count | rmsle_mean | rmsle_std | mae_mean | wape_mean | rmsle_improvement_vs_global | routing_eligible |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two_stage_lightgbm | 4.000000 | 0.541790 | 0.058160 | 4.982890 | 0.246528 | 0.008126 | True |
| global_lightgbm_tuned | 4.000000 | 0.549916 | 0.073648 | 5.192253 | 0.255122 | 0.000000 | False |
| tsb | 4.000000 | 0.624277 | 0.093661 | 7.488707 | 0.376932 | -0.074360 | False |
| sba | 4.000000 | 0.698993 | 0.078585 | 7.588147 | 0.382299 | -0.149077 | False |
| croston | 4.000000 | 0.708698 | 0.075447 | 7.692304 | 0.387762 | -0.158782 | False |

The lowest observed mean RMSLE is **two_stage_lightgbm** at **0.541790**. The tuned global control is **0.549916**.

Two-stage LightGBM improves RMSLE in **2 of 4 folds**. Its mean advantage is concentrated in the final fold; folds 1 and 3 are approximately flat or slightly worse. MAE and WAPE nevertheless improve on their four-fold means.

## Interpretation boundary

The comparison establishes predictive performance on these temporal folds, not a causal explanation for intermittent demand. Readiness labels were computed on full history and therefore remain routing/evaluation labels only. Any production router must reproduce its cohort classification from origin-available history.

## Routing recommendation

Recommend a controlled validation/shadow routing experiment for the post-hoc intermittent cohort using **two_stage_lightgbm**: mean RMSLE improves by 0.008126 versus the tuned global model. Do not deploy a router based directly on full-history readiness labels; first implement an origin-causal cohort rule. Keep the global model for all other series and monitor pooled metrics.
