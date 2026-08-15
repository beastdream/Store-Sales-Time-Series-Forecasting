"""Small, reproducible search space for global LightGBM temporal tuning."""

from collections import OrderedDict
from copy import deepcopy

import pandas as pd

from src.modeling.train_global import DEFAULT_PARAMETERS


TUNABLE_PARAMETERS = (
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_data_in_leaf",
    "feature_fraction",
    "bagging_fraction",
    "lambda_l1",
    "lambda_l2",
)
SEARCH_CONFIGS: OrderedDict[str, dict[str, object]] = OrderedDict(
    [
        ("T0_untuned", {}),
        (
            "T1_compact_regularized",
            {
                "learning_rate": 0.05,
                "num_leaves": 24,
                "max_depth": 8,
                "min_data_in_leaf": 150,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.9,
                "lambda_l1": 0.1,
                "lambda_l2": 2.0,
            },
        ),
        (
            "T2_moderate_capacity",
            {
                "learning_rate": 0.05,
                "num_leaves": 47,
                "max_depth": 10,
                "min_data_in_leaf": 100,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.9,
                "lambda_l1": 0.1,
                "lambda_l2": 2.0,
            },
        ),
        (
            "T3_robust_subsample",
            {
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": 8,
                "min_data_in_leaf": 200,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "lambda_l1": 0.5,
                "lambda_l2": 3.0,
            },
        ),
    ]
)
MINIMUM_RMSLE_IMPROVEMENT = 0.001
MAXIMUM_RMSLE_STD_DEGRADATION = 0.002
NEAR_TIE_RMSLE = 0.0005


def resolved_parameters(overrides: dict[str, object]) -> dict[str, object]:
    """Resolve a candidate against the fixed validated configuration."""
    parameters = deepcopy(DEFAULT_PARAMETERS)
    parameters.setdefault("max_depth", -1)
    parameters.setdefault("lambda_l1", 0.0)
    parameters.update(overrides)
    return parameters


def summarize_tuning(fold_scores: pd.DataFrame) -> pd.DataFrame:
    """Return one reproducible result row per complete four-fold experiment."""
    expected = set(SEARCH_CONFIGS)
    if set(fold_scores["experiment"]) != expected:
        raise ValueError("fold scores do not cover every configured experiment")
    if not fold_scores.groupby("experiment")["fold"].nunique().eq(4).all():
        raise ValueError("every tuning experiment must contain exactly four folds")

    rows: list[dict[str, object]] = []
    for experiment, overrides in SEARCH_CONFIGS.items():
        experiment_scores = fold_scores.loc[
            fold_scores["experiment"].eq(experiment)
        ].sort_values("fold")
        parameters = resolved_parameters(overrides)
        row: dict[str, object] = {
            "experiment": experiment,
            "is_untuned_control": experiment == "T0_untuned",
            **{parameter: parameters[parameter] for parameter in TUNABLE_PARAMETERS},
            "fold_count": int(experiment_scores["fold"].nunique()),
            "rmsle_mean": experiment_scores["rmsle"].mean(),
            "rmsle_std": experiment_scores["rmsle"].std(),
            "mae_mean": experiment_scores["mae"].mean(),
            "wape_mean": experiment_scores["wape"].mean(),
            "parameter_change_count": sum(
                parameters[name] != resolved_parameters({})[name]
                for name in TUNABLE_PARAMETERS
            ),
        }
        for fold in range(1, 5):
            score = experiment_scores.loc[experiment_scores["fold"].eq(fold)].iloc[0]
            row[f"fold_{fold}_rmsle"] = score["rmsle"]
        rows.append(row)

    results = pd.DataFrame(rows)
    control_rmsle = float(
        results.loc[results["experiment"].eq("T0_untuned"), "rmsle_mean"].iloc[0]
    )
    control_std = float(
        results.loc[results["experiment"].eq("T0_untuned"), "rmsle_std"].iloc[0]
    )
    results["rmsle_improvement_vs_untuned"] = control_rmsle - results["rmsle_mean"]
    results["eligible_for_selection"] = (
        ~results["is_untuned_control"]
        & results["rmsle_improvement_vs_untuned"].ge(MINIMUM_RMSLE_IMPROVEMENT)
        & results["rmsle_std"].le(
            control_std + MAXIMUM_RMSLE_STD_DEGRADATION
        )
    )
    eligible = results.loc[results["eligible_for_selection"]]
    if eligible.empty:
        chosen = "T0_untuned"
    else:
        best_mean = float(eligible["rmsle_mean"].min())
        near_best = eligible.loc[
            eligible["rmsle_mean"].le(best_mean + NEAR_TIE_RMSLE)
        ].sort_values(
            ["rmsle_std", "parameter_change_count", "rmsle_mean"], kind="stable"
        )
        chosen = str(near_best.iloc[0]["experiment"])
    results["is_chosen"] = results["experiment"].eq(chosen)
    return results


def chosen_result(results: pd.DataFrame) -> pd.Series:
    """Return the single selection made from four-fold mean RMSLE."""
    chosen = results.loc[results["is_chosen"]]
    if len(chosen) != 1:
        raise ValueError("tuning results must contain exactly one chosen experiment")
    return chosen.iloc[0]
