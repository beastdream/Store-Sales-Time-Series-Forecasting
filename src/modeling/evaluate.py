"""Evaluation and baseline comparison helpers for forecasting models."""

import pandas as pd

from src.modeling.metrics import mae, rmsle, wape


def score_predictions(actual: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
    """Score one complete date-store-family prediction frame."""
    grain = ["date", "store_nbr", "family"]
    scored = actual[grain + ["sales"]].merge(
        predictions, on=grain, how="left", validate="one_to_one"
    )
    if len(scored) != len(actual) or scored["prediction"].isna().any():
        raise ValueError("predictions do not cover the complete actual grain")
    return {
        "rmsle": rmsle(scored["sales"], scored["prediction"]),
        "mae": mae(scored["sales"], scored["prediction"]),
        "wape": wape(scored["sales"], scored["prediction"]),
    }


def summarize_model_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize mean and sample standard deviation across temporal folds."""
    return scores.groupby("model", as_index=False).agg(
        fold_count=("fold", "nunique"),
        rmsle_mean=("rmsle", "mean"),
        rmsle_std=("rmsle", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        wape_mean=("wape", "mean"),
        wape_std=("wape", "std"),
    )


def compare_with_baselines(
    model_summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create one ranked leaderboard and mark strongest-baseline improvement."""
    baseline = baseline_summary.copy()
    baseline["model_type"] = "baseline"
    candidate = model_summary.copy()
    candidate["model_type"] = "machine_learning"
    strongest_rmsle = float(baseline["rmsle_mean"].min())
    comparison = pd.concat([baseline, candidate], ignore_index=True, sort=False)
    comparison["strongest_baseline_rmsle"] = strongest_rmsle
    comparison["beats_strongest_baseline"] = (
        comparison["model_type"].eq("machine_learning")
        & comparison["rmsle_mean"].lt(strongest_rmsle)
    )
    comparison = comparison.sort_values(["rmsle_mean", "mae_mean"], kind="stable")
    comparison.insert(0, "rmsle_rank", range(1, len(comparison) + 1))
    return comparison.reset_index(drop=True)
