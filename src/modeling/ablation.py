"""Controlled feature-group definitions and reporting for LightGBM ablation."""

from collections import OrderedDict

import pandas as pd

from src.features.lag_features import DEFAULT_LAGS
from src.features.rolling_features import ROLLING_FEATURE_COLUMNS
from src.modeling.train_global import FEATURE_COLUMNS


LAG_FEATURES = [f"sales_lag_{lag}" for lag in DEFAULT_LAGS]
CALENDAR_FEATURES = [
    "day_of_week", "week_of_year", "month", "quarter", "year",
    "is_weekend", "is_month_start", "is_month_end", "is_payday",
]
PROMOTION_FEATURES = ["onpromotion", "promotion_active"]
HOLIDAY_FEATURES = ["holiday_count", "is_holiday", "is_work_day", "is_event"]
METADATA_FEATURES = ["store_nbr", "family", "store_type", "cluster", "city", "state"]
M6_NO_HOLIDAY_FEATURES = [
    feature for feature in FEATURE_COLUMNS if feature not in HOLIDAY_FEATURES
]
EXPERIMENT_FEATURES: OrderedDict[str, list[str]] = OrderedDict(
    [
        ("M1", LAG_FEATURES),
        ("M2", [*LAG_FEATURES, *ROLLING_FEATURE_COLUMNS]),
        ("M3", [*LAG_FEATURES, *ROLLING_FEATURE_COLUMNS, *CALENDAR_FEATURES]),
        ("M4", [*LAG_FEATURES, *ROLLING_FEATURE_COLUMNS, *CALENDAR_FEATURES, *PROMOTION_FEATURES]),
        ("M5", [*LAG_FEATURES, *ROLLING_FEATURE_COLUMNS, *CALENDAR_FEATURES, *PROMOTION_FEATURES, *HOLIDAY_FEATURES]),
        ("M6", list(FEATURE_COLUMNS)),
        ("M6_NO_HOLIDAY", M6_NO_HOLIDAY_FEATURES),
    ]
)
ADDED_GROUP = {
    "M0": "strongest statistical baseline",
    "M1": "lag features",
    "M2": "rolling features",
    "M3": "calendar features",
    "M4": "promotion features",
    "M5": "holiday/event features",
    "M6": "store/family metadata",
    "M6_NO_HOLIDAY": "full model without holiday/event features",
    "M7": "oil features",
}
NEGLIGIBLE_RMSLE_THRESHOLD = 0.001
EXPERIMENT_ORDER = ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M6_NO_HOLIDAY"]
COMPARISON_EXPERIMENT = {
    "M0": None,
    "M1": "M0",
    "M2": "M1",
    "M3": "M2",
    "M4": "M3",
    "M5": "M4",
    "M6": "M5",
    "M6_NO_HOLIDAY": "M6",
}


def summarize_ablation(scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold metrics and classify each incremental RMSLE change."""
    summary = scores.groupby(
        ["experiment", "model", "added_group"], as_index=False, sort=False
    ).agg(
        fold_count=("fold", "nunique"),
        rmsle_mean=("rmsle", "mean"),
        rmsle_std=("rmsle", "std"),
        mae_mean=("mae", "mean"),
        wape_mean=("wape", "mean"),
    )
    order = {experiment: index for index, experiment in enumerate(EXPERIMENT_ORDER)}
    summary["_order"] = summary["experiment"].map(order)
    summary = summary.sort_values("_order", kind="stable").reset_index(drop=True)
    means = summary.set_index("experiment")["rmsle_mean"]
    summary["comparison_experiment"] = summary["experiment"].map(COMPARISON_EXPERIMENT)
    summary["delta_rmsle_vs_reference"] = summary.apply(
        lambda row: (
            row["rmsle_mean"] - means.loc[row["comparison_experiment"]]
            if row["comparison_experiment"] is not None
            else pd.NA
        ),
        axis=1,
    )
    summary["effect"] = "reference"
    delta = pd.to_numeric(summary["delta_rmsle_vs_reference"], errors="coerce")
    summary.loc[delta.lt(-NEGLIGIBLE_RMSLE_THRESHOLD), "effect"] = "improved"
    summary.loc[delta.gt(NEGLIGIBLE_RMSLE_THRESHOLD), "effect"] = "degraded"
    summary.loc[delta.abs().le(NEGLIGIBLE_RMSLE_THRESHOLD), "effect"] = "negligible effect"
    return summary.drop(columns="_order")


def recommended_experiment(summary: pd.DataFrame) -> pd.Series:
    """Select by mean RMSLE, then stability and simplicity within a close band."""
    candidates = summary.loc[summary["experiment"].ne("M0")].copy()
    candidates["feature_count"] = candidates["experiment"].map(
        {name: len(features) for name, features in EXPERIMENT_FEATURES.items()}
    )
    best_mean = candidates["rmsle_mean"].min()
    close = candidates.loc[
        candidates["rmsle_mean"].le(best_mean + NEGLIGIBLE_RMSLE_THRESHOLD)
    ]
    return close.sort_values(
        ["rmsle_std", "feature_count", "rmsle_mean"], kind="stable"
    ).iloc[0]
