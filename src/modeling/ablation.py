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
EXPERIMENT_FEATURES: OrderedDict[str, list[str]] = OrderedDict(
    [
        ("M1", LAG_FEATURES),
        ("M2", [*LAG_FEATURES, *ROLLING_FEATURE_COLUMNS]),
        ("M3", [*LAG_FEATURES, *ROLLING_FEATURE_COLUMNS, *CALENDAR_FEATURES]),
        ("M4", [*LAG_FEATURES, *ROLLING_FEATURE_COLUMNS, *CALENDAR_FEATURES, *PROMOTION_FEATURES]),
        ("M5", [*LAG_FEATURES, *ROLLING_FEATURE_COLUMNS, *CALENDAR_FEATURES, *PROMOTION_FEATURES, *HOLIDAY_FEATURES]),
        ("M6", list(FEATURE_COLUMNS)),
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
    "M7": "oil features",
}
NEGLIGIBLE_RMSLE_THRESHOLD = 0.001


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
    order = {f"M{index}": index for index in range(8)}
    summary["_order"] = summary["experiment"].map(order)
    summary = summary.sort_values("_order", kind="stable").reset_index(drop=True)
    summary["delta_rmsle_vs_previous"] = summary["rmsle_mean"].diff()
    summary["effect"] = "reference"
    improved = summary["delta_rmsle_vs_previous"].lt(-NEGLIGIBLE_RMSLE_THRESHOLD)
    degraded = summary["delta_rmsle_vs_previous"].gt(NEGLIGIBLE_RMSLE_THRESHOLD)
    negligible = summary["delta_rmsle_vs_previous"].abs().le(NEGLIGIBLE_RMSLE_THRESHOLD)
    summary.loc[improved, "effect"] = "improved"
    summary.loc[degraded, "effect"] = "degraded"
    summary.loc[negligible, "effect"] = "negligible effect"
    return summary.drop(columns="_order")


def recommended_experiment(summary: pd.DataFrame) -> pd.Series:
    """Select the empirically strongest complete feature set by mean RMSLE."""
    return summary.loc[summary["rmsle_mean"].idxmin()]
