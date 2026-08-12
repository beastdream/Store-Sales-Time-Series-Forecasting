"""Shifted rolling sales features for daily store-family forecasting."""

from collections.abc import Callable

import pandas as pd

from src.features.lag_features import SERIES_COLUMNS, _prepare_frame


ROLLING_FEATURE_COLUMNS = (
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_mean_56",
    "rolling_median_7",
    "rolling_median_28",
    "rolling_std_28",
    "rolling_min_28",
    "rolling_max_28",
    "rolling_zero_rate_28",
)


def _rolling_transform(
    frame: pd.DataFrame,
    source_column: str,
    window: int,
    operation: Callable[[pd.core.window.rolling.Rolling], pd.Series],
) -> pd.Series:
    return frame.groupby(SERIES_COLUMNS, observed=True, sort=False)[source_column].transform(
        lambda values: operation(values.rolling(window=window, min_periods=window))
    )


def add_sales_rolling_features(
    frame: pd.DataFrame,
    *,
    forecast_origin: object | None = None,
) -> pd.DataFrame:
    """Add rolling statistics computed strictly from ``sales.shift(1)``.

    Missing observations stay missing and invalidate a full-window statistic;
    they are never treated as zero. Training uses the dense calendar frame with
    no origin mask. The optional mask supports D+1 audits only. Multi-step
    inference uses recursive_forecast, where later windows contain historical
    actuals and earlier predictions but never actual future targets.
    """
    prepared = _prepare_frame(frame, forecast_origin)
    prepared["_shifted_sales"] = prepared.groupby(
        SERIES_COLUMNS, observed=True, sort=False
    )["_feature_sales"].shift(1)

    for window in (7, 14, 28, 56):
        prepared[f"rolling_mean_{window}"] = _rolling_transform(
            prepared, "_shifted_sales", window, lambda rolling: rolling.mean()
        )
    for window in (7, 28):
        prepared[f"rolling_median_{window}"] = _rolling_transform(
            prepared, "_shifted_sales", window, lambda rolling: rolling.median()
        )
    prepared["rolling_std_28"] = _rolling_transform(
        prepared, "_shifted_sales", 28, lambda rolling: rolling.std()
    )
    prepared["rolling_min_28"] = _rolling_transform(
        prepared, "_shifted_sales", 28, lambda rolling: rolling.min()
    )
    prepared["rolling_max_28"] = _rolling_transform(
        prepared, "_shifted_sales", 28, lambda rolling: rolling.max()
    )
    prepared["_shifted_zero"] = prepared["_shifted_sales"].eq(0).where(
        prepared["_shifted_sales"].notna()
    )
    prepared["rolling_zero_rate_28"] = _rolling_transform(
        prepared, "_shifted_zero", 28, lambda rolling: rolling.mean()
    )

    return (
        prepared.sort_values("_original_order", kind="stable")
        .drop(
            columns=[
                "_original_order",
                "_feature_sales",
                "_shifted_sales",
                "_shifted_zero",
            ]
        )
        .reset_index(drop=True)
    )


def add_horizon_safe_sales_rolling_features(
    frame: pd.DataFrame,
    forecast_origin: object,
) -> pd.DataFrame:
    """Mask post-origin targets for a D+1 rolling-feature audit."""
    return add_sales_rolling_features(frame, forecast_origin=forecast_origin)
