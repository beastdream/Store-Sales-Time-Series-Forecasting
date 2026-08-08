"""Causal sales lags for daily store-family forecasting."""

from collections.abc import Iterable
from numbers import Integral

import pandas as pd


SERIES_COLUMNS = ["store_nbr", "family"]
DATE_COLUMN = "date"
TARGET_COLUMN = "sales"
DEFAULT_LAGS = (1, 2, 3, 7, 14, 21, 28, 56, 364)


def _validate_lags(lags: Iterable[int]) -> tuple[int, ...]:
    values = tuple(lags)
    if not values:
        raise ValueError("lags must contain at least one value")
    if any(isinstance(lag, bool) or not isinstance(lag, Integral) or lag <= 0 for lag in values):
        raise ValueError("every lag must be a positive integer")
    if len(set(values)) != len(values):
        raise ValueError("lags must not contain duplicates")
    return tuple(int(lag) for lag in values)


def _prepare_frame(
    frame: pd.DataFrame,
    forecast_origin: object | None,
) -> pd.DataFrame:
    required = [DATE_COLUMN, *SERIES_COLUMNS, TARGET_COLUMN]
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError("frame is missing required columns: " + ", ".join(missing))

    prepared = frame.copy()
    prepared[DATE_COLUMN] = pd.to_datetime(prepared[DATE_COLUMN], errors="coerce").dt.normalize()
    if prepared[DATE_COLUMN].isna().any():
        raise ValueError("date must contain only valid dates")
    if prepared.duplicated([DATE_COLUMN, *SERIES_COLUMNS]).any():
        raise ValueError("frame must have a unique date-store-family grain")

    prepared["_original_order"] = range(len(prepared))
    prepared = prepared.sort_values([*SERIES_COLUMNS, DATE_COLUMN], kind="stable")
    prepared["_feature_sales"] = prepared[TARGET_COLUMN]
    if forecast_origin is not None:
        cutoff = pd.Timestamp(forecast_origin).normalize()
        if pd.isna(cutoff):
            raise ValueError("forecast_origin must be a valid date")
        prepared.loc[prepared[DATE_COLUMN].gt(cutoff), "_feature_sales"] = pd.NA
    return prepared


def add_sales_lag_features(
    frame: pd.DataFrame,
    *,
    lags: Iterable[int] = DEFAULT_LAGS,
    forecast_origin: object | None = None,
) -> pd.DataFrame:
    """Add store-family sales lags without reading the current row's target.

    For ordinary training rows, omit ``forecast_origin`` and each row uses only
    earlier rows in its series. For a multi-step validation/test horizon, pass
    the single forecast cutoff. All targets after that cutoff are masked before
    shifting, so actual horizon targets can never enter later horizon features.
    This implements the horizon-safe historical-features strategy; unavailable
    short lags later in the horizon remain missing rather than being fabricated.
    """
    lag_values = _validate_lags(lags)
    prepared = _prepare_frame(frame, forecast_origin)
    grouped = prepared.groupby(SERIES_COLUMNS, observed=True, sort=False)["_feature_sales"]
    for lag in lag_values:
        prepared[f"sales_lag_{lag}"] = grouped.shift(lag)

    return (
        prepared.sort_values("_original_order", kind="stable")
        .drop(columns=["_original_order", "_feature_sales"])
        .reset_index(drop=True)
    )


def add_horizon_safe_sales_lags(
    frame: pd.DataFrame,
    forecast_origin: object,
    *,
    lags: Iterable[int] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """Explicit multi-step wrapper requiring one fixed forecast origin."""
    return add_sales_lag_features(frame, lags=lags, forecast_origin=forecast_origin)
