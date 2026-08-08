"""Horizon-safe statistical baselines for daily store-family Sales Volume."""

from collections.abc import Iterable
from numbers import Integral

import numpy as np
import pandas as pd


SERIES_COLUMNS = ["store_nbr", "family"]
DATE_COLUMN = "date"
TARGET_COLUMN = "sales"
PREDICTION_COLUMN = "prediction"
BASELINE_MODELS = (
    "last_value_naive",
    "seasonal_naive_7d",
    "seasonal_naive_14d",
    "seasonal_naive_28d",
    "weekday_historical_median",
    "rolling_historical_median_28d",
)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _prepare_inputs(
    history: pd.DataFrame,
    forecast_dates: Iterable[object],
    cutoff: object,
    series: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, pd.Timestamp, pd.DataFrame]:
    """Validate inputs and remove every target row after the forecast cutoff."""
    required = [DATE_COLUMN, *SERIES_COLUMNS, TARGET_COLUMN]
    missing = [column for column in required if column not in history.columns]
    if missing:
        raise KeyError("history is missing required columns: " + ", ".join(missing))

    cutoff_date = pd.Timestamp(cutoff).normalize()
    if pd.isna(cutoff_date):
        raise ValueError("cutoff must be a valid date")
    dates = pd.DatetimeIndex(pd.to_datetime(list(forecast_dates))).normalize().unique()
    dates = dates.sort_values()
    if dates.empty:
        raise ValueError("forecast_dates must not be empty")
    if dates.min() <= cutoff_date:
        raise ValueError("every forecast date must be after cutoff")

    prepared = history.loc[:, required].copy()
    prepared[DATE_COLUMN] = pd.to_datetime(prepared[DATE_COLUMN]).dt.normalize()
    prepared[TARGET_COLUMN] = prepared[TARGET_COLUMN].astype("float64")
    prepared = prepared.loc[prepared[DATE_COLUMN].le(cutoff_date)]
    if prepared.empty:
        raise ValueError("history contains no rows on or before cutoff")
    if prepared.duplicated([DATE_COLUMN, *SERIES_COLUMNS]).any():
        raise ValueError("history contains duplicate date-store-family rows")
    if not np.isfinite(prepared[TARGET_COLUMN]).all():
        raise ValueError("historical sales must contain only finite values")

    if series is None:
        series_frame = prepared[SERIES_COLUMNS].drop_duplicates().copy()
    else:
        missing_series = [column for column in SERIES_COLUMNS if column not in series]
        if missing_series:
            raise KeyError(
                "series is missing required columns: " + ", ".join(missing_series)
            )
        series_frame = series[SERIES_COLUMNS].drop_duplicates().copy()
    if series_frame.empty:
        raise ValueError("series must contain at least one store-family combination")

    return prepared, dates, cutoff_date, series_frame


def _forecast_grid(series: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return series.merge(
        pd.DataFrame({DATE_COLUMN: dates}), how="cross", validate="many_to_many"
    )


def _fallback_values(history: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    series_median = (
        history.groupby(SERIES_COLUMNS, as_index=False, observed=True)[TARGET_COLUMN]
        .median()
        .rename(columns={TARGET_COLUMN: "_series_fallback"})
    )
    global_median = float(history[TARGET_COLUMN].median())
    if not np.isfinite(global_median):
        global_median = 0.0
    return series_median, max(global_median, 0.0)


def _finalize_predictions(
    predictions: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Fill unavailable references from pre-cutoff history and enforce contracts."""
    series_median, global_median = _fallback_values(history)
    result = predictions.merge(
        series_median, on=SERIES_COLUMNS, how="left", validate="many_to_one"
    )
    result[PREDICTION_COLUMN] = (
        result[PREDICTION_COLUMN]
        .fillna(result["_series_fallback"])
        .fillna(global_median)
        .clip(lower=0.0)
        .astype("float64")
    )
    result = result.drop(columns="_series_fallback")
    output_columns = [DATE_COLUMN, *SERIES_COLUMNS, PREDICTION_COLUMN]
    result = result[output_columns].sort_values(
        [*SERIES_COLUMNS, DATE_COLUMN], kind="stable"
    )
    result = result.reset_index(drop=True)
    if result.duplicated([DATE_COLUMN, *SERIES_COLUMNS]).any():
        raise RuntimeError("baseline output contains duplicate forecast grain")
    if result[PREDICTION_COLUMN].isna().any() or result[PREDICTION_COLUMN].lt(0).any():
        raise RuntimeError("baseline predictions must be finite and nonnegative")
    return result


def last_value_naive(
    history: pd.DataFrame,
    forecast_dates: Iterable[object],
    cutoff: object,
    series: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Repeat each series' final observed pre-cutoff Sales Volume."""
    prepared, dates, _, series_frame = _prepare_inputs(
        history, forecast_dates, cutoff, series
    )
    last_values = (
        prepared.sort_values(DATE_COLUMN, kind="stable")
        .groupby(SERIES_COLUMNS, observed=True)
        .tail(1)[[*SERIES_COLUMNS, TARGET_COLUMN]]
        .rename(columns={TARGET_COLUMN: PREDICTION_COLUMN})
    )
    predictions = _forecast_grid(series_frame, dates).merge(
        last_values, on=SERIES_COLUMNS, how="left", validate="many_to_one"
    )
    return _finalize_predictions(predictions, prepared)


def seasonal_naive(
    history: pd.DataFrame,
    forecast_dates: Iterable[object],
    cutoff: object,
    lag_days: int,
    series: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Forecast from a seasonal reference that is always on/before cutoff.

    For a target whose direct ``target_date - lag_days`` reference lies after the
    cutoff, the reference is moved backward by additional seasonal offsets. This
    is equivalent to recursive seasonal-naive prediction but never reads an actual
    target from inside the validation horizon.
    """
    lag = _positive_integer(lag_days, "lag_days")
    prepared, dates, cutoff_date, series_frame = _prepare_inputs(
        history, forecast_dates, cutoff, series
    )
    references: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for target_date in dates:
        reference_date = target_date - pd.Timedelta(days=lag)
        while reference_date > cutoff_date:
            reference_date -= pd.Timedelta(days=lag)
        references.append((target_date, reference_date))
    reference_map = pd.DataFrame(
        references, columns=[DATE_COLUMN, "_reference_date"]
    )
    historical_values = prepared.rename(
        columns={DATE_COLUMN: "_reference_date", TARGET_COLUMN: PREDICTION_COLUMN}
    )
    predictions = (
        _forecast_grid(series_frame, dates)
        .merge(reference_map, on=DATE_COLUMN, validate="many_to_one")
        .merge(
            historical_values[[*SERIES_COLUMNS, "_reference_date", PREDICTION_COLUMN]],
            on=[*SERIES_COLUMNS, "_reference_date"],
            how="left",
            validate="many_to_one",
        )
        .drop(columns="_reference_date")
    )
    return _finalize_predictions(predictions, prepared)


def weekday_historical_median(
    history: pd.DataFrame,
    forecast_dates: Iterable[object],
    cutoff: object,
    series: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Use each series' pre-cutoff historical median for the target weekday."""
    prepared, dates, _, series_frame = _prepare_inputs(
        history, forecast_dates, cutoff, series
    )
    prepared["_weekday"] = prepared[DATE_COLUMN].dt.dayofweek
    weekday_values = (
        prepared.groupby(
            [*SERIES_COLUMNS, "_weekday"], as_index=False, observed=True
        )[TARGET_COLUMN]
        .median()
        .rename(columns={TARGET_COLUMN: PREDICTION_COLUMN})
    )
    predictions = _forecast_grid(series_frame, dates)
    predictions["_weekday"] = predictions[DATE_COLUMN].dt.dayofweek
    predictions = predictions.merge(
        weekday_values,
        on=[*SERIES_COLUMNS, "_weekday"],
        how="left",
        validate="many_to_one",
    ).drop(columns="_weekday")
    return _finalize_predictions(predictions, prepared)


def rolling_historical_median(
    history: pd.DataFrame,
    forecast_dates: Iterable[object],
    cutoff: object,
    window_days: int = 28,
    series: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Repeat the median from the final pre-cutoff calendar window per series."""
    window = _positive_integer(window_days, "window_days")
    prepared, dates, cutoff_date, series_frame = _prepare_inputs(
        history, forecast_dates, cutoff, series
    )
    window_start = cutoff_date - pd.Timedelta(days=window - 1)
    window_history = prepared.loc[prepared[DATE_COLUMN].ge(window_start)]
    rolling_values = (
        window_history.groupby(SERIES_COLUMNS, as_index=False, observed=True)[
            TARGET_COLUMN
        ]
        .median()
        .rename(columns={TARGET_COLUMN: PREDICTION_COLUMN})
    )
    predictions = _forecast_grid(series_frame, dates).merge(
        rolling_values, on=SERIES_COLUMNS, how="left", validate="many_to_one"
    )
    return _finalize_predictions(predictions, prepared)


def forecast_baseline(
    history: pd.DataFrame,
    forecast_dates: Iterable[object],
    cutoff: object,
    model: str,
    series: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Dispatch one named baseline with the repository's documented defaults."""
    if model == "last_value_naive":
        return last_value_naive(history, forecast_dates, cutoff, series)
    if model.startswith("seasonal_naive_") and model.endswith("d"):
        lag_text = model.removeprefix("seasonal_naive_").removesuffix("d")
        if lag_text.isdigit() and int(lag_text) in {7, 14, 28}:
            return seasonal_naive(
                history, forecast_dates, cutoff, int(lag_text), series
            )
    if model == "weekday_historical_median":
        return weekday_historical_median(history, forecast_dates, cutoff, series)
    if model == "rolling_historical_median_28d":
        return rolling_historical_median(
            history, forecast_dates, cutoff, window_days=28, series=series
        )
    raise ValueError(f"unsupported baseline model: {model}")
