"""Shared recursive multi-step inference for backtests and final forecasts."""

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.features.lag_features import DEFAULT_LAGS
from src.modeling.predict import predict_sales


GRAIN = ["date", "store_nbr", "family"]
PredictionFunction = Callable[[object, pd.DataFrame], pd.DataFrame]


def _validate_dense_calendar(frame: pd.DataFrame) -> None:
    """Require one row per series per calendar date across the input bounds."""
    required = [*GRAIN, "sales"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError("known_frame is missing columns: " + ", ".join(missing))
    if frame.empty or frame[GRAIN].isna().any().any():
        raise ValueError("known_frame must have a non-empty complete grain")
    if frame.duplicated(GRAIN).any():
        raise ValueError("known_frame contains duplicate date-store-family rows")
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"])).normalize()
    expected_dates = pd.date_range(dates.min(), dates.max())
    if not dates.unique().sort_values().equals(expected_dates):
        raise ValueError("known_frame dates must form a dense calendar")
    series_count = frame[["store_nbr", "family"]].drop_duplicates().shape[0]
    if not pd.Series(dates).value_counts().eq(series_count).all():
        raise ValueError(
            "known_frame must contain every series on every calendar date"
        )


def recursive_forecast(
    model: object,
    known_frame: pd.DataFrame,
    forecast_origin: object,
    forecast_start: object,
    forecast_end: object,
    *,
    prediction_function: PredictionFunction = predict_sales,
) -> pd.DataFrame:
    """Forecast sequentially and feed only prior predictions into history.

    Training and inference share the same calendar-day lag and shifted rolling
    builders. Before inference, every target after forecast_origin is masked,
    including actual validation targets present in known_frame. D+1 therefore
    uses historical actuals through the origin. Its prediction is inserted into
    a private temporary history, allowing D+2 and later features to use earlier
    predictions but never actual future validation or test targets.

    Missing historical calendar observations remain NaN and distinct from
    observed zero sales. Rolling windows may contain historical actuals and
    earlier predictions only.
    """
    _validate_dense_calendar(known_frame)
    origin = pd.Timestamp(forecast_origin).normalize()
    start = pd.Timestamp(forecast_start).normalize()
    end = pd.Timestamp(forecast_end).normalize()
    if start != origin + pd.Timedelta(days=1) or end < start:
        raise ValueError(
            "forecast_start must be the day after origin and not exceed forecast_end"
        )
    dates = pd.date_range(start, end)
    available = pd.DatetimeIndex(pd.to_datetime(known_frame["date"])).normalize()
    if not dates.isin(available.unique()).all():
        raise ValueError("known_frame does not cover the complete forecast horizon")

    working = known_frame.copy()
    working["date"] = pd.to_datetime(working["date"]).dt.normalize()
    working.loc[working["date"].gt(origin), "sales"] = np.nan
    series_index = pd.MultiIndex.from_frame(
        working[["store_nbr", "family"]].drop_duplicates().sort_values(
            ["store_nbr", "family"], kind="stable"
        )
    )
    sales_matrix = working.pivot(
        index="date", columns=["store_nbr", "family"], values="sales"
    ).reindex(columns=series_index)
    predictions: list[pd.DataFrame] = []

    for forecast_date in dates:
        day_features = working.loc[working["date"].eq(forecast_date)].copy()
        day_features = day_features.set_index(["store_nbr", "family"]).reindex(
            series_index
        )
        for lag in DEFAULT_LAGS:
            reference_date = forecast_date - pd.Timedelta(days=lag)
            day_features[f"sales_lag_{lag}"] = sales_matrix.reindex(
                [reference_date]
            ).iloc[0].to_numpy()

        def window_values(window: int) -> pd.DataFrame:
            window_end = forecast_date - pd.Timedelta(days=1)
            window_start = forecast_date - pd.Timedelta(days=window)
            expected_window = pd.date_range(window_start, window_end)
            return sales_matrix.reindex(expected_window)

        for window in (7, 14, 28, 56):
            day_features[f"rolling_mean_{window}"] = window_values(window).mean(
                axis=0, skipna=False
            ).to_numpy()
        for window in (7, 28):
            day_features[f"rolling_median_{window}"] = window_values(window).median(
                axis=0, skipna=False
            ).to_numpy()
        window_28 = window_values(28)
        day_features["rolling_std_28"] = window_28.std(
            axis=0, skipna=False
        ).to_numpy()
        day_features["rolling_min_28"] = window_28.min(
            axis=0, skipna=False
        ).to_numpy()
        day_features["rolling_max_28"] = window_28.max(
            axis=0, skipna=False
        ).to_numpy()
        day_features["rolling_zero_rate_28"] = window_28.eq(0).where(
            window_28.notna()
        ).mean(axis=0, skipna=False).to_numpy()
        day_features = day_features.reset_index()
        expected_count = working.loc[
            working["date"].eq(forecast_date), ["store_nbr", "family"]
        ].shape[0]
        if len(day_features) != expected_count:
            raise RuntimeError("recursive day does not cover every forecast series")

        day_prediction = prediction_function(model, day_features)
        required_prediction = [*GRAIN, "prediction"]
        missing = [
            column for column in required_prediction if column not in day_prediction
        ]
        if missing:
            raise KeyError(
                "prediction output is missing columns: " + ", ".join(missing)
            )
        if len(day_prediction) != len(day_features) or day_prediction.duplicated(
            GRAIN
        ).any():
            raise RuntimeError("prediction output has invalid daily grain")
        aligned = day_features[GRAIN].merge(
            day_prediction[required_prediction],
            on=GRAIN,
            how="left",
            validate="one_to_one",
        )
        values = aligned["prediction"].to_numpy(dtype="float64")
        if not np.isfinite(values).all() or (values < 0).any():
            raise RuntimeError("recursive predictions must be finite and nonnegative")

        indexed_prediction = aligned.set_index(["store_nbr", "family"])[
            "prediction"
        ].reindex(series_index)
        sales_matrix.loc[forecast_date] = indexed_prediction.to_numpy()
        predictions.append(aligned)

    result = pd.concat(predictions, ignore_index=True)
    series_count = known_frame[["store_nbr", "family"]].drop_duplicates().shape[0]
    if len(result) != len(dates) * series_count:
        raise RuntimeError("recursive forecast row count is incomplete")
    return result.sort_values(GRAIN, kind="stable").reset_index(drop=True)
