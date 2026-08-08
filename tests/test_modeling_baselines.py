"""Leakage, grain, offset, and nonnegativity contracts for baselines."""

import pandas as pd
import pytest

from src.modeling.baselines import (
    BASELINE_MODELS,
    forecast_baseline,
    last_value_naive,
    rolling_historical_median,
    seasonal_naive,
)


@pytest.fixture
def daily_history() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", "2024-01-31", freq="D")
    rows = []
    for store_nbr, family, multiplier in [(1, "A", 1.0), (2, "B", 10.0)]:
        rows.extend(
            {
                "date": date,
                "store_nbr": store_nbr,
                "family": family,
                "sales": float(date.day * multiplier),
            }
            for date in dates
        )
    return pd.DataFrame(rows)


def test_all_baselines_return_complete_unique_forecast_grain(
    daily_history: pd.DataFrame,
) -> None:
    forecast_dates = pd.date_range("2024-02-01", periods=16, freq="D")
    series = daily_history[["store_nbr", "family"]].drop_duplicates()

    for model in BASELINE_MODELS:
        result = forecast_baseline(
            daily_history,
            forecast_dates,
            cutoff="2024-01-31",
            model=model,
            series=series,
        )

        assert len(result) == len(series) * len(forecast_dates)
        assert not result.duplicated(["date", "store_nbr", "family"]).any()
        assert set(result["date"]) == set(forecast_dates)
        assert result["prediction"].ge(0).all()


def test_future_validation_actuals_cannot_change_predictions(
    daily_history: pd.DataFrame,
) -> None:
    forecast_dates = pd.date_range("2024-02-01", periods=16, freq="D")
    future = pd.DataFrame(
        {
            "date": forecast_dates,
            "store_nbr": 1,
            "family": "A",
            "sales": 999_999.0,
        }
    )
    history_with_validation_targets = pd.concat(
        [daily_history, future], ignore_index=True
    )
    series = daily_history[["store_nbr", "family"]].drop_duplicates()

    for model in BASELINE_MODELS:
        clean = forecast_baseline(
            daily_history, forecast_dates, "2024-01-31", model, series
        )
        contaminated = forecast_baseline(
            history_with_validation_targets,
            forecast_dates,
            "2024-01-31",
            model,
            series,
        )

        pd.testing.assert_frame_equal(clean, contaminated)


@pytest.mark.parametrize("lag_days", [7, 14, 28])
def test_seasonal_offsets_reference_only_pre_cutoff_dates(
    daily_history: pd.DataFrame,
    lag_days: int,
) -> None:
    forecast_dates = pd.date_range("2024-02-01", periods=16, freq="D")
    result = seasonal_naive(
        daily_history,
        forecast_dates,
        cutoff="2024-01-31",
        lag_days=lag_days,
        series=pd.DataFrame({"store_nbr": [1], "family": ["A"]}),
    ).set_index("date")

    first_reference = pd.Timestamp("2024-02-01") - pd.Timedelta(days=lag_days)
    assert result.loc[pd.Timestamp("2024-02-01"), "prediction"] == float(
        first_reference.day
    )
    if lag_days < 16:
        recursive_target = pd.Timestamp("2024-02-01") + pd.Timedelta(days=lag_days)
        assert result.loc[recursive_target, "prediction"] == float(first_reference.day)


def test_last_value_and_rolling_median_known_answers(
    daily_history: pd.DataFrame,
) -> None:
    series = pd.DataFrame({"store_nbr": [1], "family": ["A"]})
    last = last_value_naive(
        daily_history, ["2024-02-01"], "2024-01-31", series
    )
    rolling = rolling_historical_median(
        daily_history,
        ["2024-02-01"],
        "2024-01-31",
        window_days=28,
        series=series,
    )

    assert last.loc[0, "prediction"] == 31.0
    assert rolling.loc[0, "prediction"] == 17.5


def test_negative_historical_values_never_create_negative_predictions() -> None:
    history = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=35, freq="D"),
            "store_nbr": 1,
            "family": "A",
            "sales": -5.0,
        }
    )

    for model in BASELINE_MODELS:
        result = forecast_baseline(
            history,
            pd.date_range("2024-02-05", periods=16, freq="D"),
            cutoff="2024-02-04",
            model=model,
        )
        assert result["prediction"].eq(0.0).all()
