import numpy as np
import pandas as pd

from src.features.lag_features import DEFAULT_LAGS, add_sales_lag_features
from src.features.rolling_features import (
    ROLLING_FEATURE_COLUMNS,
    add_sales_rolling_features,
)


def _series(periods: int = 400) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=periods, freq="D")
    rows = []
    for store_nbr, family, offset in [(1, "A", 0.0), (1, "B", 1_000.0), (2, "A", 2_000.0)]:
        rows.extend(
            {
                "date": date,
                "store_nbr": store_nbr,
                "family": family,
                "sales": float(day + 1) + offset,
            }
            for day, date in enumerate(dates)
        )
    return pd.DataFrame(rows)


def test_all_required_lags_are_correct_and_grouped_by_store_family() -> None:
    frame = _series()
    result = add_sales_lag_features(frame)
    target_date = pd.Timestamp("2024-01-31")

    for store_nbr, family, offset in [(1, "A", 0.0), (1, "B", 1_000.0), (2, "A", 2_000.0)]:
        row = result.loc[
            result["date"].eq(target_date)
            & result["store_nbr"].eq(store_nbr)
            & result["family"].eq(family)
        ].iloc[0]
        position = (target_date - pd.Timestamp("2023-01-01")).days + 1
        for lag in DEFAULT_LAGS:
            assert row[f"sales_lag_{lag}"] == position - lag + offset


def test_lag_never_reads_current_target() -> None:
    frame = _series(periods=10)
    changed = frame.copy()
    target_date = pd.Timestamp("2023-01-06")
    changed.loc[
        changed["date"].eq(target_date)
        & changed["store_nbr"].eq(1)
        & changed["family"].eq("A"),
        "sales",
    ] = 999_999.0

    original_features = add_sales_lag_features(frame, lags=[1, 2, 3])
    changed_features = add_sales_lag_features(changed, lags=[1, 2, 3])
    columns = ["sales_lag_1", "sales_lag_2", "sales_lag_3"]
    original_row = original_features.loc[
        original_features["date"].eq(target_date)
        & original_features["store_nbr"].eq(1)
        & original_features["family"].eq("A"),
        columns,
    ]
    changed_row = changed_features.loc[
        changed_features["date"].eq(target_date)
        & changed_features["store_nbr"].eq(1)
        & changed_features["family"].eq("A"),
        columns,
    ]
    pd.testing.assert_frame_equal(original_row, changed_row)


def test_shifted_rolling_features_have_known_values() -> None:
    frame = _series(periods=60)
    result = add_sales_rolling_features(frame).set_index("date")

    day_8 = result.loc[pd.Timestamp("2023-01-08")].query(
        "store_nbr == 1 and family == 'A'"
    ).iloc[0]
    assert day_8["rolling_mean_7"] == 4.0
    assert day_8["rolling_median_7"] == 4.0

    day_29_rows = result.loc[pd.Timestamp("2023-01-29")]
    day_29 = day_29_rows.query("store_nbr == 1 and family == 'A'").iloc[0]
    assert day_29["rolling_mean_28"] == 14.5
    assert day_29["rolling_median_28"] == 14.5
    assert day_29["rolling_min_28"] == 1.0
    assert day_29["rolling_max_28"] == 28.0
    assert day_29["rolling_std_28"] == pd.Series(range(1, 29)).std()
    assert day_29["rolling_zero_rate_28"] == 0.0
    other_series = day_29_rows.query("store_nbr == 1 and family == 'B'").iloc[0]
    assert other_series["rolling_mean_28"] == 1_014.5


def test_rolling_zero_rate_uses_only_shifted_observed_history() -> None:
    frame = _series(periods=30).query("store_nbr == 1 and family == 'A'").reset_index(drop=True)
    frame.loc[:13, "sales"] = 0.0
    result = add_sales_rolling_features(frame)

    assert result.loc[28, "rolling_zero_rate_28"] == 0.5


def test_first_valid_dates_follow_shift_and_full_window() -> None:
    frame = _series(periods=60).query("store_nbr == 1 and family == 'A'")
    result = add_sales_rolling_features(frame)

    assert result["rolling_mean_7"].first_valid_index() == 7
    assert result["rolling_mean_14"].first_valid_index() == 14
    assert result["rolling_mean_28"].first_valid_index() == 28
    assert result["rolling_mean_56"].first_valid_index() == 56
    for column in ROLLING_FEATURE_COLUMNS:
        assert pd.isna(result.loc[0, column])


def test_missing_observation_is_not_treated_as_zero() -> None:
    frame = _series(periods=40).query("store_nbr == 1 and family == 'A'").reset_index(drop=True)
    frame.loc[10, "sales"] = np.nan
    result = add_sales_lag_features(frame, lags=[1])
    result = add_sales_rolling_features(result)

    assert pd.isna(result.loc[11, "sales_lag_1"])
    assert pd.isna(result.loc[29, "rolling_mean_28"])
    assert pd.isna(result.loc[29, "rolling_zero_rate_28"])


def test_future_target_changes_cannot_affect_any_horizon_feature() -> None:
    frame = _series(periods=56).query("store_nbr == 1 and family == 'A'").reset_index(drop=True)
    cutoff = pd.Timestamp("2023-02-09")
    future_mask = frame["date"].gt(cutoff)
    contaminated = frame.copy()
    contaminated.loc[future_mask, "sales"] = np.arange(future_mask.sum()) * 1_000_000.0

    original = add_sales_lag_features(frame, forecast_origin=cutoff)
    original = add_sales_rolling_features(original, forecast_origin=cutoff)
    changed = add_sales_lag_features(contaminated, forecast_origin=cutoff)
    changed = add_sales_rolling_features(changed, forecast_origin=cutoff)
    feature_columns = [
        *[f"sales_lag_{lag}" for lag in DEFAULT_LAGS],
        *ROLLING_FEATURE_COLUMNS,
    ]

    pd.testing.assert_frame_equal(
        original.loc[future_mask, feature_columns].reset_index(drop=True),
        changed.loc[future_mask, feature_columns].reset_index(drop=True),
    )
    assert original.loc[future_mask, "sales_lag_1"].iloc[0] == 40.0
    assert original.loc[future_mask, "sales_lag_1"].iloc[1:].isna().all()


def test_horizon_safe_features_preserve_input_grain_and_row_count() -> None:
    frame = _series(periods=20)
    result = add_sales_lag_features(frame, forecast_origin="2023-01-10")
    result = add_sales_rolling_features(result, forecast_origin="2023-01-10")

    assert len(result) == len(frame)
    assert not result.duplicated(["date", "store_nbr", "family"]).any()
    pd.testing.assert_frame_equal(
        result[["date", "store_nbr", "family", "sales"]],
        frame[["date", "store_nbr", "family", "sales"]],
    )
