import pandas as pd
import pytest

from src.features.build_forecast_frame import GRAIN, build_forecast_frame
from src.features.exogenous_features import EXOGENOUS_CANDIDATES_REQUIRING_REVIEW


def _stores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "store_nbr": [1, 2],
            "city": ["Quito", "Cuenca"],
            "state": ["Pichincha", "Azuay"],
            "type": ["A", "B"],
            "cluster": [1, 2],
        }
    )


def _holidays() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-03"]),
            "type": ["Holiday"],
            "locale": ["National"],
            "locale_name": ["Ecuador"],
            "description": ["Known holiday"],
            "transferred": [False],
        }
    )


def _train() -> pd.DataFrame:
    rows = []
    row_id = 0
    for date in pd.to_datetime(["2020-01-01", "2020-01-02"]):
        for store_nbr in [1, 2]:
            for family in ["A", "B"]:
                rows.append(
                    {
                        "id": row_id,
                        "date": date,
                        "store_nbr": store_nbr,
                        "family": family,
                        "sales": float(row_id),
                        "onpromotion": 0,
                    }
                )
                row_id += 1
    return pd.DataFrame(rows)


def _test() -> pd.DataFrame:
    rows = []
    row_id = 100
    for date in pd.to_datetime(["2020-01-03", "2020-01-04"]):
        for store_nbr in [1, 2]:
            for family in ["A", "B"]:
                rows.append(
                    {
                        "id": row_id,
                        "date": date,
                        "store_nbr": store_nbr,
                        "family": family,
                        "onpromotion": row_id % 2,
                    }
                )
                row_id += 1
    return pd.DataFrame(rows)


def test_frame_has_unique_complete_store_family_date_grain() -> None:
    frame = build_forecast_frame(_train(), _test(), _stores(), _holidays())

    assert len(frame) == 4 * 2 * 2
    assert not frame.duplicated(GRAIN).any()
    assert frame["date"].drop_duplicates().tolist() == pd.date_range(
        "2020-01-01", "2020-01-04"
    ).tolist()


def test_original_test_ids_are_preserved_exactly() -> None:
    test = _test()
    frame = build_forecast_frame(_train(), test, _stores(), _holidays())
    future = frame.loc[frame["is_future"].eq(1) & frame["source_row_observed"].eq(1)]

    actual = future.set_index(GRAIN)["test_id"].sort_index().astype("uint32")
    expected = test.set_index(GRAIN)["id"].sort_index().astype("uint32")
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_missing_historical_observation_remains_missing_not_zero() -> None:
    train = _train()
    removed = (
        train["date"].eq(pd.Timestamp("2020-01-02"))
        & train["store_nbr"].eq(2)
        & train["family"].eq("B")
    )
    train = train.loc[~removed]
    frame = build_forecast_frame(train, _test(), _stores(), _holidays())
    missing = frame.loc[
        frame["date"].eq("2020-01-02")
        & frame["store_nbr"].eq(2)
        & frame["family"].eq("B")
    ].iloc[0]

    assert pd.isna(missing["sales"])
    assert missing["sales_observed"] == 0
    assert missing["source_row_observed"] == 0
    assert missing["is_historical"] == 1


def test_known_future_calendar_promotion_metadata_and_holiday_are_available() -> None:
    frame = build_forecast_frame(_train(), _test(), _stores(), _holidays())
    future = frame.loc[frame["is_future"].eq(1)]
    required = [
        "day_of_week", "week_of_year", "month", "quarter", "year",
        "is_weekend", "is_month_start", "is_month_end", "is_payday",
        "store_type", "cluster", "city", "state", "family",
        "onpromotion", "promotion_active", "holiday_count", "is_holiday",
        "is_work_day", "is_event",
    ]

    assert not future[required].isna().any().any()
    assert future.loc[future["date"].eq("2020-01-03"), "is_holiday"].eq(1).all()
    assert future.loc[future["date"].eq("2020-01-04"), "is_holiday"].eq(0).all()


def test_missing_future_promotion_schedule_raises() -> None:
    test = _test().drop(index=0)

    with pytest.raises(ValueError, match="known-future features"):
        build_forecast_frame(_train(), test, _stores(), _holidays())


def test_unsafe_features_are_not_created() -> None:
    frame = build_forecast_frame(_train(), _test(), _stores(), _holidays())

    forbidden = {
        "transactions", "dcoilwtico", "future_sales", "sales_anomaly",
        "forecast_readiness", "sales_lag_7", "sales_rolling_mean_28",
    }
    assert forbidden.isdisjoint(frame.columns)
    assert "oil" in EXOGENOUS_CANDIDATES_REQUIRING_REVIEW
