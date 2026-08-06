"""Tests for the complete date-by-store warehouse dimension."""

import pandas as pd
import pytest

from src.data.build_store_date_dimension import (
    DIM_STORE_DATE_COLUMNS,
    build_dim_store_date,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    dim_date = pd.DataFrame(
        {"date_key": [20200101, 20200102, 20200103], "full_date": dates}
    )
    dim_store = pd.DataFrame(
        {"store_key": [1, 2], "store_nbr": [10, 20]}
    )
    holidays = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "store_nbr": [10, 20],
            "holiday_count": [2, 1],
            "holiday_descriptions": ["A | B", "C"],
            "holiday_types": ["Event | Holiday", "Work Day"],
            "holiday_locales": ["Local | National", "Regional"],
            "is_holiday": [1, 0],
            "is_work_day": [0, 1],
            "is_event": [1, 0],
        }
    )
    train = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
            "store_nbr": [10, 10, 20],
            "family": ["A", "B", "A"],
            "sales": [1.0, 2.0, 3.0],
        }
    )
    transactions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-03"]),
            "store_nbr": [10, 20],
            "transactions": [5, 7],
        }
    )
    return dim_date, dim_store, holidays, train, transactions


def test_build_dim_store_date_has_complete_grid_keys_and_grain() -> None:
    dim_date, dim_store, holidays, train, transactions = _inputs()

    result = build_dim_store_date(dim_date, dim_store, holidays, train, transactions)

    assert len(result) == len(dim_date) * len(dim_store) == 6
    assert result.columns.tolist() == DIM_STORE_DATE_COLUMNS
    assert not result[["date_key", "store_key"]].isna().any().any()
    assert not result.duplicated(["date_key", "store_key"]).any()
    assert result["date_store_key"].is_unique
    assert (
        result["date_store_key"]
        == result["date_key"].astype("int64") * 100 + result["store_key"]
    ).all()


def test_build_dim_store_date_preserves_holidays_and_fills_consistent_defaults() -> None:
    dim_date, dim_store, holidays, train, transactions = _inputs()

    result = build_dim_store_date(dim_date, dim_store, holidays, train, transactions)

    holiday_row = result.loc[
        result["date_key"].eq(20200102) & result["store_key"].eq(1)
    ].iloc[0]
    assert holiday_row["holiday_count"] == 2
    assert holiday_row["holiday_descriptions"] == "A | B"
    assert holiday_row["holiday_types"] == "Event | Holiday"
    assert holiday_row["holiday_locales"] == "Local | National"
    assert holiday_row["is_holiday"] == 1
    assert holiday_row["is_event"] == 1

    nonholiday = result.loc[
        result["date_key"].eq(20200101) & result["store_key"].eq(2)
    ].iloc[0]
    assert nonholiday["holiday_count"] == 0
    assert nonholiday[["is_holiday", "is_work_day", "is_event"]].eq(0).all()
    assert nonholiday[
        ["holiday_descriptions", "holiday_types", "holiday_locales"]
    ].eq("").all()


def test_build_dim_store_date_flags_source_row_existence_not_measure_values() -> None:
    dim_date, dim_store, holidays, train, transactions = _inputs()
    train.loc[0, "sales"] = 0.0
    transactions.loc[0, "transactions"] = 0

    result = build_dim_store_date(dim_date, dim_store, holidays, train, transactions)

    observed = result.loc[
        result["date_key"].eq(20200101) & result["store_key"].eq(1)
    ].iloc[0]
    missing = result.loc[
        result["date_key"].eq(20200102) & result["store_key"].eq(1)
    ].iloc[0]
    assert observed["has_sales_observation"] == 1
    assert observed["has_transaction_observation"] == 1
    assert missing["has_sales_observation"] == 0
    assert missing["has_transaction_observation"] == 0


def test_build_dim_store_date_rejects_store_key_at_or_above_100() -> None:
    dim_date, dim_store, holidays, train, transactions = _inputs()
    dim_store.loc[1, "store_key"] = 100

    with pytest.raises(ValueError, match="different date_store_key strategy"):
        build_dim_store_date(dim_date, dim_store, holidays, train, transactions)


@pytest.mark.parametrize("source_name", ["train", "transactions"])
def test_build_dim_store_date_rejects_unmapped_observations(source_name: str) -> None:
    dim_date, dim_store, holidays, train, transactions = _inputs()
    if source_name == "train":
        train.loc[0, "store_nbr"] = 99
    else:
        transactions.loc[0, "store_nbr"] = 99

    with pytest.raises(ValueError, match="unmapped store_nbr"):
        build_dim_store_date(dim_date, dim_store, holidays, train, transactions)


def test_build_dim_store_date_does_not_modify_inputs() -> None:
    inputs = _inputs()
    originals = tuple(frame.copy(deep=True) for frame in inputs)

    build_dim_store_date(*inputs)

    for frame, original in zip(inputs, originals):
        pd.testing.assert_frame_equal(frame, original)
