"""Tests for the daily sales fact builder."""

from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_facts import (
    build_fact_daily_sales,
    build_fact_oil_price,
    build_fact_store_transactions,
)


def _date_store_dimension(
    dim_date: pd.DataFrame,
    dim_store: pd.DataFrame,
) -> pd.DataFrame:
    result = dim_date[["date_key"]].merge(
        dim_store[["store_key"]], how="cross"
    )
    result["date_store_key"] = result["date_key"] * 100 + result["store_key"]
    return result[["date_store_key", "date_key", "store_key"]]


def _inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    train = pd.DataFrame(
        {
            "id": [10, 11, 12],
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
            "store_nbr": [1, 1, 2],
            "family": ["A", "B", "A"],
            "sales": pd.Series([1.1234567, 0.0, 3.7654321], dtype="float64"),
            "onpromotion": [1, 0, 2],
            "is_promotion": [1, 0, 1],
        }
    )
    dim_date = pd.DataFrame(
        {
            "date_key": [20200101, 20200102],
            "full_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        }
    )
    dim_store = pd.DataFrame({"store_key": [1, 2], "store_nbr": [1, 2]})
    dim_family = pd.DataFrame({"family_key": [1, 2], "family": ["A", "B"]})
    dim_store_date = _date_store_dimension(dim_date, dim_store)
    return train, dim_date, dim_store, dim_family, dim_store_date


def test_fact_daily_sales_reconciles_rows_and_measures() -> None:
    """Fact rows, sales, and promotion totals reconcile to cleaned train."""
    train, dim_date, dim_store, dim_family, dim_store_date = _inputs()
    original = train.copy(deep=True)

    fact = build_fact_daily_sales(
        train, dim_date, dim_store, dim_family, dim_store_date
    )

    assert len(fact) == len(train)
    assert fact["sales"].sum() == pytest.approx(train["sales"].sum())
    assert fact["onpromotion"].sum() == train["onpromotion"].sum()
    assert pd.api.types.is_float_dtype(fact["sales"])
    pd.testing.assert_frame_equal(train, original)


def test_fact_daily_sales_has_expected_columns_and_unique_grain() -> None:
    """Final fact uses only surrogate keys and has no duplicate grain."""
    train, dim_date, dim_store, dim_family, dim_store_date = _inputs()

    fact = build_fact_daily_sales(
        train, dim_date, dim_store, dim_family, dim_store_date
    )

    assert fact.columns.tolist() == [
        "sales_id",
        "date_key",
        "store_key",
        "date_store_key",
        "family_key",
        "sales",
        "onpromotion",
        "is_promotion",
    ]
    assert not fact.duplicated(["date_key", "store_key", "family_key"]).any()
    assert not fact[["date_key", "store_key", "family_key"]].isna().any().any()
    assert {"date", "store_nbr", "family"}.isdisjoint(fact.columns)


@pytest.mark.parametrize(
    ("dimension_name", "row_to_remove", "missing_business_key"),
    [
        ("date", 1, "date"),
        ("store", 1, "store_nbr"),
        ("family", 1, "family"),
    ],
)
def test_fact_daily_sales_rejects_unmapped_business_keys(
    dimension_name: str,
    row_to_remove: int,
    missing_business_key: str,
) -> None:
    """Any missing dimension mapping raises an error naming its business key."""
    train, dim_date, dim_store, dim_family, dim_store_date = _inputs()
    dimensions = {
        "date": dim_date,
        "store": dim_store,
        "family": dim_family,
    }
    dimensions[dimension_name] = dimensions[dimension_name].drop(index=row_to_remove)

    with pytest.raises(ValueError, match=missing_business_key):
        build_fact_daily_sales(
            train,
            dimensions["date"],
            dimensions["store"],
            dimensions["family"],
            dim_store_date,
        )


def test_fact_daily_sales_rejects_duplicate_grain() -> None:
    """Multiple source rows at one dimensional grain are rejected."""
    train, dim_date, dim_store, dim_family, dim_store_date = _inputs()
    duplicate = train.iloc[[0]].copy()
    duplicate["id"] = 99

    with pytest.raises(ValueError, match="duplicate"):
        build_fact_daily_sales(
            pd.concat([train, duplicate], ignore_index=True),
            dim_date,
            dim_store,
            dim_family,
            dim_store_date,
        )


def _transaction_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    transactions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-01", "2020-01-01"]),
            "store_nbr": [2, 1, 2],
            "transactions": [30, 10, 20],
        }
    )
    dim_date = pd.DataFrame(
        {
            "date_key": [20200101, 20200102],
            "full_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        }
    )
    dim_store = pd.DataFrame({"store_key": [1, 2], "store_nbr": [1, 2]})
    return transactions, dim_date, dim_store


def test_fact_store_transactions_reconciles_row_count_and_total() -> None:
    """Transaction fact preserves one row and the measure total per source row."""
    transactions, dim_date, dim_store = _transaction_inputs()
    original = transactions.copy(deep=True)

    fact = build_fact_store_transactions(transactions, dim_date, dim_store)

    assert len(fact) == len(transactions)
    assert fact["transactions"].sum() == transactions["transactions"].sum()
    assert fact.columns.tolist() == [
        "date_key",
        "store_key",
        "transactions",
    ]
    pd.testing.assert_frame_equal(transactions, original)


def test_fact_store_transactions_has_unique_non_missing_grain() -> None:
    """Date and store surrogate keys form a complete unique grain."""
    transactions, dim_date, dim_store = _transaction_inputs()

    fact = build_fact_store_transactions(transactions, dim_date, dim_store)

    assert not fact.duplicated(["date_key", "store_key"]).any()
    assert not fact[["date_key", "store_key"]].isna().any().any()


def test_fact_store_transactions_rejects_duplicate_grain() -> None:
    """Two transaction rows for one date-store grain must never be multiplied."""
    transactions, dim_date, dim_store = _transaction_inputs()
    duplicate = pd.concat(
        [transactions, transactions.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate date_key, store_key grain"):
        build_fact_store_transactions(duplicate, dim_date, dim_store)


@pytest.mark.parametrize(
    ("dimension_name", "missing_business_key"),
    [("date", "date"), ("store", "store_nbr")],
)
def test_fact_store_transactions_rejects_unmapped_keys(
    dimension_name: str,
    missing_business_key: str,
) -> None:
    """Unmapped transaction dates or stores raise clear errors."""
    transactions, dim_date, dim_store = _transaction_inputs()
    if dimension_name == "date":
        dim_date = dim_date.iloc[[0]]
    else:
        dim_store = dim_store.iloc[[0]]

    with pytest.raises(ValueError, match=missing_business_key):
        build_fact_store_transactions(transactions, dim_date, dim_store)


def test_facts_reject_unmapped_date_store_keys() -> None:
    train, dim_date, dim_store, dim_family, dim_store_date = _inputs()

    with pytest.raises(ValueError, match="unmapped date_key \\+ store_key"):
        build_fact_daily_sales(
            train,
            dim_date,
            dim_store,
            dim_family,
            dim_store_date.iloc[1:],
        )


def _oil_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    oil = pd.DataFrame(
        {
            "date": dates,
            "oil_price": [10.0, 11.5, 11.0],
            "oil_change_1d": [None, 1.5, -0.5],
            "oil_change_7d": [None, None, None],
            "oil_pct_change_7d": [None, None, None],
            "oil_was_imputed": [0, 1, 0],
        }
    )
    dim_date = pd.DataFrame(
        {"date_key": [20200101, 20200102, 20200103], "full_date": dates}
    )
    return oil, dim_date


def test_fact_oil_price_reconciles_range_grain_and_prices() -> None:
    """Oil fact covers the date range once and preserves every mapped price."""
    oil, dim_date = _oil_inputs()
    original = oil.copy(deep=True)

    fact = build_fact_oil_price(oil, dim_date)

    assert len(fact) == len(dim_date)
    assert fact["date_key"].is_unique
    assert not fact["date_key"].isna().any()
    assert not fact["oil_price"].isna().any()
    assert fact["oil_price"].tolist() == oil["oil_price"].tolist()
    assert fact.columns.tolist() == [
        "date_key",
        "oil_price",
        "oil_change_1d",
        "oil_change_7d",
        "oil_pct_change_7d",
        "oil_was_imputed",
    ]
    pd.testing.assert_frame_equal(oil, original)


def test_fact_oil_price_rejects_unmapped_date() -> None:
    """An oil date absent from the date dimension raises a clear error."""
    oil, dim_date = _oil_inputs()

    with pytest.raises(ValueError, match="date"):
        build_fact_oil_price(oil, dim_date.iloc[:-1])


def test_fact_oil_price_rejects_missing_price() -> None:
    """The fact rejects any remaining missing oil price."""
    oil, dim_date = _oil_inputs()
    oil.loc[1, "oil_price"] = None

    with pytest.raises(ValueError, match="oil_price"):
        build_fact_oil_price(oil, dim_date)
