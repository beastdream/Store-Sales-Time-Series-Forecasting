"""Tests for transaction cleaning and daily sales integration."""

from pathlib import Path
import sys

import pandas as pd
import pytest
from pandas.errors import MergeError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.clean_transactions import (
    clean_transactions,
    create_daily_store_sales,
    merge_sales_transactions,
)


def test_clean_transactions_copies_deduplicates_and_sorts() -> None:
    """Cleaning removes exact duplicates without adding missing dates."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-01-03", "2017-01-01", "2017-01-03"]),
            "store_nbr": [2, 1, 2],
            "transactions": [30, 10, 30],
        }
    )
    original = frame.copy(deep=True)

    result = clean_transactions(frame)

    assert len(result) == 2
    assert result["date"].tolist() == pd.to_datetime(
        ["2017-01-01", "2017-01-03"]
    ).tolist()
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("invalid_value", [-1, None])
def test_clean_transactions_rejects_invalid_counts(invalid_value: object) -> None:
    """Negative or missing transaction counts violate the non-negative rule."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-01-01"]),
            "store_nbr": [1],
            "transactions": [invalid_value],
        }
    )

    with pytest.raises(ValueError, match=">= 0"):
        clean_transactions(frame)


def test_clean_transactions_rejects_duplicate_grain() -> None:
    """Conflicting transaction rows at the same grain raise an error."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-01-01", "2017-01-01"]),
            "store_nbr": [1, 1],
            "transactions": [10, 11],
        }
    )

    with pytest.raises(ValueError, match="duplicate grain"):
        clean_transactions(frame)


def test_create_daily_store_sales_aggregates_families() -> None:
    """Family sales aggregate to date and store with the total_sales name."""
    train = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-01-01", "2017-01-01", "2017-01-02"]),
            "store_nbr": [1, 1, 1],
            "family": ["A", "B", "A"],
            "sales": [2.5, 3.5, 4.0],
        }
    )
    original = train.copy(deep=True)

    result = create_daily_store_sales(train)

    assert result.columns.tolist() == ["date", "store_nbr", "total_sales"]
    assert result["total_sales"].tolist() == [6.0, 4.0]
    assert not result.duplicated(["date", "store_nbr"]).any()
    pd.testing.assert_frame_equal(train, original)


def test_merge_sales_transactions_handles_zero_and_missing_denominators() -> None:
    """Zero and absent transaction counts produce NaN instead of division by zero."""
    daily_sales = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-01-01", "2017-01-02", "2017-01-03"]),
            "store_nbr": [1, 1, 1],
            "total_sales": [100.0, 50.0, 25.0],
        }
    )
    transactions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-01-01", "2017-01-02"]),
            "store_nbr": [1, 1],
            "transactions": [20, 0],
        }
    )

    result = merge_sales_transactions(daily_sales, transactions)

    assert result["sales_volume_per_transaction"].iloc[0] == 5.0
    assert pd.isna(result["sales_volume_per_transaction"].iloc[1])
    assert pd.isna(result["sales_volume_per_transaction"].iloc[2])
    assert "revenue_per_transaction" not in result.columns


def test_merge_sales_transactions_enforces_one_to_one() -> None:
    """Duplicate merge keys are rejected by pandas one-to-one validation."""
    daily_sales = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-01-01", "2017-01-01"]),
            "store_nbr": [1, 1],
            "total_sales": [10.0, 20.0],
        }
    )
    transactions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-01-01"]),
            "store_nbr": [1],
            "transactions": [2],
        }
    )

    with pytest.raises(MergeError):
        merge_sales_transactions(daily_sales, transactions)
