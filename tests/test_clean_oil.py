"""Tests for daily oil-price cleaning and feature creation."""

from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.clean_oil import clean_oil


@pytest.fixture
def sparse_oil() -> pd.DataFrame:
    """Return a small oil series with missing calendar days and an explicit null."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-04"]),
            "dcoilwtico": [10.0, None, 14.0],
        }
    )


def test_clean_oil_creates_continuous_calendar(sparse_oil: pd.DataFrame) -> None:
    """The output includes every day in the requested inclusive range."""
    result = clean_oil(
        sparse_oil,
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-05"),
    )

    expected_dates = pd.date_range("2020-01-01", "2020-01-05", freq="D")
    assert result["date"].tolist() == expected_dates.tolist()
    assert len(result) == len(expected_dates)


def test_clean_oil_has_unique_date_grain(sparse_oil: pd.DataFrame) -> None:
    """The final calendar contains exactly one row per date."""
    result = clean_oil(
        sparse_oil,
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-05"),
    )

    assert result["date"].is_unique


def test_clean_oil_preserves_observed_prices(sparse_oil: pd.DataFrame) -> None:
    """Observed source prices are unchanged by interpolation and edge filling."""
    original = sparse_oil.copy(deep=True)
    result = clean_oil(
        sparse_oil,
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-05"),
    ).set_index("date")

    assert result.loc[pd.Timestamp("2020-01-02"), "oil_price"] == 10.0
    assert result.loc[pd.Timestamp("2020-01-04"), "oil_price"] == 14.0
    pd.testing.assert_frame_equal(sparse_oil, original)


def test_clean_oil_marks_missing_source_values(sparse_oil: pd.DataFrame) -> None:
    """Missing source values and absent dates are marked before imputation."""
    result = clean_oil(
        sparse_oil,
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-05"),
    )

    assert result["oil_was_imputed"].tolist() == [1, 0, 1, 0, 1]
    assert result["oil_price"].tolist() == [10.0, 10.0, 12.0, 14.0, 14.0]


def test_clean_oil_has_no_missing_price_and_creates_features(
    sparse_oil: pd.DataFrame,
) -> None:
    """Oil prices are complete and all requested change columns are present."""
    result = clean_oil(
        sparse_oil,
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-10"),
    )

    assert not result["oil_price"].isna().any()
    assert {
        "oil_change_1d",
        "oil_change_7d",
        "oil_pct_change_7d",
    }.issubset(result.columns)


def test_clean_oil_rejects_reversed_range(sparse_oil: pd.DataFrame) -> None:
    """A reversed calendar range raises a clear error."""
    with pytest.raises(ValueError, match="start_date"):
        clean_oil(
            sparse_oil,
            pd.Timestamp("2020-01-05"),
            pd.Timestamp("2020-01-01"),
        )


def test_clean_oil_rejects_duplicate_source_dates() -> None:
    """Duplicate source dates cannot satisfy the one-row-per-day grain."""
    oil = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "dcoilwtico": [10.0, 11.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate date grain"):
        clean_oil(oil, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"))
