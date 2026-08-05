"""Tests for the daily date dimension builder."""

from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_date_dimension import build_date_dimension


def test_leap_year_includes_february_29() -> None:
    """A leap-year range includes February 29 exactly once."""
    result = build_date_dimension(
        pd.Timestamp("2020-02-28"),
        pd.Timestamp("2020-03-01"),
    )

    assert pd.Timestamp("2020-02-29") in result["full_date"].tolist()
    assert len(result) == 3


def test_day_15_is_payday() -> None:
    """The 15th day of a month is flagged as the hypothesized payday."""
    result = build_date_dimension(
        pd.Timestamp("2021-04-15"),
        pd.Timestamp("2021-04-15"),
    )

    assert result.loc[0, "is_payday"] == 1


def test_month_end_is_payday() -> None:
    """The final calendar day of a month is flagged as payday."""
    result = build_date_dimension(
        pd.Timestamp("2021-04-29"),
        pd.Timestamp("2021-05-01"),
    ).set_index("full_date")

    assert result.loc[pd.Timestamp("2021-04-30"), "is_month_end"] == 1
    assert result.loc[pd.Timestamp("2021-04-30"), "is_payday"] == 1
    assert result.loc[pd.Timestamp("2021-04-29"), "is_payday"] == 0


def test_weekend_and_monday_numbering() -> None:
    """Monday is zero and only Saturday and Sunday are weekend days."""
    result = build_date_dimension(
        pd.Timestamp("2021-08-02"),
        pd.Timestamp("2021-08-08"),
    ).set_index("full_date")

    assert result.loc[pd.Timestamp("2021-08-02"), "day_of_week"] == 0
    assert result.loc[pd.Timestamp("2021-08-02"), "is_weekend"] == 0
    assert result.loc[pd.Timestamp("2021-08-07"), "is_weekend"] == 1
    assert result.loc[pd.Timestamp("2021-08-08"), "is_weekend"] == 1


def test_date_key_uses_yyyymmdd_integer_format() -> None:
    """Date keys use integer YYYYMMDD values and remain unique."""
    result = build_date_dimension(
        pd.Timestamp("2023-01-02"),
        pd.Timestamp("2023-01-03"),
    )

    assert result["date_key"].tolist() == [20230102, 20230103]
    assert pd.api.types.is_integer_dtype(result["date_key"])
    assert result["date_key"].is_unique


def test_calendar_is_continuous_complete_and_has_no_missing_values() -> None:
    """The output has one complete row for every requested day."""
    start = pd.Timestamp("2022-12-28")
    end = pd.Timestamp("2023-01-03")
    result = build_date_dimension(start, end)

    assert result["full_date"].tolist() == pd.date_range(start, end, freq="D").tolist()
    assert len(result) == (end - start).days + 1
    assert not result.isna().any().any()


def test_reversed_date_range_raises_error() -> None:
    """The builder rejects a start date after the end date."""
    with pytest.raises(ValueError, match="start_date"):
        build_date_dimension(
            pd.Timestamp("2023-01-02"),
            pd.Timestamp("2023-01-01"),
        )
