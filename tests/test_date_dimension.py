"""Additional contract tests for the processed date dimension."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_date_dimension import build_date_dimension


EXPECTED_COLUMNS = [
    "date_key",
    "full_date",
    "day",
    "day_of_week",
    "day_name",
    "week_of_year",
    "month",
    "month_name",
    "quarter",
    "year",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_payday",
]


def test_date_dimension_contract() -> None:
    """The dimension exposes its stable schema and daily grain."""
    result = build_date_dimension(
        pd.Timestamp("2019-12-30"),
        pd.Timestamp("2020-01-02"),
    )

    assert result.columns.tolist() == EXPECTED_COLUMNS
    assert result["date_key"].is_unique
    assert not result.isna().any().any()


def test_date_dimension_inclusive_bounds_and_continuity() -> None:
    """Both bounds and every intervening day are present exactly once."""
    start = pd.Timestamp("2020-02-27")
    end = pd.Timestamp("2020-03-02")
    result = build_date_dimension(start, end)

    assert result["full_date"].tolist() == pd.date_range(start, end, freq="D").tolist()
    assert len(result) == 5
    assert result["full_date"].diff().dropna().eq(pd.Timedelta(days=1)).all()


def test_date_dimension_calendar_flags() -> None:
    """Leap day, weekend, month-end, and payday flags remain consistent."""
    result = build_date_dimension(
        pd.Timestamp("2020-02-15"),
        pd.Timestamp("2020-02-29"),
    ).set_index("full_date")

    assert result.loc[pd.Timestamp("2020-02-15"), "is_payday"] == 1
    assert result.loc[pd.Timestamp("2020-02-29"), "is_weekend"] == 1
    assert result.loc[pd.Timestamp("2020-02-29"), "is_month_end"] == 1
    assert result.loc[pd.Timestamp("2020-02-29"), "is_payday"] == 1
