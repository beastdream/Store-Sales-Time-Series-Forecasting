"""Validation tests for sales trend and seasonality notebook artifacts."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "sales_trend_seasonality"


def test_sales_trend_csv_outputs_are_valid() -> None:
    expected_columns = {
        "daily_sales_summary.csv": {
            "date",
            "total_sales",
            "sales_ma_7",
            "sales_ma_28",
            "has_sales_observation",
            "year",
            "month",
            "day_of_week",
            "is_weekend",
            "is_payday",
        },
        "monthly_sales_summary.csv": {
            "year_month",
            "total_sales",
            "mom_growth",
            "yoy_growth",
            "missing_sales_dates",
        },
        "weekday_month_summary.csv": {
            "month",
            "month_name",
            "day_of_week",
            "day_name",
            "average_sales",
            "observed_days",
        },
    }
    for filename, columns in expected_columns.items():
        path = TABLE_DIR / filename
        assert path.is_file() and path.stat().st_size > 0
        frame = pd.read_csv(path)
        assert not frame.empty
        assert columns.issubset(frame.columns)

    daily = pd.read_csv(TABLE_DIR / "daily_sales_summary.csv", parse_dates=["date"])
    missing = daily["has_sales_observation"].eq(0)
    assert daily.loc[missing, "total_sales"].isna().all()
    christmas = daily.loc[
        daily["date"].dt.month.eq(12) & daily["date"].dt.day.eq(25)
    ]
    assert not christmas.empty
    assert christmas["has_sales_observation"].eq(0).all()


def test_sales_trend_png_outputs_are_valid() -> None:
    filenames = [
        "daily_sales_with_28d_ma.png",
        "monthly_sales_trend.png",
        "weekday_average_sales.png",
        "month_weekday_heatmap.png",
        "ytd_sales_comparison.png",
    ]
    png_signature = b"\x89PNG\r\n\x1a\n"
    for filename in filenames:
        path = FIGURE_DIR / filename
        assert path.is_file() and path.stat().st_size > len(png_signature)
        assert path.read_bytes()[:8] == png_signature
