"""Tests for store-level recent and year-over-year growth windows."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "notebooks" / "04_business_eda.py"
SPEC = importlib.util.spec_from_file_location("business_eda", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
BUSINESS_EDA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUSINESS_EDA)


def test_store_growth_uses_known_calendar_windows_without_zero_imputation() -> None:
    daily_sales = pd.DataFrame(
        {
            "store_key": [1, 1, 1, 1, 2, 2],
            "full_date": pd.to_datetime(
                [
                    "2023-05-01",  # Store 1 YoY comparison window.
                    "2024-03-01",  # Immediately previous 90-day window.
                    "2024-05-01",  # Recent 90-day window.
                    "2024-06-30",  # Sets the actual dataset end date.
                    "2024-03-01",
                    "2024-05-01",
                ]
            ),
            "daily_sales": [75.0, 100.0, 50.0, 100.0, 0.0, 20.0],
        }
    )

    result = BUSINESS_EDA._calculate_growth(daily_sales).set_index("store_key")

    assert result.loc[1, "recent_90d_sales"] == 150.0
    assert result.loc[1, "previous_90d_sales"] == 100.0
    assert result.loc[1, "recent_90d_growth"] == 0.5
    assert result.loc[1, "recent_90d_yoy_growth"] == 1.0
    assert result.loc[1, "has_yoy_comparison"] == 1
    assert result.loc[1, "recent_90d_observed_days"] == 2
    assert result.loc[1, "recent_90d_start_date"] == pd.Timestamp("2024-04-02")
    assert result.loc[1, "recent_90d_end_date"] == pd.Timestamp("2024-06-30")

    assert np.isnan(result.loc[2, "recent_90d_growth"])
    assert np.isnan(result.loc[2, "recent_90d_yoy_growth"])
    assert result.loc[2, "has_yoy_comparison"] == 0


def test_store_performance_artifact_exposes_growth_contract() -> None:
    performance = pd.read_csv(PROJECT_ROOT / "reports" / "tables" / "store_performance.csv")
    required = {
        "first_vs_last_90d_growth_proxy",
        "recent_90d_growth",
        "recent_90d_yoy_growth",
        "has_yoy_comparison",
        "rank_recent_90d_growth",
    }

    assert required.issubset(performance.columns)
    assert "growth_rate" not in performance.columns
    assert set(performance["has_yoy_comparison"].unique()).issubset({0, 1})
    assert performance["store_nbr"].is_unique
