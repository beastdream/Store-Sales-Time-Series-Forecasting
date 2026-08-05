"""Tests for warehouse bridge table builders."""

from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_bridges import build_bridge_store_holiday


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holidays = pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-12-31", "2020-01-01", "2020-01-02"]),
            "store_nbr": [1, 1, 2],
            "holiday_count": [1, 2, 1],
            "holiday_descriptions": ["Old", "A | B", "C"],
            "holiday_types": ["Holiday", "Event | Holiday", "Work Day"],
            "holiday_locales": ["National", "Local | National", "Regional"],
            "is_holiday": [1, 1, 0],
            "is_work_day": [0, 0, 1],
            "is_event": [0, 1, 0],
        }
    )
    dim_date = pd.DataFrame(
        {
            "date_key": [20200101, 20200102],
            "full_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        }
    )
    dim_store = pd.DataFrame({"store_key": [10, 20], "store_nbr": [1, 2]})
    return holidays, dim_date, dim_store


def test_bridge_maps_keys_preserves_text_and_analysis_range() -> None:
    """Bridge maps in-range holidays while preserving descriptive fields."""
    holidays, dim_date, dim_store = _inputs()
    original = holidays.copy(deep=True)

    bridge = build_bridge_store_holiday(holidays, dim_date, dim_store)

    assert len(bridge) == 2
    assert bridge["date_key"].tolist() == [20200101, 20200102]
    assert bridge["store_key"].tolist() == [10, 20]
    assert bridge.loc[0, "holiday_descriptions"] == "A | B"
    assert bridge.loc[0, "holiday_types"] == "Event | Holiday"
    assert "family_key" not in bridge.columns
    pd.testing.assert_frame_equal(holidays, original)


def test_bridge_has_complete_unique_grain() -> None:
    """Date and store keys are complete and unique in the bridge."""
    holidays, dim_date, dim_store = _inputs()

    bridge = build_bridge_store_holiday(holidays, dim_date, dim_store)

    grain = ["date_key", "store_key"]
    assert not bridge[grain].isna().any().any()
    assert not bridge.duplicated(grain).any()


def test_bridge_rejects_unmapped_store() -> None:
    """An in-range holiday for an unknown store raises a clear error."""
    holidays, dim_date, dim_store = _inputs()
    holidays.loc[1, "store_nbr"] = 99

    with pytest.raises(ValueError, match="store_nbr"):
        build_bridge_store_holiday(holidays, dim_date, dim_store)


def test_bridge_rejects_date_gap_inside_analysis_range() -> None:
    """A holiday on a missing dimension date inside the range is rejected."""
    holidays, dim_date, dim_store = _inputs()
    dim_date = pd.concat(
        [
            dim_date.iloc[[0]],
            pd.DataFrame(
                {
                    "date_key": [20200103],
                    "full_date": pd.to_datetime(["2020-01-03"]),
                }
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="date"):
        build_bridge_store_holiday(holidays, dim_date, dim_store)
