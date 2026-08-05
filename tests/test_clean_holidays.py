"""Tests for expanding holiday events to the daily store grain."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.clean_holidays import clean_holidays


def _stores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "store_nbr": [1, 2, 3],
            "city": ["Quito", "Cuenca", "Quito"],
            "state": ["Pichincha", "Azuay", "Pichincha"],
        }
    )


def _holiday(
    *,
    event_type: str = "Holiday",
    locale: str = "National",
    locale_name: str = "Ecuador",
    description: str = "Holiday",
    transferred: bool = False,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "type": [event_type],
            "locale": [locale],
            "locale_name": [locale_name],
            "description": [description],
            "transferred": [transferred],
        }
    )


def test_national_event_applies_to_every_store() -> None:
    """National events expand to all stores."""
    result = clean_holidays(_holiday(), _stores())

    assert result["store_nbr"].tolist() == [1, 2, 3]
    assert result["is_holiday"].eq(1).all()


def test_regional_event_matches_state() -> None:
    """Regional events expand only to stores in the matching state."""
    result = clean_holidays(
        _holiday(locale="Regional", locale_name="Pichincha"),
        _stores(),
    )

    assert result["store_nbr"].tolist() == [1, 3]


def test_local_event_matches_city() -> None:
    """Local events expand only to stores in the matching city."""
    result = clean_holidays(
        _holiday(locale="Local", locale_name="Cuenca"),
        _stores(),
    )

    assert result["store_nbr"].tolist() == [2]


def test_transferred_holiday_is_not_an_actual_holiday() -> None:
    """A transferred Holiday row does not create a store holiday record."""
    result = clean_holidays(_holiday(transferred=True), _stores())

    assert result.empty


def test_work_day_is_retained_and_flagged() -> None:
    """Work Day rows remain available with their dedicated flag."""
    result = clean_holidays(_holiday(event_type="Work Day"), _stores())

    assert result["is_work_day"].eq(1).all()
    assert result["is_holiday"].eq(0).all()
    assert result["holiday_types"].eq("Work Day").all()


def test_multiple_same_day_events_are_aggregated_deterministically() -> None:
    """Multiple events share one grain row with distinct sorted labels."""
    events = pd.concat(
        [
            _holiday(description="Zoo", event_type="Holiday"),
            _holiday(
                description="Anniversary",
                event_type="Event",
                locale="Local",
                locale_name="Quito",
            ),
            _holiday(
                description="Anniversary",
                event_type="Additional",
                locale="Local",
                locale_name="Quito",
            ),
        ],
        ignore_index=True,
    )

    result = clean_holidays(events, _stores())
    store_one = result.loc[result["store_nbr"].eq(1)].iloc[0]

    assert store_one["holiday_count"] == 3
    assert store_one["holiday_descriptions"] == "Anniversary | Zoo"
    assert store_one["holiday_types"] == "Additional | Event | Holiday"
    assert store_one["holiday_locales"] == "Local | National"
    assert store_one["is_holiday"] == 1
    assert store_one["is_event"] == 1
    assert not result.duplicated(["date", "store_nbr"]).any()
