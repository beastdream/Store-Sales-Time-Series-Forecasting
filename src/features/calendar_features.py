"""Calendar features known for the full forecast horizon."""

import pandas as pd

from src.data.build_date_dimension import build_date_dimension


CALENDAR_FEATURE_COLUMNS = [
    "day_of_week",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_payday",
]


def build_calendar_features(dates: pd.Series) -> pd.DataFrame:
    """Return one calendar-feature row per distinct input date.

    The implementation delegates to the warehouse date-dimension builder so
    payday and calendar definitions remain consistent across the project.
    """
    parsed_dates = pd.to_datetime(dates, errors="coerce")
    if parsed_dates.isna().any() or parsed_dates.empty:
        raise ValueError("dates must be non-empty and contain only valid dates")

    date_dimension = build_date_dimension(parsed_dates.min(), parsed_dates.max())
    requested = pd.DataFrame({"date": parsed_dates.dt.normalize().drop_duplicates()})
    calendar = date_dimension.rename(columns={"full_date": "date"})
    return requested.merge(
        calendar[["date", *CALENDAR_FEATURE_COLUMNS]],
        on="date",
        how="left",
        validate="one_to_one",
    ).sort_values("date", kind="stable").reset_index(drop=True)
