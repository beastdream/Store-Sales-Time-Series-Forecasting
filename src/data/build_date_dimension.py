"""Build the reusable daily date dimension."""

import pandas as pd


def build_date_dimension(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Build an inclusive, continuous date dimension with one row per day.

    ``is_payday`` is an analytical hypothesis based on the dataset context: it
    flags the 15th and final calendar day of each month. It is not a confirmed
    business rule.
    """
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if pd.isna(start) or pd.isna(end):
        raise ValueError("start_date and end_date must be valid dates")
    if start > end:
        raise ValueError("start_date must be less than or equal to end_date")

    full_date = pd.Series(pd.date_range(start, end, freq="D"), name="full_date")
    date_dimension = pd.DataFrame({"full_date": full_date})
    date_dimension.insert(
        0,
        "date_key",
        date_dimension["full_date"].dt.strftime("%Y%m%d").astype("int32"),
    )
    date_dimension["day"] = date_dimension["full_date"].dt.day.astype("uint8")
    date_dimension["day_of_week"] = (
        date_dimension["full_date"].dt.dayofweek.astype("uint8")
    )
    date_dimension["day_name"] = date_dimension["full_date"].dt.day_name()
    date_dimension["week_of_year"] = (
        date_dimension["full_date"].dt.isocalendar().week.astype("uint8")
    )
    date_dimension["month"] = date_dimension["full_date"].dt.month.astype("uint8")
    date_dimension["month_name"] = date_dimension["full_date"].dt.month_name()
    date_dimension["quarter"] = (
        date_dimension["full_date"].dt.quarter.astype("uint8")
    )
    date_dimension["year"] = date_dimension["full_date"].dt.year.astype("int16")
    date_dimension["is_weekend"] = (
        date_dimension["day_of_week"].isin([5, 6]).astype("uint8")
    )
    date_dimension["is_month_start"] = (
        date_dimension["full_date"].dt.is_month_start.astype("uint8")
    )
    date_dimension["is_month_end"] = (
        date_dimension["full_date"].dt.is_month_end.astype("uint8")
    )
    date_dimension["is_payday"] = (
        date_dimension["day"].eq(15)
        | date_dimension["is_month_end"].eq(1)
    ).astype("uint8")

    expected_rows = (end - start).days + 1
    if len(date_dimension) != expected_rows:
        raise RuntimeError("date dimension row count does not match the requested range")
    if date_dimension["date_key"].duplicated().any():
        raise RuntimeError("date dimension contains duplicate date_key values")
    if date_dimension.isna().any().any():
        raise RuntimeError("date dimension contains missing values")

    return date_dimension
