"""Expand holiday events to stores and aggregate them to a daily store grain."""

from collections.abc import Iterable

import pandas as pd


OUTPUT_COLUMNS = [
    "date",
    "store_nbr",
    "holiday_count",
    "holiday_descriptions",
    "holiday_types",
    "holiday_locales",
    "is_holiday",
    "is_work_day",
    "is_event",
]

HOLIDAY_TYPES = {"Holiday", "Transfer", "Additional", "Bridge"}


def _require_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    """Raise a clear error when required columns are absent."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(f"{table_name}: required columns not found: {', '.join(missing)}")


def _join_unique(values: Iterable[object]) -> str:
    """Join distinct non-missing values in alphabetical order."""
    unique_values = sorted({str(value) for value in values if pd.notna(value)})
    return " | ".join(unique_values)


def clean_holidays(
    holidays_df: pd.DataFrame,
    stores_df: pd.DataFrame,
) -> pd.DataFrame:
    """Expand holiday scope to stores and return one summary row per store and date."""
    holiday_columns = [
        "date",
        "type",
        "locale",
        "locale_name",
        "description",
        "transferred",
    ]
    store_columns = ["store_nbr", "city", "state"]
    _require_columns(holidays_df, holiday_columns, "holidays")
    _require_columns(stores_df, store_columns, "stores")

    holidays = holidays_df.loc[:, holiday_columns].copy(deep=True)
    stores = stores_df.loc[:, store_columns].copy(deep=True)
    holidays["date"] = pd.to_datetime(holidays["date"])

    if stores["store_nbr"].duplicated().any():
        raise ValueError("stores: store_nbr must be unique before holiday expansion")

    transferred_holiday = holidays["type"].eq("Holiday") & holidays[
        "transferred"
    ].fillna(False)
    actual_events = holidays.loc[~transferred_holiday].copy()

    national = actual_events.loc[actual_events["locale"].eq("National")].merge(
        stores[["store_nbr"]],
        how="cross",
        validate="many_to_many",
    )
    regional = actual_events.loc[actual_events["locale"].eq("Regional")].merge(
        stores[["store_nbr", "state"]],
        left_on="locale_name",
        right_on="state",
        how="inner",
        validate="many_to_many",
    )
    local = actual_events.loc[actual_events["locale"].eq("Local")].merge(
        stores[["store_nbr", "city"]],
        left_on="locale_name",
        right_on="city",
        how="inner",
        validate="many_to_many",
    )

    expanded = pd.concat([national, regional, local], ignore_index=True, sort=False)
    if expanded.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    result = (
        expanded.groupby(["date", "store_nbr"], dropna=False, observed=True)
        .agg(
            holiday_count=("description", "size"),
            holiday_descriptions=("description", _join_unique),
            holiday_types=("type", _join_unique),
            holiday_locales=("locale", _join_unique),
            is_holiday=("type", lambda values: int(values.isin(HOLIDAY_TYPES).any())),
            is_work_day=("type", lambda values: int(values.eq("Work Day").any())),
            is_event=("type", lambda values: int(values.eq("Event").any())),
        )
        .reset_index()
    )
    for column in ("is_holiday", "is_work_day", "is_event"):
        result[column] = result[column].astype("uint8")

    if result.duplicated(["date", "store_nbr"]).any():
        raise ValueError("holidays: duplicate date, store_nbr grain after aggregation")

    return result.loc[:, OUTPUT_COLUMNS].sort_values(
        ["store_nbr", "date"], kind="stable"
    ).reset_index(drop=True)
