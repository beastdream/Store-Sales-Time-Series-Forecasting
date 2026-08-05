"""Build warehouse bridge tables at their validated business grain."""

import pandas as pd


BRIDGE_STORE_HOLIDAY_COLUMNS = [
    "date_key",
    "store_key",
    "holiday_count",
    "holiday_descriptions",
    "holiday_types",
    "holiday_locales",
    "is_holiday",
    "is_work_day",
    "is_event",
]


def _require_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    """Raise a clear error when required columns are absent."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(f"{table_name}: required columns not found: {', '.join(missing)}")


def build_bridge_store_holiday(
    holiday_store_daily: pd.DataFrame,
    dim_date: pd.DataFrame,
    dim_store: pd.DataFrame,
) -> pd.DataFrame:
    """Map daily store holiday summaries to date and store surrogate keys."""
    holiday_columns = [
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
    _require_columns(holiday_store_daily, holiday_columns, "holiday_store_daily")
    _require_columns(dim_date, ["date_key", "full_date"], "dim_date")
    _require_columns(dim_store, ["store_key", "store_nbr"], "dim_store")
    if dim_date.empty:
        raise ValueError("dim_date must contain the analysis date range")

    source = holiday_store_daily.loc[:, holiday_columns].copy(deep=True)
    analysis_start = dim_date["full_date"].min()
    analysis_end = dim_date["full_date"].max()
    bridge = source.loc[source["date"].between(analysis_start, analysis_end)].copy()

    bridge = bridge.merge(
        dim_date[["date_key", "full_date"]],
        left_on="date",
        right_on="full_date",
        how="left",
        validate="many_to_one",
    )
    if bridge["date_key"].isna().any():
        missing_dates = bridge.loc[bridge["date_key"].isna(), "date"].unique()[:5]
        raise ValueError(
            f"bridge_store_holiday: unmapped date values: {missing_dates.tolist()}"
        )
    bridge = bridge.drop(columns=["date", "full_date"])

    bridge = bridge.merge(
        dim_store[["store_key", "store_nbr"]],
        on="store_nbr",
        how="left",
        validate="many_to_one",
    )
    if bridge["store_key"].isna().any():
        missing_stores = (
            bridge.loc[bridge["store_key"].isna(), "store_nbr"]
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        raise ValueError(
            f"bridge_store_holiday: unmapped store_nbr values: {missing_stores}"
        )
    bridge = bridge.drop(columns="store_nbr").loc[
        :, BRIDGE_STORE_HOLIDAY_COLUMNS
    ]

    grain = ["date_key", "store_key"]
    if bridge[grain].isna().any().any():
        raise RuntimeError("bridge_store_holiday: surrogate keys contain missing values")
    if bridge.duplicated(grain).any():
        raise ValueError("bridge_store_holiday: duplicate date_key, store_key grain")

    return bridge.sort_values(grain, kind="stable").reset_index(drop=True)
