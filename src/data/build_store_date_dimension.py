"""Build the complete date-by-store dimension for observation-aware analytics."""

import pandas as pd

from src.data.build_bridges import build_bridge_store_holiday


DIM_STORE_DATE_COLUMNS = [
    "date_store_key",
    "date_key",
    "store_key",
    "holiday_count",
    "holiday_descriptions",
    "holiday_types",
    "holiday_locales",
    "is_holiday",
    "is_work_day",
    "is_event",
    "has_sales_observation",
    "has_transaction_observation",
]

HOLIDAY_TEXT_COLUMNS = [
    "holiday_descriptions",
    "holiday_types",
    "holiday_locales",
]

HOLIDAY_FLAG_COLUMNS = ["is_holiday", "is_work_day", "is_event"]


def _require_columns(
    frame: pd.DataFrame,
    required: list[str],
    table_name: str,
) -> None:
    """Raise a clear error when a required source column is absent."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"{table_name}: required columns not found: {', '.join(missing)}")


def _map_observation_grain(
    observations: pd.DataFrame,
    dim_date: pd.DataFrame,
    dim_store: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    """Map distinct source date-store observations to surrogate-key grain."""
    _require_columns(observations, ["date", "store_nbr"], source_name)
    distinct = observations[["date", "store_nbr"]].drop_duplicates().copy()
    mapped = distinct.merge(
        dim_date[["date_key", "full_date"]],
        left_on="date",
        right_on="full_date",
        how="left",
        validate="many_to_one",
    )
    if mapped["date_key"].isna().any():
        examples = mapped.loc[mapped["date_key"].isna(), "date"].head(5).tolist()
        raise ValueError(f"{source_name}: unmapped date values: {examples}")
    mapped = mapped.merge(
        dim_store[["store_key", "store_nbr"]],
        on="store_nbr",
        how="left",
        validate="many_to_one",
    )
    if mapped["store_key"].isna().any():
        examples = (
            mapped.loc[mapped["store_key"].isna(), "store_nbr"]
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        raise ValueError(f"{source_name}: unmapped store_nbr values: {examples}")
    return mapped[["date_key", "store_key"]]


def build_dim_store_date(
    dim_date: pd.DataFrame,
    dim_store: pd.DataFrame,
    holiday_store_daily: pd.DataFrame,
    train_clean: pd.DataFrame,
    transactions_clean: pd.DataFrame,
) -> pd.DataFrame:
    """Return every date-store combination with holiday and observation flags.

    ``dim_date`` defines the complete analysis range. The warehouse pipeline builds
    that dimension from the minimum train date through the maximum test date.
    Missing source observations are represented only by zero-valued presence flags;
    no sales or transaction measure is imputed.
    """
    _require_columns(dim_date, ["date_key", "full_date"], "dim_date")
    _require_columns(dim_store, ["store_key", "store_nbr"], "dim_store")
    if dim_date.empty:
        raise ValueError("dim_date must contain the complete analysis date range")
    if dim_store.empty:
        raise ValueError("dim_store must contain at least one store")
    if dim_date[["date_key", "full_date"]].isna().any().any():
        raise ValueError("dim_date contains missing date keys or dates")
    if dim_store[["store_key", "store_nbr"]].isna().any().any():
        raise ValueError("dim_store contains missing store keys or business keys")
    if dim_date["date_key"].duplicated().any() or dim_date["full_date"].duplicated().any():
        raise ValueError("dim_date must have unique date_key and full_date values")
    if dim_store["store_key"].duplicated().any() or dim_store["store_nbr"].duplicated().any():
        raise ValueError("dim_store must have unique store_key and store_nbr values")
    if dim_store["store_key"].ge(100).any():
        raise ValueError(
            "dim_store contains store_key >= 100; use a different date_store_key strategy"
        )

    grid = dim_date[["date_key"]].merge(
        dim_store[["store_key"]],
        how="cross",
        validate="many_to_many",
    )
    expected_rows = len(dim_date) * len(dim_store)
    if len(grid) != expected_rows:
        raise RuntimeError("dim_store_date: date-store Cartesian row count failed")

    holiday = build_bridge_store_holiday(
        holiday_store_daily,
        dim_date,
        dim_store,
    )
    result = grid.merge(
        holiday,
        on=["date_key", "store_key"],
        how="left",
        validate="one_to_one",
        indicator="_holiday_mapping",
    )
    if int(result["_holiday_mapping"].eq("both").sum()) != len(holiday):
        raise RuntimeError("dim_store_date: not every valid holiday mapping was retained")
    result = result.drop(columns="_holiday_mapping")
    result["holiday_count"] = result["holiday_count"].fillna(0).astype("int64")
    for column in HOLIDAY_FLAG_COLUMNS:
        result[column] = result[column].fillna(0).astype("uint8")
    for column in HOLIDAY_TEXT_COLUMNS:
        result[column] = result[column].fillna("").astype("string")

    sales_observations = _map_observation_grain(
        train_clean,
        dim_date,
        dim_store,
        "train_clean",
    ).assign(has_sales_observation=1)
    transaction_observations = _map_observation_grain(
        transactions_clean,
        dim_date,
        dim_store,
        "transactions_clean",
    ).assign(has_transaction_observation=1)
    result = result.merge(
        sales_observations,
        on=["date_key", "store_key"],
        how="left",
        validate="one_to_one",
    ).merge(
        transaction_observations,
        on=["date_key", "store_key"],
        how="left",
        validate="one_to_one",
    )
    for column in ("has_sales_observation", "has_transaction_observation"):
        result[column] = result[column].fillna(0).astype("uint8")

    result["date_store_key"] = (
        result["date_key"].astype("int64") * 100
        + result["store_key"].astype("int64")
    )
    result = result.loc[:, DIM_STORE_DATE_COLUMNS]

    grain = ["date_key", "store_key"]
    if len(result) != expected_rows:
        raise RuntimeError("dim_store_date: final row count reconciliation failed")
    if result[grain].isna().any().any():
        raise RuntimeError("dim_store_date: date_key or store_key contains missing values")
    if result.duplicated(grain).any():
        raise ValueError("dim_store_date: duplicate date_key, store_key grain")
    if result["date_store_key"].duplicated().any():
        raise ValueError("dim_store_date: date_store_key is not unique")

    return result.sort_values(grain, kind="stable").reset_index(drop=True)
