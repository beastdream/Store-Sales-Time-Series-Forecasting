"""Cleaning and sales-volume integration for daily store transactions."""

import pandas as pd

from src.data.audit import check_grain


DAILY_STORE_GRAIN = ["date", "store_nbr"]


def _require_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    """Raise a clear error when required columns are absent."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(f"{table_name}: required columns not found: {', '.join(missing)}")


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Copy, deduplicate, validate, and sort daily store transactions."""
    required = ["date", "store_nbr", "transactions"]
    _require_columns(df, required, "transactions")
    cleaned = df.copy(deep=True).drop_duplicates().copy()

    if not cleaned["transactions"].ge(0).all():
        raise ValueError("transactions: values must be non-missing and >= 0")

    duplicate_grains = check_grain(
        cleaned,
        DAILY_STORE_GRAIN,
        "transactions",
        sample_size=1,
    )
    if not duplicate_grains.empty:
        raise ValueError("transactions: duplicate grain detected for date, store_nbr")

    return cleaned.sort_values(
        ["store_nbr", "date"], kind="stable"
    ).reset_index(drop=True)


def create_daily_store_sales(train_clean: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned family-level sales to one row per date and store."""
    _require_columns(train_clean, [*DAILY_STORE_GRAIN, "sales"], "train_clean")

    daily_sales = (
        train_clean.groupby(DAILY_STORE_GRAIN, dropna=False, observed=True)["sales"]
        .sum(min_count=1)
        .rename("total_sales")
        .reset_index()
    )
    return daily_sales.sort_values(
        ["store_nbr", "date"], kind="stable"
    ).reset_index(drop=True)


def merge_sales_transactions(
    daily_store_sales: pd.DataFrame,
    transactions_clean: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join daily sales to transactions and derive sales volume per transaction."""
    _require_columns(
        daily_store_sales,
        [*DAILY_STORE_GRAIN, "total_sales"],
        "daily_store_sales",
    )
    _require_columns(
        transactions_clean,
        [*DAILY_STORE_GRAIN, "transactions"],
        "transactions_clean",
    )

    merged = daily_store_sales.merge(
        transactions_clean,
        on=DAILY_STORE_GRAIN,
        how="left",
        validate="one_to_one",
    )
    transaction_denominator = (
        merged["transactions"].astype("float64").mask(merged["transactions"].eq(0))
    )
    merged["sales_volume_per_transaction"] = (
        merged["total_sales"] / transaction_denominator
    )
    return merged
