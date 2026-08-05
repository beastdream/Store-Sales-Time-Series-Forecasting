"""Core cleaning rules for train, test, and store metadata tables."""

import re

import pandas as pd

from src.data.audit import check_grain


SALES_GRAIN = ["date", "store_nbr", "family"]


def _require_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    """Raise a clear error when required columns are absent."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(f"{table_name}: required columns not found: {', '.join(missing)}")


def _validate_sales_grain(df: pd.DataFrame, table_name: str) -> None:
    """Ensure a sales table has one row per date, store, and family."""
    duplicate_grains = check_grain(df, SALES_GRAIN, table_name, sample_size=1)
    if not duplicate_grains.empty:
        raise ValueError(
            f"{table_name}: duplicate grain detected for "
            f"{', '.join(SALES_GRAIN)}"
        )


def clean_train(df: pd.DataFrame) -> pd.DataFrame:
    """Clean training rows while preserving valid zero and floating-point sales."""
    required = ["date", "store_nbr", "family", "sales", "onpromotion"]
    _require_columns(df, required, "train")
    cleaned = df.copy(deep=True).drop_duplicates().copy()

    if cleaned["sales"].lt(0).any():
        raise ValueError("train: sales contains negative values")
    if cleaned["onpromotion"].lt(0).any():
        raise ValueError("train: onpromotion contains negative values")

    cleaned["is_promotion"] = cleaned["onpromotion"].gt(0).astype("uint8")
    _validate_sales_grain(cleaned, "train")
    return cleaned.sort_values(SALES_GRAIN, kind="stable").reset_index(drop=True)


def clean_test(df: pd.DataFrame) -> pd.DataFrame:
    """Clean test rows, validate promotions and enforce the expected grain."""
    required = ["date", "store_nbr", "family", "onpromotion"]
    _require_columns(df, required, "test")
    cleaned = df.copy(deep=True).drop_duplicates().copy()

    if cleaned["onpromotion"].lt(0).any():
        raise ValueError("test: onpromotion contains negative values")

    cleaned["is_promotion"] = cleaned["onpromotion"].gt(0).astype("uint8")
    _validate_sales_grain(cleaned, "test")
    return cleaned.sort_values(SALES_GRAIN, kind="stable").reset_index(drop=True)


def _to_snake_case(column_name: object) -> str:
    """Convert one column label to lowercase snake_case."""
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", str(column_name).strip())
    return normalized.strip("_").lower()


def clean_stores(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize store columns and trim metadata without recoding its content."""
    cleaned = df.copy(deep=True)
    cleaned.columns = [_to_snake_case(column) for column in cleaned.columns]
    cleaned = cleaned.rename(columns={"type": "store_type"})

    if cleaned.columns.duplicated().any():
        raise ValueError("stores: column normalization produced duplicate names")

    required = ["store_nbr", "city", "state", "store_type"]
    _require_columns(cleaned, required, "stores")
    for column in ("city", "state", "store_type"):
        cleaned[column] = cleaned[column].str.strip()

    if cleaned["store_nbr"].duplicated().any():
        raise ValueError("stores: store_nbr must be unique")

    return cleaned.reset_index(drop=True)
