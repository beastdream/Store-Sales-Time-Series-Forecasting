"""Build stable store and product-family dimensions."""

import pandas as pd


def _require_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    """Raise a clear error when required columns are absent."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(f"{table_name}: required columns not found: {', '.join(missing)}")


def build_dim_store(stores_clean: pd.DataFrame) -> pd.DataFrame:
    """Build a stable store dimension keyed after sorting by ``store_nbr``."""
    columns = ["store_nbr", "city", "state", "store_type", "cluster"]
    _require_columns(stores_clean, columns, "stores_clean")
    dimension = stores_clean.loc[:, columns].copy(deep=True)

    if dimension["store_nbr"].isna().any():
        raise ValueError("dim_store: store_nbr must not contain missing values")
    if dimension["store_nbr"].duplicated().any():
        raise ValueError("dim_store: store_nbr must be unique")

    dimension = dimension.sort_values("store_nbr", kind="stable").reset_index(drop=True)
    dimension.insert(
        0,
        "store_key",
        pd.Series(range(1, len(dimension) + 1), dtype="int32"),
    )
    return dimension


def build_dim_family(train_clean: pd.DataFrame) -> pd.DataFrame:
    """Build a stable family dimension keyed after sorting by ``family``."""
    _require_columns(train_clean, ["family"], "train_clean")
    if train_clean["family"].isna().any():
        raise ValueError("dim_family: family must not contain missing values")

    dimension = train_clean.loc[:, ["family"]].drop_duplicates().copy()
    dimension["_business_key_sort"] = dimension["family"].astype(str)
    dimension = dimension.sort_values(
        "_business_key_sort", kind="stable"
    ).drop(columns="_business_key_sort").reset_index(drop=True)
    dimension.insert(
        0,
        "family_key",
        pd.Series(range(1, len(dimension) + 1), dtype="int32"),
    )
    return dimension
