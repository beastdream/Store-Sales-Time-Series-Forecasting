"""Build validated fact tables from cleaned business-key data."""

import numpy as np
import pandas as pd


FACT_DAILY_SALES_COLUMNS = [
    "sales_id",
    "date_key",
    "store_key",
    "date_store_key",
    "family_key",
    "sales",
    "onpromotion",
    "is_promotion",
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


def _raise_for_unmapped_key(
    mapped: pd.DataFrame,
    surrogate_key: str,
    business_key: str,
    fact_name: str,
) -> None:
    """Raise when a dimension merge leaves any business key unmapped."""
    missing_mask = mapped[surrogate_key].isna()
    if missing_mask.any():
        examples = mapped.loc[missing_mask, business_key].drop_duplicates().head(5)
        raise ValueError(
            f"{fact_name}: unmapped {business_key} values: "
            f"{examples.astype(str).tolist()}"
        )


def _merge_date_store_key(
    fact: pd.DataFrame,
    dim_store_date: pd.DataFrame,
    fact_name: str,
) -> pd.DataFrame:
    """Attach the conformed date-store key without removing audit keys."""
    _require_columns(
        dim_store_date,
        ["date_store_key", "date_key", "store_key"],
        "dim_store_date",
    )
    mapped = fact.merge(
        dim_store_date[["date_store_key", "date_key", "store_key"]],
        on=["date_key", "store_key"],
        how="left",
        validate="many_to_one",
    )
    if mapped["date_store_key"].isna().any():
        examples = (
            mapped.loc[mapped["date_store_key"].isna(), ["date_key", "store_key"]]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            f"{fact_name}: unmapped date_key + store_key combinations: {examples}"
        )
    return mapped


def build_fact_daily_sales(
    train_clean: pd.DataFrame,
    dim_date: pd.DataFrame,
    dim_store: pd.DataFrame,
    dim_family: pd.DataFrame,
    dim_store_date: pd.DataFrame,
) -> pd.DataFrame:
    """Map cleaned daily family sales to dimension keys and reconcile measures."""
    train_columns = [
        "id",
        "date",
        "store_nbr",
        "family",
        "sales",
        "onpromotion",
        "is_promotion",
    ]
    _require_columns(train_clean, train_columns, "train_clean")
    _require_columns(dim_date, ["date_key", "full_date"], "dim_date")
    _require_columns(dim_store, ["store_key", "store_nbr"], "dim_store")
    _require_columns(dim_family, ["family_key", "family"], "dim_family")

    source_row_count = len(train_clean)
    source_sales = train_clean["sales"].sum()
    source_onpromotion = train_clean["onpromotion"].sum()
    fact = train_clean.loc[:, train_columns].copy(deep=True)

    fact = fact.merge(
        dim_date[["date_key", "full_date"]],
        left_on="date",
        right_on="full_date",
        how="left",
        validate="many_to_one",
    )
    _raise_for_unmapped_key(fact, "date_key", "date", "fact_daily_sales")
    fact = fact.drop(columns=["date", "full_date"])

    fact = fact.merge(
        dim_store[["store_key", "store_nbr"]],
        on="store_nbr",
        how="left",
        validate="many_to_one",
    )
    _raise_for_unmapped_key(fact, "store_key", "store_nbr", "fact_daily_sales")
    fact = fact.drop(columns="store_nbr")

    fact = fact.merge(
        dim_family[["family_key", "family"]],
        on="family",
        how="left",
        validate="many_to_one",
    )
    _raise_for_unmapped_key(fact, "family_key", "family", "fact_daily_sales")
    fact = fact.drop(columns="family").rename(columns={"id": "sales_id"})
    fact = _merge_date_store_key(fact, dim_store_date, "fact_daily_sales")
    fact = fact.loc[:, FACT_DAILY_SALES_COLUMNS]

    if len(fact) != source_row_count:
        raise RuntimeError("fact_daily_sales: row count reconciliation failed")
    if not np.isclose(
        fact["sales"].sum(),
        source_sales,
        rtol=0,
        atol=1e-6,
    ):
        raise RuntimeError("fact_daily_sales: sales total reconciliation failed")
    if fact["onpromotion"].sum() != source_onpromotion:
        raise RuntimeError("fact_daily_sales: onpromotion total reconciliation failed")

    surrogate_keys = ["date_key", "store_key", "family_key"]
    if fact[surrogate_keys].isna().any().any():
        raise RuntimeError("fact_daily_sales: surrogate keys contain missing values")
    if fact.duplicated(surrogate_keys).any():
        raise ValueError(
            "fact_daily_sales: duplicate date_key, store_key, family_key grain"
        )
    if not pd.api.types.is_float_dtype(fact["sales"]):
        raise TypeError("fact_daily_sales: sales must retain a floating-point dtype")

    return fact.sort_values(surrogate_keys, kind="stable").reset_index(drop=True)


def build_fact_store_transactions(
    transactions_clean: pd.DataFrame,
    dim_date: pd.DataFrame,
    dim_store: pd.DataFrame,
) -> pd.DataFrame:
    """Build one transaction row per date and store using conformed keys."""
    transaction_columns = ["date", "store_nbr", "transactions"]
    _require_columns(transactions_clean, transaction_columns, "transactions_clean")
    _require_columns(dim_date, ["date_key", "full_date"], "dim_date")
    _require_columns(dim_store, ["store_key", "store_nbr"], "dim_store")

    source_row_count = len(transactions_clean)
    source_total = transactions_clean["transactions"].sum()
    fact = transactions_clean.loc[:, transaction_columns].copy(deep=True)

    fact = fact.merge(
        dim_date[["date_key", "full_date"]],
        left_on="date",
        right_on="full_date",
        how="left",
        validate="many_to_one",
    )
    _raise_for_unmapped_key(
        fact,
        "date_key",
        "date",
        "fact_store_transactions",
    )
    fact = fact.drop(columns=["date", "full_date"])

    fact = fact.merge(
        dim_store[["store_key", "store_nbr"]],
        on="store_nbr",
        how="left",
        validate="many_to_one",
    )
    _raise_for_unmapped_key(
        fact,
        "store_key",
        "store_nbr",
        "fact_store_transactions",
    )
    fact = fact.drop(columns="store_nbr")
    fact = fact.loc[:, ["date_key", "store_key", "transactions"]]

    if len(fact) != source_row_count:
        raise RuntimeError("fact_store_transactions: row count reconciliation failed")
    if fact["transactions"].sum() != source_total:
        raise RuntimeError("fact_store_transactions: total reconciliation failed")

    grain = ["date_key", "store_key"]
    if fact[grain].isna().any().any():
        raise RuntimeError(
            "fact_store_transactions: date_key or store_key contains missing values"
        )
    if fact.duplicated(grain).any():
        raise ValueError(
            "fact_store_transactions: duplicate date_key, store_key grain"
        )

    return fact.sort_values(grain, kind="stable").reset_index(drop=True)


def build_fact_oil_price(
    oil_clean: pd.DataFrame,
    dim_date: pd.DataFrame,
) -> pd.DataFrame:
    """Map the complete daily oil series to the date dimension without changing prices."""
    oil_columns = [
        "date",
        "oil_price",
        "oil_change_1d",
        "oil_change_7d",
        "oil_pct_change_7d",
        "oil_was_imputed",
    ]
    _require_columns(oil_clean, oil_columns, "oil_clean")
    _require_columns(dim_date, ["date_key", "full_date"], "dim_date")

    source_prices = oil_clean["oil_price"].reset_index(drop=True).copy()
    fact = oil_clean.loc[:, oil_columns].copy(deep=True).merge(
        dim_date[["date_key", "full_date"]],
        left_on="date",
        right_on="full_date",
        how="left",
        validate="one_to_one",
    )
    _raise_for_unmapped_key(fact, "date_key", "date", "fact_oil_price")

    if len(fact) != len(dim_date):
        raise RuntimeError(
            "fact_oil_price: row count does not match the analysis date range"
        )
    if fact["oil_price"].isna().any():
        raise ValueError("fact_oil_price: oil_price contains missing values")
    if not np.array_equal(
        fact["oil_price"].to_numpy(),
        source_prices.to_numpy(),
        equal_nan=True,
    ):
        raise RuntimeError("fact_oil_price: oil prices changed during date mapping")

    fact = fact.drop(columns=["date", "full_date"]).loc[
        :,
        [
            "date_key",
            "oil_price",
            "oil_change_1d",
            "oil_change_7d",
            "oil_pct_change_7d",
            "oil_was_imputed",
        ],
    ]
    if fact["date_key"].isna().any():
        raise RuntimeError("fact_oil_price: date_key contains missing values")
    if fact["date_key"].duplicated().any():
        raise ValueError("fact_oil_price: duplicate date_key grain")

    return fact.sort_values("date_key", kind="stable").reset_index(drop=True)
