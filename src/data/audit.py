"""Column-level data quality summaries for pandas DataFrames."""

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


AUDIT_COLUMNS = [
    "table_name",
    "column_name",
    "dtype",
    "row_count",
    "missing_count",
    "missing_pct",
    "unique_count",
    "zero_count",
    "negative_count",
    "min_value",
    "max_value",
]

GRAIN_DEFINITIONS: dict[str, list[str]] = {
    "train": ["date", "store_nbr", "family"],
    "test": ["date", "store_nbr", "family"],
    "stores": ["store_nbr"],
    "transactions": ["date", "store_nbr"],
    "oil": ["date"],
    "sample_submission": ["id"],
}

GRAIN_SUMMARY_COLUMNS = [
    "table_name",
    "grain_columns",
    "duplicate_combinations",
    "affected_rows",
    "is_valid",
]

FOREIGN_KEY_COLUMNS = [
    "child_table",
    "child_column",
    "parent_table",
    "parent_column",
    "invalid_value",
    "affected_rows",
]


def audit_dataframe(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Return one audit-summary row for each column without modifying ``df``."""
    row_count = len(df)
    records: list[dict[str, object]] = []

    for column_name in df.columns:
        series = df[column_name]
        missing_count = int(series.isna().sum())
        numeric = is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype)

        record: dict[str, object] = {
            "table_name": table_name,
            "column_name": str(column_name),
            "dtype": str(series.dtype),
            "row_count": row_count,
            "missing_count": missing_count,
            "missing_pct": round(missing_count / row_count * 100, 2)
            if row_count
            else 0.0,
            "unique_count": int(series.nunique(dropna=True)),
            "zero_count": int(series.eq(0).sum()) if numeric else None,
            "negative_count": int(series.lt(0).sum()) if numeric else None,
            "min_value": series.min() if numeric and not series.empty else None,
            "max_value": series.max() if numeric and not series.empty else None,
        }
        records.append(record)

    return pd.DataFrame.from_records(records, columns=AUDIT_COLUMNS)


def audit_all_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Audit every named DataFrame and concatenate the column summaries."""
    audits = [audit_dataframe(table, name) for name, table in tables.items()]
    if not audits:
        return pd.DataFrame(columns=AUDIT_COLUMNS)

    records = [
        record
        for audit in audits
        for record in audit.to_dict(orient="records")
    ]
    return pd.DataFrame.from_records(records, columns=AUDIT_COLUMNS)


def _count_grain_rows(
    df: pd.DataFrame,
    grain_columns: list[str],
    table_name: str,
) -> pd.DataFrame:
    """Count rows per grain after validating the requested columns."""
    if not grain_columns:
        raise ValueError("grain_columns must contain at least one column")

    missing_columns = [column for column in grain_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise KeyError(f"{table_name}: grain columns not found: {missing}")

    return (
        df.groupby(grain_columns, dropna=False, observed=True)
        .size()
        .rename("row_count")
        .reset_index()
    )


def check_grain(
    df: pd.DataFrame,
    grain_columns: list[str],
    table_name: str,
    sample_size: int = 20,
) -> pd.DataFrame:
    """Return up to ``sample_size`` duplicated grain combinations from ``df``."""
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")

    counts = _count_grain_rows(df, grain_columns, table_name)
    duplicates = counts.loc[counts["row_count"] > 1].head(sample_size).copy()
    duplicates.insert(0, "table_name", table_name)
    return duplicates.reset_index(drop=True)


def summarize_grain(
    df: pd.DataFrame,
    grain_columns: list[str],
    table_name: str,
) -> pd.DataFrame:
    """Return a one-row summary of duplicate combinations for a table grain."""
    counts = _count_grain_rows(df, grain_columns, table_name)
    duplicates = counts.loc[counts["row_count"] > 1, "row_count"]
    duplicate_combinations = len(duplicates)

    return pd.DataFrame(
        [
            {
                "table_name": table_name,
                "grain_columns": list(grain_columns),
                "duplicate_combinations": duplicate_combinations,
                "affected_rows": int(duplicates.sum()),
                "is_valid": duplicate_combinations == 0,
            }
        ],
        columns=GRAIN_SUMMARY_COLUMNS,
    )


def check_foreign_key(
    child_df: pd.DataFrame,
    child_column: str,
    parent_df: pd.DataFrame,
    parent_column: str,
    child_table: str,
    parent_table: str,
) -> pd.DataFrame:
    """Return non-missing child keys absent from the referenced parent column."""
    if child_column not in child_df.columns:
        raise KeyError(f"{child_table}: foreign key column not found: {child_column}")
    if parent_column not in parent_df.columns:
        raise KeyError(f"{parent_table}: referenced key column not found: {parent_column}")

    child_keys = child_df[child_column]
    parent_keys = parent_df[parent_column].dropna()
    invalid_mask = child_keys.notna() & ~child_keys.isin(parent_keys)
    invalid_rows = child_df.loc[invalid_mask, [child_column]]

    if invalid_rows.empty:
        return pd.DataFrame(columns=FOREIGN_KEY_COLUMNS)

    counts = (
        invalid_rows.groupby(child_column, dropna=False, observed=True)
        .size()
        .rename("affected_rows")
        .reset_index()
        .rename(columns={child_column: "invalid_value"})
    )
    counts.insert(0, "parent_column", parent_column)
    counts.insert(0, "parent_table", parent_table)
    counts.insert(0, "child_column", child_column)
    counts.insert(0, "child_table", child_table)
    return counts.loc[:, FOREIGN_KEY_COLUMNS]
