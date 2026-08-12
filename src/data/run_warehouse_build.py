"""Build and reconcile the processed analytics warehouse from interim tables."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_INTERIM, DATA_PROCESSED, REPORTS_DIR
from src.data.build_bridges import build_bridge_store_holiday
from src.data.build_date_dimension import build_date_dimension
from src.data.build_dimensions import build_dim_family, build_dim_store
from src.data.build_facts import (
    build_fact_daily_sales,
    build_fact_oil_price,
    build_fact_store_transactions,
)
from src.data.build_store_date_dimension import build_dim_store_date


INTERIM_PATHS = {
    "train": DATA_INTERIM / "train_clean.parquet",
    "test": DATA_INTERIM / "test_clean.parquet",
    "stores": DATA_INTERIM / "stores_clean.parquet",
    "transactions": DATA_INTERIM / "transactions_clean.parquet",
    "oil": DATA_INTERIM / "oil_clean.parquet",
    "holidays": DATA_INTERIM / "holiday_store_daily.parquet",
}

WAREHOUSE_PATHS = {
    "dim_date": DATA_PROCESSED / "dim_date.parquet",
    "dim_store": DATA_PROCESSED / "dim_store.parquet",
    "dim_family": DATA_PROCESSED / "dim_family.parquet",
    "dim_store_date": DATA_PROCESSED / "dim_store_date.parquet",
    "fact_daily_sales": DATA_PROCESSED / "fact_daily_sales.parquet",
    "fact_store_transactions": DATA_PROCESSED / "fact_store_transactions.parquet",
    "fact_oil_price": DATA_PROCESSED / "fact_oil_price.parquet",
    "bridge_store_holiday": DATA_PROCESSED / "bridge_store_holiday.parquet",
}

TABLE_GRAINS = {
    "dim_date": ["date_key"],
    "dim_store": ["store_key"],
    "dim_family": ["family_key"],
    "dim_store_date": ["date_key", "store_key"],
    "fact_daily_sales": ["date_key", "store_key", "family_key"],
    "fact_store_transactions": ["date_key", "store_key"],
    "fact_oil_price": ["date_key"],
    "bridge_store_holiday": ["date_key", "store_key"],
}


def _load_interim_tables() -> dict[str, pd.DataFrame]:
    """Load every required interim table or raise a clear missing-file error."""
    missing = [str(path) for path in INTERIM_PATHS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Warehouse build requires missing interim files: " + ", ".join(missing)
        )
    return {name: pd.read_parquet(path) for name, path in INTERIM_PATHS.items()}


def _load_or_build_dim_date(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Reuse a valid date dimension or rebuild it for the requested range."""
    path = WAREHOUSE_PATHS["dim_date"]
    expected_dates = pd.Series(
        pd.date_range(start_date, end_date, freq="D"),
        name="full_date",
    )
    if path.is_file():
        existing = pd.read_parquet(path)
        required = {"date_key", "full_date"}
        if (
            required.issubset(existing.columns)
            and existing["date_key"].is_unique
            and not existing.isna().any().any()
            and existing["full_date"].reset_index(drop=True).equals(expected_dates)
        ):
            return existing
    return build_date_dimension(start_date, end_date)


def _date_bounds(table: pd.DataFrame) -> tuple[str, str]:
    """Return printable date bounds for a warehouse table when applicable."""
    if table.empty:
        return "N/A", "N/A"
    if "full_date" in table.columns:
        dates = pd.to_datetime(table["full_date"])
    elif "date_key" in table.columns:
        dates = pd.to_datetime(table["date_key"].astype(str), format="%Y%m%d")
    else:
        return "N/A", "N/A"
    return dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")


def _warehouse_validation(
    warehouse: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return table-level grain, key, row-count, and date-range validation."""
    rows: list[dict[str, object]] = []
    for name, table in warehouse.items():
        grain = TABLE_GRAINS[name]
        missing_keys = int(table[grain].isna().sum().sum())
        duplicate_grain = int(table.duplicated(grain).sum())
        minimum_date, maximum_date = _date_bounds(table)
        rows.append(
            {
                "table_name": name,
                "row_count": len(table),
                "expected_grain": " + ".join(grain),
                "duplicate_grain_count": duplicate_grain,
                "missing_surrogate_key_count": missing_keys,
                "minimum_date": minimum_date,
                "maximum_date": maximum_date,
            }
        )
    return pd.DataFrame(rows)


def _format_markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """Format a compact Markdown table without optional dependencies."""
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _build_report(
    validation: pd.DataFrame,
    reconciliations: list[tuple[str, object, object, bool]],
) -> str:
    """Build a Markdown warehouse validation and reconciliation report."""
    validation_rows = validation.astype(object).values.tolist()
    reconciliation_rows = [
        [name, clean_value, warehouse_value, "PASS" if passed else "FAIL"]
        for name, clean_value, warehouse_value, passed in reconciliations
    ]
    return "\n".join(
        [
            "# Warehouse Reconciliation",
            "",
            "All validations and reconciliations passed before warehouse tables were saved.",
            "",
            "## Table validation",
            "",
            _format_markdown_table(
                [
                    "Table name",
                    "Row count",
                    "Expected grain",
                    "Duplicate grain count",
                    "Missing surrogate key count",
                    "Minimum date",
                    "Maximum date",
                ],
                validation_rows,
            ),
            "",
            "## Required reconciliations",
            "",
            _format_markdown_table(
                ["Check", "Clean value", "Warehouse value", "Status"],
                reconciliation_rows,
            ),
            "",
        ]
    )


def run_warehouse_build() -> None:
    """Build, validate, reconcile, and persist the processed warehouse."""
    interim = _load_interim_tables()
    start_date = interim["train"]["date"].min()
    end_date = interim["test"]["date"].max()
    if pd.isna(start_date) or pd.isna(end_date):
        raise RuntimeError("Warehouse build stopped: train/test date range is unavailable")

    dim_date = _load_or_build_dim_date(start_date, end_date)
    dim_store = build_dim_store(interim["stores"])
    dim_family = build_dim_family(interim["train"])
    bridge_store_holiday = build_bridge_store_holiday(
        interim["holidays"], dim_date, dim_store
    )
    dim_store_date = build_dim_store_date(
        dim_date,
        dim_store,
        interim["holidays"],
        interim["train"],
        interim["transactions"],
    )
    warehouse = {
        "dim_date": dim_date,
        "dim_store": dim_store,
        "dim_family": dim_family,
        "dim_store_date": dim_store_date,
        "fact_daily_sales": build_fact_daily_sales(
            interim["train"], dim_date, dim_store, dim_family, dim_store_date
        ),
        "fact_store_transactions": build_fact_store_transactions(
            interim["transactions"], dim_date, dim_store
        ),
        "fact_oil_price": build_fact_oil_price(interim["oil"], dim_date),
        "bridge_store_holiday": bridge_store_holiday,
    }

    validation = _warehouse_validation(warehouse)
    if validation["duplicate_grain_count"].gt(0).any():
        raise RuntimeError("Warehouse build stopped: duplicate grain validation failed")
    if validation["missing_surrogate_key_count"].gt(0).any():
        raise RuntimeError("Warehouse build stopped: missing surrogate key validation failed")

    fact_sales = warehouse["fact_daily_sales"]
    fact_transactions = warehouse["fact_store_transactions"]
    reconciliations = [
        (
            "Total sales",
            interim["train"]["sales"].sum(),
            fact_sales["sales"].sum(),
            bool(
                np.isclose(
                    interim["train"]["sales"].sum(),
                    fact_sales["sales"].sum(),
                    rtol=0,
                    atol=1e-6,
                )
            ),
        ),
        (
            "Total onpromotion",
            int(interim["train"]["onpromotion"].sum()),
            int(fact_sales["onpromotion"].sum()),
            interim["train"]["onpromotion"].sum()
            == fact_sales["onpromotion"].sum(),
        ),
        (
            "Total transactions",
            int(interim["transactions"]["transactions"].sum()),
            int(fact_transactions["transactions"].sum()),
            interim["transactions"]["transactions"].sum()
            == fact_transactions["transactions"].sum(),
        ),
        (
            "Store count",
            len(interim["stores"]),
            len(dim_store),
            len(interim["stores"]) == len(dim_store),
        ),
        (
            "Family count",
            interim["train"]["family"].nunique(),
            len(dim_family),
            interim["train"]["family"].nunique() == len(dim_family),
        ),
        (
            "Store-date row count",
            len(dim_date) * len(dim_store),
            len(dim_store_date),
            len(dim_store_date) == len(dim_date) * len(dim_store),
        ),
        (
            "Store-date holiday mappings",
            len(bridge_store_holiday),
            int(dim_store_date["holiday_count"].gt(0).sum()),
            int(dim_store_date["holiday_count"].gt(0).sum())
            == len(bridge_store_holiday),
        ),
        (
            "Store-date sales observations",
            interim["train"][["date", "store_nbr"]].drop_duplicates().shape[0],
            int(dim_store_date["has_sales_observation"].sum()),
            int(dim_store_date["has_sales_observation"].sum())
            == interim["train"][["date", "store_nbr"]].drop_duplicates().shape[0],
        ),
        (
            "Store-date transaction observations",
            interim["transactions"][["date", "store_nbr"]]
            .drop_duplicates()
            .shape[0],
            int(dim_store_date["has_transaction_observation"].sum()),
            int(dim_store_date["has_transaction_observation"].sum())
            == interim["transactions"][["date", "store_nbr"]]
            .drop_duplicates()
            .shape[0],
        ),
        (
            "Sales fact unmapped date-store keys",
            0,
            int(
                (~fact_sales["date_store_key"].isin(dim_store_date["date_store_key"]))
                .sum()
            ),
            bool(fact_sales["date_store_key"].isin(dim_store_date["date_store_key"]).all()),
        ),
    ]
    failed = [name for name, _, _, passed in reconciliations if not passed]
    if failed:
        raise RuntimeError(
            "Warehouse build stopped: reconciliation failed for " + ", ".join(failed)
        )

    # No warehouse output is written before all validations above have passed.
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, table in warehouse.items():
        table.to_parquet(WAREHOUSE_PATHS[name], index=False)

    report = _build_report(validation, reconciliations)
    report_path: Path = REPORTS_DIR / "data_quality" / "warehouse_reconciliation.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print("Warehouse build completed successfully.")
    for path in (*WAREHOUSE_PATHS.values(), report_path):
        print(path)


if __name__ == "__main__":
    run_warehouse_build()
