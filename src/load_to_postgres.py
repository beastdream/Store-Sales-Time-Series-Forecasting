"""Load the validated processed warehouse into PostgreSQL transactionally."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sqlparse
from sqlalchemy import text
from sqlalchemy.engine import Connection

from src.config import DATA_PROCESSED, SQL_DIR
from src.database import get_engine, test_connection


DDL_PATHS = [
    SQL_DIR / "ddl" / "01_create_schemas.sql",
    SQL_DIR / "ddl" / "02_create_dimensions.sql",
    SQL_DIR / "ddl" / "03_create_facts.sql",
]

TABLE_LOAD_ORDER = [
    "dim_date",
    "dim_store",
    "dim_family",
    "dim_store_date",
    "fact_daily_sales",
    "fact_store_transactions",
    "fact_oil_price",
    "bridge_store_holiday",
]

PARQUET_PATHS = {
    table: DATA_PROCESSED / f"{table}.parquet" for table in TABLE_LOAD_ORDER
}

DIM_DATE_BOOLEAN_COLUMNS = [
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_payday",
]


def _read_sql_statements(path: Path) -> list[str]:
    """Read complete SQL statements without splitting semicolons in literals."""
    if not path.is_file():
        raise FileNotFoundError(f"DDL file not found: {path.name}")
    return [
        statement.strip()
        for statement in sqlparse.split(path.read_text(encoding="utf-8"))
        if statement.strip()
    ]


def _execute_ddl(connection: Connection) -> None:
    """Execute schema, dimension, and fact DDL scripts in their required order."""
    for path in DDL_PATHS:
        for statement in _read_sql_statements(path):
            connection.exec_driver_sql(statement)


def _truncate_warehouse(connection: Connection) -> None:
    """Truncate all warehouse tables only after explicit CLI authorization."""
    qualified_tables = ", ".join(
        f"analytics.{table}" for table in reversed(TABLE_LOAD_ORDER)
    )
    connection.exec_driver_sql(f"TRUNCATE TABLE {qualified_tables}")


def _load_tables(
    connection: Connection,
    *,
    chunk_size: int,
) -> dict[str, Any]:
    """Append dimensions first and facts/bridge second using chunked inserts."""
    missing = [path.name for path in PARQUET_PATHS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Processed Parquet files not found: " + ", ".join(missing)
        )

    source_metrics: dict[str, Any] = {"row_counts": {}}
    for table_name in TABLE_LOAD_ORDER:
        frame = pd.read_parquet(PARQUET_PATHS[table_name])
        source_metrics["row_counts"][table_name] = len(frame)

        if table_name == "dim_date":
            source_metrics["minimum_date"] = frame["full_date"].min().date()
            source_metrics["maximum_date"] = frame["full_date"].max().date()
            frame = frame.copy()
            for column in DIM_DATE_BOOLEAN_COLUMNS:
                frame[column] = frame[column].astype(bool)
        elif table_name == "dim_store":
            source_metrics["store_count"] = len(frame)
        elif table_name == "dim_family":
            source_metrics["family_count"] = len(frame)
        elif table_name == "fact_daily_sales":
            source_metrics["total_sales"] = frame["sales"].sum()
            source_metrics["total_onpromotion"] = int(frame["onpromotion"].sum())
        elif table_name == "fact_store_transactions":
            source_metrics["total_transactions"] = int(frame["transactions"].sum())

        frame.to_sql(
            table_name,
            con=connection,
            schema="analytics",
            if_exists="append",
            index=False,
            chunksize=chunk_size,
        )
    return source_metrics


def _validate_loaded_warehouse(
    connection: Connection,
    source_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Reconcile PostgreSQL row counts, measures, dimensions, and date range."""
    database_counts = {
        table: int(
            connection.execute(
                text(f"SELECT COUNT(*) FROM analytics.{table}")
            ).scalar_one()
        )
        for table in TABLE_LOAD_ORDER
    }
    for table, expected_count in source_metrics["row_counts"].items():
        if database_counts[table] != expected_count:
            raise RuntimeError(f"Post-load row count validation failed for {table}")

    database_sales = connection.execute(
        text("SELECT COALESCE(SUM(sales), 0) FROM analytics.fact_daily_sales")
    ).scalar_one()
    database_onpromotion = int(
        connection.execute(
            text(
                "SELECT COALESCE(SUM(onpromotion), 0) "
                "FROM analytics.fact_daily_sales"
            )
        ).scalar_one()
    )
    database_transactions = int(
        connection.execute(
            text(
                "SELECT COALESCE(SUM(transactions), 0) "
                "FROM analytics.fact_store_transactions"
            )
        ).scalar_one()
    )
    database_store_count = database_counts["dim_store"]
    database_family_count = database_counts["dim_family"]
    minimum_date, maximum_date = connection.execute(
        text("SELECT MIN(full_date), MAX(full_date) FROM analytics.dim_date")
    ).one()

    if not np.isclose(
        float(database_sales),
        float(source_metrics["total_sales"]),
        rtol=0,
        atol=1e-6,
    ):
        raise RuntimeError("Post-load total sales validation failed")
    if database_onpromotion != source_metrics["total_onpromotion"]:
        raise RuntimeError("Post-load total onpromotion validation failed")
    if database_transactions != source_metrics["total_transactions"]:
        raise RuntimeError("Post-load total transactions validation failed")
    if database_store_count != source_metrics["store_count"]:
        raise RuntimeError("Post-load store count validation failed")
    if database_family_count != source_metrics["family_count"]:
        raise RuntimeError("Post-load family count validation failed")
    if (
        minimum_date != source_metrics["minimum_date"]
        or maximum_date != source_metrics["maximum_date"]
    ):
        raise RuntimeError("Post-load date range validation failed")

    return {
        "row_counts": database_counts,
        "total_sales": database_sales,
        "total_onpromotion": database_onpromotion,
        "total_transactions": database_transactions,
        "store_count": database_store_count,
        "family_count": database_family_count,
        "minimum_date": minimum_date,
        "maximum_date": maximum_date,
    }


def load_to_postgres(
    *,
    truncate: bool = False,
    chunk_size: int = 50_000,
) -> dict[str, Any]:
    """Run DDL, optionally truncate, load, and validate in one transaction."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if not test_connection():
        raise RuntimeError("Database connection check failed")

    engine = get_engine()
    try:
        with engine.begin() as connection:
            _execute_ddl(connection)
            if truncate:
                _truncate_warehouse(connection)
            source_metrics = _load_tables(connection, chunk_size=chunk_size)
            return _validate_loaded_warehouse(connection, source_metrics)
    finally:
        engine.dispose()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the PostgreSQL load."""
    parser = argparse.ArgumentParser(
        description="Load processed Store Sales warehouse tables into PostgreSQL."
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Explicitly truncate warehouse tables before loading.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Rows per pandas SQL insert chunk (default: 50000).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI without exposing connection credentials on failure."""
    args = _parse_args(argv)
    try:
        result = load_to_postgres(
            truncate=args.truncate,
            chunk_size=args.chunk_size,
        )
    except Exception:
        print(
            "PostgreSQL load failed and the transaction was rolled back. "
            "Check database availability, environment configuration, and server logs."
        )
        return 1

    print("PostgreSQL warehouse load completed successfully.")
    for table_name, row_count in result["row_counts"].items():
        print(f"{table_name}: {row_count} rows")
    print(f"total_sales: {result['total_sales']}")
    print(f"total_transactions: {result['total_transactions']}")
    print(f"store_count: {result['store_count']}")
    print(f"family_count: {result['family_count']}")
    print(f"date_range: {result['minimum_date']} to {result['maximum_date']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
