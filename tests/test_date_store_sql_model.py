"""Static contracts for the SQL date-store dimension and fact foreign keys."""

from pathlib import Path

from src.load_to_postgres import TABLE_LOAD_ORDER


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_sql_ddl_defines_date_store_dimension_and_fact_foreign_keys() -> None:
    dimensions = (
        PROJECT_ROOT / "sql" / "ddl" / "02_create_dimensions.sql"
    ).read_text(encoding="utf-8").lower()
    facts = (
        PROJECT_ROOT / "sql" / "ddl" / "03_create_facts.sql"
    ).read_text(encoding="utf-8").lower()

    assert "create table if not exists analytics.dim_store_date" in dimensions
    assert "date_store_key bigint primary key" in dimensions
    assert "unique (date_key, store_key)" in dimensions
    assert facts.count("date_store_key bigint not null") == 1
    assert facts.count("references analytics.dim_store_date (date_store_key)") == 1
    assert "foreign key (date_key, store_key)" in facts
    assert "references analytics.dim_store_date (date_key, store_key)" in facts


def test_postgres_load_order_places_date_store_dimension_before_facts() -> None:
    dimension_index = TABLE_LOAD_ORDER.index("dim_store_date")

    assert dimension_index < TABLE_LOAD_ORDER.index("fact_daily_sales")
    assert dimension_index < TABLE_LOAD_ORDER.index("fact_store_transactions")
