"""Data and documentation contracts for the Power BI holiday model."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
DESIGN_PATH = PROJECT_ROOT / "reports" / "data_quality" / "powerbi_model_design.md"


def test_date_store_key_is_unique_and_all_fact_keys_map() -> None:
    dim_store_date = pd.read_parquet(
        PROCESSED / "dim_store_date.parquet",
        columns=["date_store_key", "date_key", "store_key"],
    )
    sales_keys = pd.read_parquet(
        PROCESSED / "fact_daily_sales.parquet",
        columns=["date_store_key"],
    )
    transaction_keys = pd.read_parquet(
        PROCESSED / "fact_store_transactions.parquet",
        columns=["date_store_key"],
    )

    assert dim_store_date["date_store_key"].is_unique
    assert not dim_store_date.duplicated(["date_key", "store_key"]).any()
    assert sales_keys["date_store_key"].isin(dim_store_date["date_store_key"]).all()
    assert transaction_keys["date_store_key"].isin(
        dim_store_date["date_store_key"]
    ).all()


def test_holiday_slicer_dimension_contains_both_holiday_states() -> None:
    slicer = pd.read_parquet(
        PROCESSED / "dim_store_date.parquet",
        columns=["is_holiday", "holiday_count"],
    )

    assert set(slicer["is_holiday"].unique()) == {0, 1}
    assert slicer["holiday_count"].eq(0).any()
    assert slicer["holiday_count"].gt(0).any()


def test_powerbi_design_documents_required_single_direction_model() -> None:
    design = DESIGN_PATH.read_text(encoding="utf-8")
    normalized_design = " ".join(design.split())
    required_relationships = [
        "`DimDate → DimStoreDate`",
        "`DimStore → DimStoreDate`",
        "`DimStoreDate → FactDailySales`",
        "`DimStoreDate → FactStoreTransactions`",
        "`DimFamily → FactDailySales`",
        "`DimDate → FactOilPrice`",
    ]

    assert all(relationship in design for relationship in required_relationships)
    assert "no many-to-many relationships" in normalized_design
    assert "no bidirectional filtering" in normalized_design
    assert "Missing observation is not zero sales" in design
    assert "not the primary slicer table" in design
