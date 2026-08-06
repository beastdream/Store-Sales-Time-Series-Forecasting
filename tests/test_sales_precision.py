"""End-to-end precision contract tests for the sales measure."""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.data import load_raw
from src.data.build_facts import build_fact_daily_sales
from src.data.clean_core import clean_train


def test_sales_precision_survives_raw_clean_fact_and_parquet(
    monkeypatch,
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw"
    raw_path.mkdir()
    (raw_path / "train.csv").write_text(
        "id,date,store_nbr,family,sales,onpromotion\n"
        "1,2020-01-01,1,A,1.1234567,0\n"
        "2,2020-01-01,1,B,2.7654321,1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(load_raw, "DATA_RAW", raw_path)

    raw = load_raw.load_train()
    clean = clean_train(raw)
    dim_date = pd.DataFrame(
        {"date_key": [20200101], "full_date": pd.to_datetime(["2020-01-01"])}
    )
    dim_store = pd.DataFrame({"store_key": [1], "store_nbr": [1]})
    dim_family = pd.DataFrame({"family_key": [1, 2], "family": ["A", "B"]})
    dim_store_date = pd.DataFrame(
        {"date_store_key": [2020010101], "date_key": [20200101], "store_key": [1]}
    )
    fact = build_fact_daily_sales(
        clean, dim_date, dim_store, dim_family, dim_store_date
    )

    parquet_path = tmp_path / "fact_daily_sales.parquet"
    fact.to_parquet(parquet_path, index=False)
    persisted = pd.read_parquet(parquet_path)

    expected_values = np.array([1.1234567, 2.7654321], dtype="float64")
    assert raw["sales"].dtype == "float64"
    assert clean["sales"].dtype == "float64"
    assert fact["sales"].dtype == "float64"
    assert persisted["sales"].dtype == "float64"
    np.testing.assert_array_equal(raw["sales"].to_numpy(), expected_values)
    np.testing.assert_array_equal(clean["sales"].to_numpy(), expected_values)
    np.testing.assert_array_equal(fact["sales"].to_numpy(), expected_values)
    source_total = raw["sales"].sum()
    assert abs(source_total - clean["sales"].sum()) <= 1e-6
    assert abs(source_total - fact["sales"].sum()) <= 1e-6


def test_fact_daily_sales_ddl_uses_required_numeric_precision() -> None:
    ddl_path = Path(__file__).resolve().parents[1] / "sql" / "ddl" / "03_create_facts.sql"
    ddl = ddl_path.read_text(encoding="utf-8")

    assert re.search(r"\bsales\s+NUMERIC\s*\(\s*20\s*,\s*7\s*\)", ddl, re.I)
    assert not re.search(r"\bsales\s+NUMERIC\s*\(\s*18\s*,\s*4\s*\)", ddl, re.I)
