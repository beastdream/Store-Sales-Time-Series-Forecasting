"""Responsibility-boundary tests for the cleaning pipeline entrypoint."""

import inspect
from pathlib import Path

import pandas as pd

from src.data import run_cleaning as cleaning_pipeline


def test_cleaning_entrypoint_has_no_warehouse_dependencies() -> None:
    source = inspect.getsource(cleaning_pipeline)

    assert "DATA_PROCESSED" not in source
    assert "build_dimensions" not in source
    assert "build_facts" not in source
    assert "build_bridges" not in source
    assert "build_date_dimension" not in source


def test_cleaning_pipeline_writes_only_six_interim_files(
    monkeypatch,
    tmp_path: Path,
) -> None:
    day_one = pd.Timestamp("2020-01-01")
    day_two = pd.Timestamp("2020-01-02")
    raw_tables = {
        "train": pd.DataFrame(
            {
                "date": [day_one],
                "store_nbr": [1],
                "family": ["GROCERY I"],
                "sales": [10.0],
                "onpromotion": [0],
            }
        ),
        "test": pd.DataFrame(
            {
                "date": [day_two],
                "store_nbr": [1],
                "family": ["GROCERY I"],
                "onpromotion": [0],
            }
        ),
        "stores": pd.DataFrame({"store_nbr": [1]}),
        "transactions": pd.DataFrame(
            {"date": [day_one], "store_nbr": [1], "transactions": [5]}
        ),
        "oil": pd.DataFrame({"date": [day_one], "dcoilwtico": [50.0]}),
        "holidays": pd.DataFrame(
            {"date": [day_one], "type": ["Holiday"], "transferred": [False]}
        ),
    }
    clean_tables = {
        **raw_tables,
        "oil": pd.DataFrame(
            {
                "date": [day_one, day_two],
                "oil_price": [50.0, 50.0],
                "oil_was_imputed": [False, True],
            }
        ),
        "holidays": pd.DataFrame({"date": [day_one], "store_nbr": [1]}),
    }
    interim_dir = tmp_path / "data" / "interim"
    processed_dir = tmp_path / "data" / "processed"
    output_paths = {
        "train": interim_dir / "train_clean.parquet",
        "test": interim_dir / "test_clean.parquet",
        "stores": interim_dir / "stores_clean.parquet",
        "transactions": interim_dir / "transactions_clean.parquet",
        "oil": interim_dir / "oil_clean.parquet",
        "holidays": interim_dir / "holiday_store_daily.parquet",
    }
    written_paths: list[Path] = []

    monkeypatch.setattr(cleaning_pipeline, "load_all_raw_tables", lambda: raw_tables)
    monkeypatch.setattr(cleaning_pipeline, "DATA_INTERIM", interim_dir)
    monkeypatch.setattr(cleaning_pipeline, "clean_train", lambda _: clean_tables["train"])
    monkeypatch.setattr(cleaning_pipeline, "clean_test", lambda _: clean_tables["test"])
    monkeypatch.setattr(cleaning_pipeline, "clean_stores", lambda _: clean_tables["stores"])
    monkeypatch.setattr(
        cleaning_pipeline,
        "clean_transactions",
        lambda _: clean_tables["transactions"],
    )
    monkeypatch.setattr(
        cleaning_pipeline,
        "clean_oil",
        lambda _raw, _start, _end: clean_tables["oil"],
    )
    monkeypatch.setattr(
        cleaning_pipeline,
        "clean_holidays",
        lambda _raw, _stores: clean_tables["holidays"],
    )
    monkeypatch.setattr(cleaning_pipeline, "OUTPUT_PATHS", output_paths)
    monkeypatch.setattr(cleaning_pipeline, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda _frame, path, index=False: written_paths.append(Path(path)),
    )

    cleaning_pipeline.run_cleaning()

    assert written_paths == list(output_paths.values())
    assert len(written_paths) == 6
    assert all(path.parent == interim_dir for path in written_paths)
    assert all(processed_dir not in path.parents for path in written_paths)
    assert (tmp_path / "reports" / "data_quality" / "cleaning_summary.md").is_file()
