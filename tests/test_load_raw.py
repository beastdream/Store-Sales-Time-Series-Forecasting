"""Tests for the raw Store Sales table loaders."""

from collections.abc import Callable
from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_raw


@pytest.fixture
def sample_raw_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a minimal raw-data directory and point the loaders to it."""
    csv_files = {
        "train.csv": (
            "id,date,store_nbr,family,sales,onpromotion\n"
            "1,2017-01-01,1,GROCERY I,10.5,2\n"
        ),
        "test.csv": (
            "id,date,store_nbr,family,onpromotion\n"
            "2,2017-01-02,1,GROCERY I,0\n"
        ),
        "stores.csv": (
            "store_nbr,city,state,type,cluster\n"
            "1,Quito,Pichincha,D,13\n"
        ),
        "transactions.csv": (
            "date,store_nbr,transactions\n"
            "2017-01-01,1,1200\n"
        ),
        "oil.csv": "date,dcoilwtico\n2017-01-01,52.3\n",
        "holidays_events.csv": (
            "date,type,locale,locale_name,description,transferred\n"
            "2017-01-01,Holiday,National,Ecuador,New Year,False\n"
        ),
        "sample_submission.csv": "id,sales\n2,0.0\n",
    }

    for filename, content in csv_files.items():
        (tmp_path / filename).write_text(content, encoding="utf-8")

    monkeypatch.setattr(load_raw, "DATA_RAW", tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    "loader",
    [
        load_raw.load_train,
        load_raw.load_test,
        load_raw.load_stores,
        load_raw.load_transactions,
        load_raw.load_oil,
        load_raw.load_holidays,
        load_raw.load_sample_submission,
    ],
)
def test_each_loader_returns_dataframe(
    sample_raw_dir: Path,
    loader: Callable[[], pd.DataFrame],
) -> None:
    """Each public table loader returns a pandas DataFrame."""
    assert isinstance(loader(), pd.DataFrame)


@pytest.mark.parametrize(
    ("loader", "expected_columns"),
    [
        (
            load_raw.load_train,
            {"id", "date", "store_nbr", "family", "sales", "onpromotion"},
        ),
        (
            load_raw.load_test,
            {"id", "date", "store_nbr", "family", "onpromotion"},
        ),
        (
            load_raw.load_stores,
            {"store_nbr", "city", "state", "type", "cluster"},
        ),
    ],
)
def test_table_has_expected_columns(
    sample_raw_dir: Path,
    loader: Callable[[], pd.DataFrame],
    expected_columns: set[str],
) -> None:
    """Core tables expose all required columns."""
    assert expected_columns.issubset(loader().columns)


@pytest.mark.parametrize(
    "loader",
    [load_raw.load_transactions, load_raw.load_oil, load_raw.load_holidays],
)
def test_date_column_is_datetime(
    sample_raw_dir: Path,
    loader: Callable[[], pd.DataFrame],
) -> None:
    """Date-based supporting tables parse their date column as datetime."""
    frame = loader()
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])


def test_missing_file_raises_error_with_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing raw file raises FileNotFoundError naming that file."""
    monkeypatch.setattr(load_raw, "DATA_RAW", tmp_path / "missing")

    with pytest.raises(FileNotFoundError, match="train[.]csv"):
        load_raw.load_train()


def test_load_all_raw_tables_returns_expected_mapping(sample_raw_dir: Path) -> None:
    """The aggregate loader returns every supported table by logical name."""
    tables = load_raw.load_all_raw_tables()

    assert list(tables) == [
        "train",
        "test",
        "stores",
        "transactions",
        "oil",
        "holidays",
        "sample_submission",
    ]
    assert all(isinstance(table, pd.DataFrame) for table in tables.values())
