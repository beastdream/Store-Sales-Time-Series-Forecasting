"""Load the raw Store Sales dataset tables without transforming their values."""

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from src.config import DATA_RAW


def _load_csv(
    filename: str,
    *,
    dtype: Mapping[str, str],
    parse_dates: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Read one raw CSV after verifying that it exists."""
    path: Path = DATA_RAW / filename
    if not path.is_file():
        raise FileNotFoundError(f"Raw data file not found: {filename} (expected at {path})")

    return pd.read_csv(path, dtype=dtype, parse_dates=list(parse_dates))


def load_train() -> pd.DataFrame:
    """Load ``train.csv`` with memory-efficient dtypes and a parsed date column."""
    return _load_csv(
        "train.csv",
        dtype={
            "id": "uint32",
            "store_nbr": "uint8",
            "family": "category",
            "sales": "float32",
            "onpromotion": "uint16",
        },
        parse_dates=("date",),
    )


def load_test() -> pd.DataFrame:
    """Load ``test.csv`` with memory-efficient dtypes and a parsed date column."""
    return _load_csv(
        "test.csv",
        dtype={
            "id": "uint32",
            "store_nbr": "uint8",
            "family": "category",
            "onpromotion": "uint16",
        },
        parse_dates=("date",),
    )


def load_stores() -> pd.DataFrame:
    """Load ``stores.csv`` with compact numeric and categorical dtypes."""
    return _load_csv(
        "stores.csv",
        dtype={
            "store_nbr": "uint8",
            "city": "category",
            "state": "category",
            "type": "category",
            "cluster": "uint8",
        },
    )


def load_transactions() -> pd.DataFrame:
    """Load ``transactions.csv`` with a parsed date column."""
    return _load_csv(
        "transactions.csv",
        dtype={"store_nbr": "uint8", "transactions": "uint32"},
        parse_dates=("date",),
    )


def load_oil() -> pd.DataFrame:
    """Load ``oil.csv`` with a parsed date column."""
    return _load_csv(
        "oil.csv",
        dtype={"dcoilwtico": "float32"},
        parse_dates=("date",),
    )


def load_holidays() -> pd.DataFrame:
    """Load ``holidays_events.csv`` with a parsed date column."""
    return _load_csv(
        "holidays_events.csv",
        dtype={
            "type": "category",
            "locale": "category",
            "locale_name": "category",
            "description": "category",
            "transferred": "bool",
        },
        parse_dates=("date",),
    )


def load_sample_submission() -> pd.DataFrame:
    """Load ``sample_submission.csv`` with compact numeric dtypes."""
    return _load_csv(
        "sample_submission.csv",
        dtype={"id": "uint32", "sales": "float32"},
    )


def load_all_raw_tables() -> dict[str, pd.DataFrame]:
    """Load and return every supported raw dataset table by its logical name."""
    return {
        "train": load_train(),
        "test": load_test(),
        "stores": load_stores(),
        "transactions": load_transactions(),
        "oil": load_oil(),
        "holidays": load_holidays(),
        "sample_submission": load_sample_submission(),
    }
