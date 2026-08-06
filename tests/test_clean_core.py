"""Tests for core train, test, and store cleaning rules."""

from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.clean_core import clean_stores, clean_test, clean_train


def _train_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [2, 1],
            "date": pd.to_datetime(["2017-01-02", "2017-01-01"]),
            "store_nbr": [1, 1],
            "family": ["B", "A"],
            "sales": pd.Series([3.1234567, 0.0], dtype="float64"),
            "onpromotion": [2, 0],
        }
    )


def test_clean_train_copies_deduplicates_sorts_and_keeps_zero_sales() -> None:
    """Train cleaning follows the requested safe transformations."""
    frame = _train_frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    original = frame.copy(deep=True)

    result = clean_train(frame)

    assert len(result) == 2
    assert result["family"].tolist() == ["A", "B"]
    assert result["is_promotion"].tolist() == [0, 1]
    assert result["sales"].eq(0).sum() == 1
    assert pd.api.types.is_float_dtype(result["sales"])
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    ("column", "value"),
    [("sales", -1.0), ("onpromotion", -1)],
)
def test_clean_train_rejects_negative_values(column: str, value: float) -> None:
    """Negative sales and promotion counts are rejected."""
    frame = _train_frame()
    frame.loc[0, column] = value

    with pytest.raises(ValueError, match=column):
        clean_train(frame)


def test_clean_train_rejects_duplicate_grain() -> None:
    """Different rows at the same train grain raise an error."""
    frame = _train_frame()
    conflicting = frame.iloc[[0]].copy()
    conflicting["id"] = 99
    conflicting["sales"] = 8.0

    with pytest.raises(ValueError, match="duplicate grain"):
        clean_train(pd.concat([frame, conflicting], ignore_index=True))


def test_clean_test_deduplicates_adds_flag_and_sorts() -> None:
    """Test cleaning deduplicates, flags promotions, and sorts by sales grain."""
    frame = _train_frame().drop(columns="sales")
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    result = clean_test(frame)

    assert len(result) == 2
    assert result["family"].tolist() == ["A", "B"]
    assert result["is_promotion"].tolist() == [0, 1]


def test_clean_test_rejects_negative_promotion() -> None:
    """Negative test promotion counts are rejected."""
    frame = _train_frame().drop(columns="sales")
    frame.loc[0, "onpromotion"] = -1

    with pytest.raises(ValueError, match="onpromotion"):
        clean_test(frame)


def test_clean_test_rejects_duplicate_grain() -> None:
    """Different test rows at the same grain raise an error."""
    frame = _train_frame().drop(columns="sales")
    conflicting = frame.iloc[[0]].copy()
    conflicting["id"] = 99

    with pytest.raises(ValueError, match="duplicate grain"):
        clean_test(pd.concat([frame, conflicting], ignore_index=True))


def test_clean_stores_normalizes_columns_and_trims_metadata() -> None:
    """Store cleaning renames columns and only strips metadata content."""
    frame = pd.DataFrame(
        {
            "Store Nbr": [1, 2],
            "City": [" Quito ", "Cuenca"],
            "STATE": [" Pichincha", "Azuay "],
            "Type": [" D ", "A"],
            "Cluster": [13, 1],
        }
    )
    original = frame.copy(deep=True)

    result = clean_stores(frame)

    assert result.columns.tolist() == [
        "store_nbr",
        "city",
        "state",
        "store_type",
        "cluster",
    ]
    assert result["city"].tolist() == ["Quito", "Cuenca"]
    assert result["state"].tolist() == ["Pichincha", "Azuay"]
    assert result["store_type"].tolist() == ["D", "A"]
    pd.testing.assert_frame_equal(frame, original)


def test_clean_stores_rejects_duplicate_store_number() -> None:
    """Store numbers must remain unique."""
    frame = pd.DataFrame(
        {
            "store_nbr": [1, 1],
            "city": ["Quito", "Quito"],
            "state": ["Pichincha", "Pichincha"],
            "type": ["D", "D"],
        }
    )

    with pytest.raises(ValueError, match="store_nbr"):
        clean_stores(frame)
