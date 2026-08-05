"""Tests for column-level DataFrame auditing."""

from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.audit import (
    AUDIT_COLUMNS,
    audit_all_tables,
    audit_dataframe,
    check_foreign_key,
    check_grain,
    summarize_grain,
)


def test_numeric_column_statistics() -> None:
    """Numeric columns report counts and extrema."""
    result = audit_dataframe(pd.DataFrame({"value": [2, 4, 4]}), "numbers")
    row = result.iloc[0]

    assert row["unique_count"] == 2
    assert row["min_value"] == 2
    assert row["max_value"] == 4


def test_string_column_has_no_numeric_statistics() -> None:
    """String columns leave numeric-only fields empty."""
    result = audit_dataframe(pd.DataFrame({"label": ["a", "b"]}), "labels")
    row = result.iloc[0]

    for field in ("zero_count", "negative_count", "min_value", "max_value"):
        assert pd.isna(row[field])


def test_missing_values_and_percentage() -> None:
    """Missing counts and percentages are correct and rounded to two decimals."""
    result = audit_dataframe(pd.DataFrame({"value": [1.0, None, None]}), "sample")
    row = result.iloc[0]

    assert row["row_count"] == 3
    assert row["missing_count"] == 2
    assert row["missing_pct"] == 66.67


def test_negative_values_are_counted() -> None:
    """Negative numeric values are counted."""
    result = audit_dataframe(pd.DataFrame({"value": [-2, -1, 3]}), "sample")

    assert result.loc[0, "negative_count"] == 2


def test_zero_values_are_counted() -> None:
    """Numeric zero values are counted."""
    result = audit_dataframe(pd.DataFrame({"value": [0, 0, 1]}), "sample")

    assert result.loc[0, "zero_count"] == 2


def test_empty_dataframe_with_columns() -> None:
    """An empty DataFrame returns one safe audit row per declared column."""
    frame = pd.DataFrame(
        {
            "number": pd.Series(dtype="float64"),
            "label": pd.Series(dtype="string"),
        }
    )
    result = audit_dataframe(frame, "empty")

    assert list(result.columns) == AUDIT_COLUMNS
    assert len(result) == 2
    assert result["row_count"].eq(0).all()
    assert result["missing_pct"].eq(0.0).all()
    assert result["unique_count"].eq(0).all()


def test_empty_dataframe_without_columns() -> None:
    """A completely empty DataFrame returns an empty result with audit columns."""
    result = audit_dataframe(pd.DataFrame(), "empty")

    assert result.empty
    assert list(result.columns) == AUDIT_COLUMNS


def test_audit_does_not_modify_input() -> None:
    """Auditing leaves the source DataFrame unchanged."""
    frame = pd.DataFrame({"value": [1.0, None], "label": ["a", "b"]})
    original = frame.copy(deep=True)

    audit_dataframe(frame, "sample")

    pd.testing.assert_frame_equal(frame, original)


def test_audit_all_tables_combines_results() -> None:
    """Audits from all named tables are concatenated."""
    tables = {
        "first": pd.DataFrame({"a": [1], "b": ["x"]}),
        "second": pd.DataFrame({"c": [0]}),
    }

    result = audit_all_tables(tables)

    assert len(result) == 3
    assert result["table_name"].tolist() == ["first", "first", "second"]


def test_audit_all_tables_handles_empty_mapping() -> None:
    """No input tables produce an empty result with the stable schema."""
    result = audit_all_tables({})

    assert result.empty
    assert list(result.columns) == AUDIT_COLUMNS


def test_check_grain_valid() -> None:
    """A valid grain returns no duplicate combinations and a valid summary."""
    frame = pd.DataFrame({"id": [1, 2], "value": ["a", "b"]})

    duplicates = check_grain(frame, ["id"], "sample")
    summary = summarize_grain(frame, ["id"], "sample").iloc[0]

    assert duplicates.empty
    assert bool(summary["is_valid"])
    assert summary["duplicate_combinations"] == 0
    assert summary["affected_rows"] == 0


def test_check_grain_duplicates_and_sample_limit() -> None:
    """Duplicated grains are counted without deleting or hiding affected rows."""
    frame = pd.DataFrame({"id": [1, 1, 2, 2, 2, 3], "value": list("abcdef")})
    original = frame.copy(deep=True)

    duplicates = check_grain(frame, ["id"], "sample", sample_size=1)
    summary = summarize_grain(frame, ["id"], "sample").iloc[0]

    assert len(duplicates) == 1
    assert duplicates.columns.tolist() == ["table_name", "id", "row_count"]
    assert duplicates.loc[0, "table_name"] == "sample"
    assert summary["duplicate_combinations"] == 2
    assert summary["affected_rows"] == 5
    assert not bool(summary["is_valid"])
    pd.testing.assert_frame_equal(frame, original)


def test_check_grain_missing_column() -> None:
    """A missing grain column raises a clear error."""
    frame = pd.DataFrame({"id": [1]})

    with pytest.raises(KeyError, match="missing_column"):
        check_grain(frame, ["missing_column"], "sample")


def test_check_grain_empty_dataframe() -> None:
    """An empty DataFrame with its grain column is valid and has no duplicates."""
    frame = pd.DataFrame({"id": pd.Series(dtype="int64")})

    duplicates = check_grain(frame, ["id"], "empty")
    summary = summarize_grain(frame, ["id"], "empty").iloc[0]

    assert duplicates.empty
    assert duplicates.columns.tolist() == ["table_name", "id", "row_count"]
    assert bool(summary["is_valid"])
    assert summary["duplicate_combinations"] == 0
    assert summary["affected_rows"] == 0


def test_foreign_key_valid() -> None:
    """Valid child keys produce no foreign-key violations."""
    child = pd.DataFrame({"store_nbr": [1, 2, 1]})
    parent = pd.DataFrame({"store_nbr": [1, 2]})

    result = check_foreign_key(
        child, "store_nbr", parent, "store_nbr", "train", "stores"
    )

    assert result.empty


def test_foreign_key_invalid() -> None:
    """Invalid child keys are grouped with their affected row counts."""
    child = pd.DataFrame({"store_nbr": [1, 9, 9, 10]})
    parent = pd.DataFrame({"store_nbr": [1, 2]})

    result = check_foreign_key(
        child, "store_nbr", parent, "store_nbr", "train", "stores"
    )

    assert result["invalid_value"].tolist() == [9, 10]
    assert result["affected_rows"].tolist() == [2, 1]
    assert result.loc[0, "child_table"] == "train"
    assert result.loc[0, "parent_table"] == "stores"


def test_foreign_key_missing_is_not_invalid() -> None:
    """Missing child keys remain distinct from invalid foreign keys."""
    child = pd.DataFrame({"store_nbr": [1, None, 9]})
    parent = pd.DataFrame({"store_nbr": [1, None]})
    original_child = child.copy(deep=True)
    original_parent = parent.copy(deep=True)

    result = check_foreign_key(
        child, "store_nbr", parent, "store_nbr", "train", "stores"
    )

    assert result["invalid_value"].tolist() == [9.0]
    assert result["affected_rows"].tolist() == [1]
    pd.testing.assert_frame_equal(child, original_child)
    pd.testing.assert_frame_equal(parent, original_parent)


@pytest.mark.parametrize(
    ("child_column", "parent_column", "missing_name"),
    [
        ("missing_child", "id", "missing_child"),
        ("id", "missing_parent", "missing_parent"),
    ],
)
def test_foreign_key_missing_column_has_clear_error(
    child_column: str,
    parent_column: str,
    missing_name: str,
) -> None:
    """Missing child or parent columns raise errors naming the absent column."""
    child = pd.DataFrame({"id": [1]})
    parent = pd.DataFrame({"id": [1]})

    with pytest.raises(KeyError, match=missing_name):
        check_foreign_key(
            child, child_column, parent, parent_column, "child", "parent"
        )
