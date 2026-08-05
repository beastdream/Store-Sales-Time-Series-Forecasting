"""Tests for store and family dimension builders."""

from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_dimensions import build_dim_family, build_dim_store


def _stores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "store_nbr": [3, 1, 2],
            "city": ["Quito", "Cuenca", "Loja"],
            "state": ["Pichincha", "Azuay", "Loja"],
            "store_type": ["D", "A", "B"],
            "cluster": [3, 1, 2],
        }
    )


def test_build_dim_store_uniqueness_key_range_and_row_count() -> None:
    """Store keys start at one and preserve one row per business key."""
    stores = _stores()
    original = stores.copy(deep=True)

    result = build_dim_store(stores)

    assert result.columns.tolist() == [
        "store_key",
        "store_nbr",
        "city",
        "state",
        "store_type",
        "cluster",
    ]
    assert len(result) == len(stores)
    assert result["store_nbr"].is_unique
    assert result["store_nbr"].tolist() == [1, 2, 3]
    assert result["store_key"].tolist() == [1, 2, 3]
    assert not result["store_key"].isna().any()
    pd.testing.assert_frame_equal(stores, original)


def test_build_dim_store_rejects_duplicate_business_key() -> None:
    """Duplicate store numbers cannot receive separate surrogate keys."""
    stores = pd.concat([_stores(), _stores().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="store_nbr"):
        build_dim_store(stores)


def test_build_dim_family_uniqueness_key_range_and_row_count() -> None:
    """Family dimension has one sorted row and one key per distinct family."""
    train = pd.DataFrame({"family": ["SEAFOOD", "BREAD", "SEAFOOD", "DAIRY"]})
    original = train.copy(deep=True)

    result = build_dim_family(train)

    assert result.columns.tolist() == ["family_key", "family"]
    assert len(result) == train["family"].nunique()
    assert result["family"].is_unique
    assert result["family"].tolist() == ["BREAD", "DAIRY", "SEAFOOD"]
    assert result["family_key"].tolist() == [1, 2, 3]
    assert not result["family_key"].isna().any()
    pd.testing.assert_frame_equal(train, original)


def test_dimension_keys_are_stable_for_reordered_input() -> None:
    """Input row order does not change business-to-surrogate key mappings."""
    stores = _stores()
    families = pd.DataFrame({"family": ["SEAFOOD", "BREAD", "DAIRY"]})

    store_a = build_dim_store(stores)
    store_b = build_dim_store(stores.sample(frac=1, random_state=7))
    family_a = build_dim_family(families)
    family_b = build_dim_family(families.sample(frac=1, random_state=7))

    pd.testing.assert_frame_equal(store_a, store_b)
    pd.testing.assert_frame_equal(family_a, family_b)
