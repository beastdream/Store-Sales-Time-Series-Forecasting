"""Artifact contracts for overlapping forecast-readiness risk flags."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
READINESS_PATH = PROJECT_ROOT / "reports" / "tables" / "forecast_readiness.csv"
PROCESSED = PROJECT_ROOT / "data" / "processed"
RISK_COLUMNS = [
    "is_insufficient_history",
    "is_intermittent",
    "is_promotion_dependent",
    "is_high_volatility",
]


def test_forecast_readiness_flags_are_binary_and_counted_independently() -> None:
    readiness = pd.read_csv(READINESS_PATH)

    assert set(readiness["readiness_class"].unique()) == {
        "Ready",
        "Ready with caution",
        "Intermittent demand",
        "Insufficient history",
        "High volatility",
        "Promotion dependent",
    }
    for column in [*RISK_COLUMNS, "is_ready"]:
        assert set(readiness[column].unique()).issubset({0, 1})
    expected_count = readiness[RISK_COLUMNS].sum(axis=1)
    pd.testing.assert_series_equal(
        readiness["risk_flag_count"],
        expected_count,
        check_names=False,
        check_dtype=False,
    )
    assert readiness.loc[readiness["is_ready"].eq(1), "risk_flag_count"].eq(0).all()


def test_forecast_readiness_supports_overlapping_risks() -> None:
    readiness = pd.read_csv(READINESS_PATH)
    overlapping = readiness.loc[readiness["risk_flag_count"].ge(2)]

    assert not overlapping.empty
    assert overlapping[RISK_COLUMNS].sum(axis=1).ge(2).all()


def test_forecast_readiness_retains_complete_store_family_matrix() -> None:
    readiness = pd.read_csv(READINESS_PATH)
    stores = pd.read_parquet(PROCESSED / "dim_store.parquet", columns=["store_key"])
    families = pd.read_parquet(
        PROCESSED / "dim_family.parquet", columns=["family_key"]
    )

    assert len(readiness) == len(stores) * len(families)
    assert not readiness.duplicated(["store_nbr", "family"]).any()


def test_forecast_readiness_report_contains_required_overlap_summaries() -> None:
    report = (PROJECT_ROOT / "reports" / "forecast_readiness.md").read_text(
        encoding="utf-8"
    )
    required_sections = [
        "## Phân bố readiness",
        "### Số chuỗi theo từng flag độc lập",
        "### Số chuỗi theo số lượng risk flags",
        "### Family có nhiều overlapping risks",
        "### Store có nhiều overlapping risks",
        "thứ tự ưu tiên đã công bố",
    ]

    assert all(section in report for section in required_sections)
