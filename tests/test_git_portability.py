"""Regression contracts for cross-platform Git and artifact policy."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_gitattributes_normalizes_text_and_protects_binary_artifacts() -> None:
    attributes = (PROJECT_ROOT / ".gitattributes").read_text(encoding="utf-8")

    assert "* text=auto eol=lf" in attributes
    for pattern in ("*.pbix", "*.parquet", "*.png", "*.jpg", "*.jpeg"):
        assert f"{pattern} binary" in attributes


def test_gitignore_keeps_only_canonical_model_contract() -> None:
    lines = {
        line.strip()
        for line in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "models/*" in lines
    assert "!models/global_lightgbm_chosen_config.json" in lines
    assert "!models/final_global_lightgbm.txt" in lines
    assert "!models/final_global_lightgbm_metadata.json" in lines
    assert "data/raw/*.csv" in lines
    assert "reports/tables/holiday_analysis.csv" in lines
    assert "reports/tables/promotion_analysis_matched.csv" in lines
    assert "reports/tables/transactions_analysis.csv" in lines


def test_canonical_model_and_pbix_artifacts_exist() -> None:
    required = [
        "models/global_lightgbm_chosen_config.json",
        "models/final_global_lightgbm.txt",
        "models/final_global_lightgbm_metadata.json",
        "powerbi/store_sales_analytics.pbix",
    ]

    assert all(
        (PROJECT_ROOT / relative_path).is_file()
        and (PROJECT_ROOT / relative_path).stat().st_size > 0
        for relative_path in required
    )
