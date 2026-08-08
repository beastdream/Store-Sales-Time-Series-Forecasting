"""Artifact contracts for the forecast problem-definition audit."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "10_forecast_problem_definition.py"
REPORT_PATH = (
    PROJECT_ROOT / "reports" / "modeling" / "forecast_problem_definition.md"
)


def test_forecast_problem_report_contains_verified_contract() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")
    required = [
        "| Forecast target | sales |",
        "| Forecast grain | store × family × day |",
        "| Historical period | 2013-01-01 through 2017-08-15 |",
        "| Test forecast period | 2017-08-16 through 2017-08-31 |",
        "| Forecast horizon | 16 calendar days |",
        "| Stores | 54 |",
        "| Product families | 33 |",
        "| Store-family series | 1,782 |",
        "| Expected predictions | 28,512 |",
        "| Test ID | id |",
    ]

    assert all(item in report for item in required)


def test_feature_audit_documents_required_availability_and_leakage_rules() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")
    required_features = [
        "Calendar",
        "Store metadata",
        "Family",
        "onpromotion",
        "Holiday / event",
        "Oil",
        "Transactions",
        "Historical sales lags",
        "ForecastReadiness outputs",
        "SalesAnomalies outputs",
    ]

    assert all(feature in report for feature in required_features)
    assert "Future-known in the supplied Kaggle test" in report
    assert "Do not use current-day transactions" in report
    assert report.count("do not use automatically unless") == 2
    assert "competition oil source" in report
    assert "production forecast may not know future oil prices" in report


def test_missing_observations_remain_distinct_from_zero_sales() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")

    assert "Missing observation date != zero sales" in report
    assert "`216` of `91,152` store-days" in report
    assert "has_sales_observation = 0" in report
    assert "must not be silently materialized as `sales = 0`" in report


def test_notebook_scope_writes_only_the_definition_report() -> None:
    source = NOTEBOOK_PATH.read_text(encoding="utf-8")

    assert 'REPORT_PATH = REPORTS_DIR / "modeling" / "forecast_problem_definition.md"' in source
    assert ".to_parquet(" not in source
    assert "model.fit(" not in source
    assert "submission" not in source.lower().replace(
        "or submission is created", ""
    )
