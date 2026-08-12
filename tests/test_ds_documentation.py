"""Keep final Data Science documentation tied to persisted evidence."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
README = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
ROADMAP = (PROJECT_ROOT / "docs" / "data_science_roadmap.md").read_text(
    encoding="utf-8"
)
ARCHITECTURE = (PROJECT_ROOT / "docs" / "architecture.md").read_text(
    encoding="utf-8"
)


def test_readme_distinguishes_all_required_project_phases() -> None:
    required = [
        "Data Engineering",
        "Data Analysis",
        "Power BI",
        "Forecast Readiness",
        "Forecast Modeling",
        "Final Forecast",
    ]

    assert all(phase in README for phase in required)
    assert "Forecasting / Data Science    | **Not started" not in README


def test_documented_selected_metrics_match_persisted_reports() -> None:
    tuning = pd.read_csv(
        PROJECT_ROOT / "reports" / "modeling" / "tuning_results.csv"
    )
    chosen = tuning.loc[tuning["is_chosen"]].iloc[0]
    strongest_baseline = pd.read_csv(
        PROJECT_ROOT / "reports" / "modeling" / "baseline_summary.csv"
    ).sort_values("rmsle_mean", kind="stable").iloc[0]

    for document in (README, ROADMAP):
        assert f"{chosen['rmsle_mean']:.6f}" in document
        assert f"{chosen['rmsle_std']:.6f}" in document
        assert f"{strongest_baseline['rmsle_mean']:.6f}" in document
        assert f"{strongest_baseline['rmsle_std']:.6f}" in document


def test_documentation_includes_exact_reproduction_entrypoints() -> None:
    commands = [
        "python -m pytest -q",
        "python notebooks/13_feature_engineering.py",
        "python notebooks/11_temporal_backtesting.py",
        "python notebooks/14_global_lightgbm.py",
        "python notebooks/16_global_lightgbm_tuning.py",
        "python notebooks/17_forecast_error_analysis.py",
        "python notebooks/20_final_competition_forecast.py",
    ]

    assert all(command in README for command in commands)
    assert all(command in ROADMAP for command in commands)


def test_architecture_records_nonproduction_specialists_and_intervals() -> None:
    assert "routing shadow-only" in ARCHITECTURE
    assert "Evaluated prototype" in ARCHITECTURE
    assert "28,512-row submission" in ARCHITECTURE
