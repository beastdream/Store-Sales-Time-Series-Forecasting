"""Contracts for the read-only Data Science artifact validator."""

from datetime import datetime
from zoneinfo import ZoneInfo

from src import validate_ds_project as validator


def test_current_ds_artifacts_pass_structural_validation() -> None:
    outcome = validator.run_validation()

    assert outcome.results
    assert outcome.exit_code == 0
    assert {item.status for item in outcome.results} == {"PASS"}
    assert {item.name for item in outcome.results} >= {
        "Temporal split configuration",
        "Baseline artifacts",
        "Current modeling reports",
        "Selected model metadata",
        "Final model artifact",
        "Final submission schema and row count",
        "Final submission IDs",
        "Final submission predictions",
    }


def test_ds_report_states_structural_and_quality_boundaries() -> None:
    outcome = validator.DSValidationOutcome(
        timestamp=datetime(2026, 8, 15, 12, 0, tzinfo=ZoneInfo("Asia/Bangkok")),
        results=[validator.CheckResult("artifact", "PASS", "exists")],
    )

    report = validator.render_report(outcome)

    assert "do not independently prove model quality" in report
    assert "Final-horizon accuracy is unknown" in report
    assert "never retrains, tunes, forecasts" in report
    assert "historical evidence" in report


def test_ds_validator_does_not_import_training_or_forecast_entrypoints() -> None:
    source = validator.Path(validator.__file__).read_text(encoding="utf-8")

    assert "train_global_model" not in source
    assert "recursive_forecast" not in source
    assert "generate_final" not in source
