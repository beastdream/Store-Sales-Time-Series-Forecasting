"""Contracts for the separated end-to-end project validator."""

from datetime import datetime
from zoneinfo import ZoneInfo

from src import validate_da_project, validate_ds_project, validate_project


def _timestamp() -> datetime:
    return datetime(2026, 8, 15, 12, 0, tzinfo=ZoneInfo("Asia/Bangkok"))


def test_project_report_preserves_da_and_ds_scope_ownership() -> None:
    outcome = validate_project.ProjectValidationOutcome(
        timestamp=_timestamp(),
        da=validate_da_project.ValidationOutcome(
            timestamp=_timestamp(),
            results=[validate_da_project.CheckResult("warehouse", "PASS", "ok")],
        ),
        ds=validate_ds_project.DSValidationOutcome(
            timestamp=_timestamp(),
            results=[validate_ds_project.CheckResult("model", "PASS", "ok")],
        ),
    )

    report = validate_project.render_report(outcome)

    assert outcome.exit_code == 0
    assert "Overall status: **PASS**" in report
    assert "`src.validate_da_project`" in report
    assert "`src.validate_ds_project`" in report
    assert "final submission contracts" in report


def test_project_failure_propagates_from_either_scope() -> None:
    outcome = validate_project.ProjectValidationOutcome(
        timestamp=_timestamp(),
        da=validate_da_project.ValidationOutcome(timestamp=_timestamp()),
        ds=validate_ds_project.DSValidationOutcome(
            timestamp=_timestamp(),
            results=[validate_ds_project.CheckResult("submission", "FAIL", "bad")],
        ),
    )

    assert outcome.exit_code == 1
    assert "Overall status: **FAIL**" in validate_project.render_report(outcome)
