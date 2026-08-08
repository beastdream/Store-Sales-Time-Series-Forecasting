"""Unit tests for the reproducible DA validation orchestrator."""

from datetime import datetime
from subprocess import CompletedProcess
from zoneinfo import ZoneInfo

import pytest

from src import validate_da_project as validator


def test_check_result_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="Unsupported validation status"):
        validator.CheckResult("bad", "SUCCESS", "invalid vocabulary")


def test_command_check_uses_mocked_runner_and_maps_exit_codes() -> None:
    success = validator._command_check(
        "mock pass",
        ["python", "-m", "example"],
        lambda command: CompletedProcess(command, 0, "completed\n", ""),
    )
    failure = validator._command_check(
        "mock fail",
        ["python", "-m", "example"],
        lambda command: CompletedProcess(command, 7, "", "failed safely\n"),
    )

    assert success.status == "PASS"
    assert failure.status == "FAIL"
    assert "exit code 7" in failure.details


def test_command_check_sanitizes_machine_paths() -> None:
    workspace_output = f"{validator.PROJECT_ROOT}\\reports\\result.md\n"
    result = validator._command_check(
        "sanitized",
        [validator.sys.executable, "-m", "example"],
        lambda command: CompletedProcess(command, 0, workspace_output, ""),
    )

    assert str(validator.PROJECT_ROOT) not in result.details
    assert str(validator.sys.executable) not in result.details
    assert "`python -m example`" in result.details


def test_absolute_path_scan_ignores_escape_sequences_but_detects_paths() -> None:
    assert not validator._contains_absolute_personal_path('"one segment:\\n\\n"')
    assert validator._contains_absolute_personal_path("D:\\Project Folder\\reports\\x.md")
    assert validator._contains_absolute_personal_path("/home/example/project/file.py")


def test_postgres_runtime_is_not_run_without_configuration(monkeypatch) -> None:
    for variable in validator.POSTGRES_ENV:
        monkeypatch.delenv(variable, raising=False)

    result = validator._postgres_runtime_status()

    assert result.status == "NOT RUN"
    assert "missing configuration" in result.details


def test_render_report_contains_every_required_section() -> None:
    outcome = validator.ValidationOutcome(
        timestamp=datetime(2026, 8, 6, 12, 0, tzinfo=ZoneInfo("Asia/Bangkok")),
        results=[
            validator.CheckResult("good", "PASS", "ok"),
            validator.CheckResult("caution", "WARNING", "review"),
            validator.CheckResult("bad", "FAIL", "broken"),
            validator.CheckResult("database", "NOT RUN", "not configured"),
        ],
        reconciliations=[("Sales", "fact", 10, "PASS")],
        artifacts=["one artifact"],
        git_hygiene=["clean"],
        powerbi_readiness=["designed only"],
    )

    report = validator.render_report(outcome)

    required = [
        "## Execution timestamp",
        "## Environment",
        "## Passed checks",
        "## Warnings",
        "## Failed checks",
        "## Not-run checks",
        "## Data reconciliation",
        "## Artifacts generated",
        "## Git hygiene",
        "## Power BI readiness",
        "## Remaining work",
        "## Commands to reproduce",
    ]
    assert all(section in report for section in required)
    assert outcome.exit_code == 1


def test_main_writes_report_and_returns_nonzero_on_failure(monkeypatch, tmp_path) -> None:
    outcome = validator.ValidationOutcome(
        timestamp=datetime.now(ZoneInfo("Asia/Bangkok")),
        results=[validator.CheckResult("critical", "FAIL", "expected test failure")],
    )
    report_path = tmp_path / "validation.md"
    monkeypatch.setattr(validator, "run_validation", lambda: outcome)
    monkeypatch.setattr(validator, "REPORT_PATH", report_path)

    exit_code = validator.main([])

    assert exit_code == 1
    assert report_path.is_file()
    assert "expected test failure" in report_path.read_text(encoding="utf-8")
