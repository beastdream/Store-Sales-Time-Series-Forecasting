"""Tests for connection-free SQL quality validation."""

from src import run_sql_quality_checks as quality_runner


def test_sql_quality_dry_run_parses_files_without_database(monkeypatch) -> None:
    monkeypatch.setattr(
        quality_runner,
        "test_connection",
        lambda: (_ for _ in ()).throw(AssertionError("database must not be used")),
    )

    result = quality_runner.dry_run_sql_quality_checks()

    assert result["file_count"] == len(quality_runner.QUALITY_SQL_PATHS)
    assert result["statement_count"] >= result["file_count"]


def test_sql_quality_dry_run_cli_returns_success_without_database(monkeypatch) -> None:
    monkeypatch.setattr(
        quality_runner,
        "test_connection",
        lambda: (_ for _ in ()).throw(AssertionError("database must not be used")),
    )

    assert quality_runner.main(["--dry-run"]) == 0
