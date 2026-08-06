"""Execute PostgreSQL warehouse quality checks and write a Markdown report."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import sqlparse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.config import REPORTS_DIR, SQL_DIR
from src.database import get_engine, test_connection


QUALITY_SQL_PATHS = [
    SQL_DIR / "data_quality" / "01_row_counts.sql",
    SQL_DIR / "data_quality" / "02_duplicate_grain.sql",
    SQL_DIR / "data_quality" / "03_foreign_keys.sql",
    SQL_DIR / "data_quality" / "04_measure_validation.sql",
    SQL_DIR / "data_quality" / "05_mart_reconciliation.sql",
]

REPORT_PATH = REPORTS_DIR / "data_quality" / "sql_quality_report.md"
RESULT_COLUMNS = {
    "check_name",
    "severity",
    "actual_value",
    "expected_value",
    "passed",
    "details",
}


def _read_statements(path: Path) -> list[str]:
    """Read complete SQL statements without splitting semicolons in literals."""
    if not path.is_file():
        raise FileNotFoundError(f"SQL quality file not found: {path.name}")
    return [
        statement.strip()
        for statement in sqlparse.split(path.read_text(encoding="utf-8"))
        if statement.strip()
    ]


def _status(passed: bool, severity: str) -> str:
    """Classify a check as PASS or its configured WARNING/FAIL severity."""
    if passed:
        return "PASS"
    normalized = severity.upper()
    return normalized if normalized in {"WARNING", "FAIL"} else "FAIL"


def _safe_markdown(value: object) -> str:
    """Escape a scalar for use inside a Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _write_report(results: list[dict[str, Any]]) -> None:
    """Write all SQL check outcomes and their status totals to Markdown."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    status_counts = {
        status: sum(result["status"] == status for result in results)
        for status in ("PASS", "WARNING", "FAIL")
    }
    lines = [
        "# SQL Quality Report",
        "",
        f"- PASS: `{status_counts['PASS']}`",
        f"- WARNING: `{status_counts['WARNING']}`",
        f"- FAIL: `{status_counts['FAIL']}`",
        "",
        "| SQL file | Check | Status | Actual | Expected | Details |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                _safe_markdown(result[key])
                for key in (
                    "sql_file",
                    "check_name",
                    "status",
                    "actual_value",
                    "expected_value",
                    "details",
                )
            )
            + " |"
        )
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_sql_quality_checks() -> list[dict[str, Any]]:
    """Run every SQL quality file and return classified check results."""
    if not test_connection():
        results = [
            {
                "sql_file": "connection",
                "check_name": "database_connection",
                "status": "FAIL",
                "actual_value": "unavailable",
                "expected_value": "available",
                "details": "Database connection is unavailable; SQL checks were not run.",
            }
        ]
        _write_report(results)
        return results

    engine = get_engine()
    results: list[dict[str, Any]] = []
    try:
        with engine.connect() as connection:
            for path in QUALITY_SQL_PATHS:
                try:
                    for statement in _read_statements(path):
                        rows = connection.execute(text(statement)).mappings().all()
                        for row in rows:
                            missing_columns = RESULT_COLUMNS.difference(row.keys())
                            if missing_columns:
                                raise RuntimeError(
                                    f"Quality result missing columns in {path.name}"
                                )
                            passed = bool(row["passed"])
                            results.append(
                                {
                                    "sql_file": path.name,
                                    "check_name": row["check_name"],
                                    "status": _status(passed, str(row["severity"])),
                                    "actual_value": row["actual_value"],
                                    "expected_value": row["expected_value"],
                                    "details": row["details"],
                                }
                            )
                except (OSError, RuntimeError, SQLAlchemyError):
                    results.append(
                        {
                            "sql_file": path.name,
                            "check_name": "sql_file_execution",
                            "status": "FAIL",
                            "actual_value": "error",
                            "expected_value": "successful execution",
                            "details": "The SQL file could not be executed; inspect server logs.",
                        }
                    )
    finally:
        engine.dispose()

    _write_report(results)
    return results


def dry_run_sql_quality_checks() -> dict[str, int]:
    """Parse every quality SQL statement without creating a database connection."""
    statement_count = 0
    for path in QUALITY_SQL_PATHS:
        statements = _read_statements(path)
        if not statements:
            raise RuntimeError(f"SQL quality file contains no statements: {path.name}")
        for statement in statements:
            if not sqlparse.parse(statement):
                raise RuntimeError(f"SQL statement could not be parsed: {path.name}")
        statement_count += len(statements)
    return {"file_count": len(QUALITY_SQL_PATHS), "statement_count": statement_count}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for runtime or connection-free validation."""
    parser = argparse.ArgumentParser(description="Run warehouse SQL quality checks.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse all quality SQL files without connecting to PostgreSQL.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run SQL checks and return nonzero when any serious failure exists."""
    args = _parse_args(argv)
    if args.dry_run:
        result = dry_run_sql_quality_checks()
        print(
            "SQL quality dry-run passed: "
            f"{result['file_count']} files, {result['statement_count']} statements"
        )
        return 0

    results = run_sql_quality_checks()
    failures = sum(result["status"] == "FAIL" for result in results)
    warnings = sum(result["status"] == "WARNING" for result in results)
    passes = sum(result["status"] == "PASS" for result in results)
    print(f"SQL quality checks: {passes} PASS, {warnings} WARNING, {failures} FAIL")
    print(REPORT_PATH)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
