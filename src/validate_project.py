"""Orchestrate separated DA and DS validation into one project status report."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable
from zoneinfo import ZoneInfo

from src import validate_da_project, validate_ds_project
from src.config import PROJECT_ROOT, REPORTS_DIR


REPORT_PATH = REPORTS_DIR / "project_validation.md"
TIMEZONE = ZoneInfo("Asia/Bangkok")


@dataclass(frozen=True)
class ProjectValidationOutcome:
    timestamp: datetime
    da: validate_da_project.ValidationOutcome
    ds: validate_ds_project.DSValidationOutcome

    @property
    def exit_code(self) -> int:
        return 1 if self.da.exit_code or self.ds.exit_code else 0


def run_validation() -> ProjectValidationOutcome:
    """Run the DA and DS validators without collapsing their scopes."""
    return ProjectValidationOutcome(
        timestamp=datetime.now(TIMEZONE),
        da=validate_da_project.run_validation(),
        ds=validate_ds_project.run_validation(),
    )


def _counts(results: list[object]) -> str:
    return ", ".join(
        f"{status}={sum(item.status == status for item in results)}"
        for status in ("PASS", "WARNING", "FAIL", "NOT RUN")
    )


def render_report(outcome: ProjectValidationOutcome) -> str:
    status = "PASS" if outcome.exit_code == 0 else "FAIL"
    lines = [
        "# End-to-End Project Validation",
        "",
        f"- Execution timestamp: `{outcome.timestamp.isoformat()}`",
        f"- Overall status: **{status}**",
        f"- DA scope: {_counts(outcome.da.results)}",
        f"- DS scope: {_counts(outcome.ds.results)}",
        "",
        "## Scope ownership",
        "",
        "- `src.validate_da_project`: raw/processed data, warehouse, EDA/report "
        "artifacts, repository hygiene, and local Power BI artifact existence.",
        "- `src.validate_ds_project`: temporal splits, baselines, current modeling "
        "reports, chosen metadata, final model, and final submission contracts.",
        "",
        "## Detailed evidence",
        "",
        f"- `{validate_da_project.REPORT_PATH.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- `{validate_ds_project.REPORT_PATH.relative_to(PROJECT_ROOT).as_posix()}`",
        "",
        "## Command to reproduce",
        "",
        "```powershell",
        "python -m src.validate_project",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    del argv
    outcome = run_validation()
    validate_da_project.REPORT_PATH.write_text(
        validate_da_project.render_report(outcome.da), encoding="utf-8"
    )
    validate_ds_project.REPORT_PATH.write_text(
        validate_ds_project.render_report(outcome.ds), encoding="utf-8"
    )
    REPORT_PATH.write_text(render_report(outcome), encoding="utf-8")
    print(f"Project validation: {'PASS' if outcome.exit_code == 0 else 'FAIL'}")
    print(REPORT_PATH.relative_to(PROJECT_ROOT).as_posix())
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
