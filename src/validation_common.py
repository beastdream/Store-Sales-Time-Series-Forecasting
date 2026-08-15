"""Shared status vocabulary for project validation entrypoints."""

from dataclasses import dataclass


VALID_STATUSES = {"PASS", "WARNING", "FAIL", "NOT RUN"}


@dataclass(frozen=True)
class CheckResult:
    """One validation result with an explicit, report-safe status."""

    name: str
    status: str
    details: str

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Unsupported validation status: {self.status}")
