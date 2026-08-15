"""Reproduce the file-based DA project validation report with one command."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence
from zoneinfo import ZoneInfo

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from src.config import DATA_INTERIM, DATA_PROCESSED, DATA_RAW, PROJECT_ROOT, REPORTS_DIR


REPORT_PATH = REPORTS_DIR / "da_project_validation.md"
TIMEZONE = ZoneInfo("Asia/Bangkok")
VALID_STATUSES = {"PASS", "WARNING", "FAIL", "NOT RUN"}

INTERIM_PATHS = {
    "train_clean": DATA_INTERIM / "train_clean.parquet",
    "test_clean": DATA_INTERIM / "test_clean.parquet",
    "stores_clean": DATA_INTERIM / "stores_clean.parquet",
    "transactions_clean": DATA_INTERIM / "transactions_clean.parquet",
    "oil_clean": DATA_INTERIM / "oil_clean.parquet",
    "holiday_store_daily": DATA_INTERIM / "holiday_store_daily.parquet",
}
PROCESSED_PATHS = {
    name: DATA_PROCESSED / f"{name}.parquet"
    for name in (
        "dim_date",
        "dim_store",
        "dim_family",
        "dim_store_date",
        "fact_daily_sales",
        "fact_store_transactions",
        "fact_oil_price",
        "bridge_store_holiday",
    )
}
GRAINS = {
    "dim_date": ["date_key"],
    "dim_store": ["store_key"],
    "dim_family": ["family_key"],
    "dim_store_date": ["date_key", "store_key"],
    "fact_daily_sales": ["date_key", "store_key", "family_key"],
    "fact_store_transactions": ["date_key", "store_key"],
    "fact_oil_price": ["date_key"],
    "bridge_store_holiday": ["date_key", "store_key"],
}
RISK_COLUMNS = [
    "is_insufficient_history",
    "is_intermittent",
    "is_promotion_dependent",
    "is_high_volatility",
]
REQUIRED_GITIGNORE_RULES = {
    "data/raw/*.csv",
    "reports/tables/holiday_analysis.csv",
    "reports/tables/promotion_analysis_matched.csv",
    "reports/tables/transactions_analysis.csv",
    "!**/.gitkeep",
}
POSTGRES_ENV = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]


@dataclass(frozen=True)
class CheckResult:
    """One validation result with an explicit, report-safe status."""

    name: str
    status: str
    details: str

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Unsupported validation status: {self.status}")


@dataclass
class ValidationOutcome:
    """All evidence required to render and exit from one validation run."""

    timestamp: datetime
    results: list[CheckResult] = field(default_factory=list)
    reconciliations: list[tuple[str, str, object, str]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    git_hygiene: list[str] = field(default_factory=list)
    powerbi_readiness: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return 1 if any(item.status == "FAIL" for item in self.results) else 0


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _sanitize_output(value: str) -> str:
    """Remove local executable/workspace paths from persisted diagnostics."""
    sanitized = value.replace(str(PROJECT_ROOT), ".")
    sanitized = sanitized.replace(str(PROJECT_ROOT).replace("\\", "/"), ".")
    sanitized = sanitized.replace(str(Path(sys.executable)), "python")
    sanitized = sanitized.replace(
        str(Path(sys.executable)).replace("\\", "/"), "python"
    )
    return sanitized.replace("\\", "/")


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run a project command without invoking a shell."""
    return subprocess.run(
        list(command),
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )


def _command_check(
    name: str,
    command: Sequence[str],
    runner: CommandRunner,
) -> CheckResult:
    completed = runner(command)
    display_command = [
        "python" if index == 0 and Path(part) == Path(sys.executable) else part
        for index, part in enumerate(command)
    ]
    detail = (
        f"`{' '.join(display_command)}` returned exit code {completed.returncode}."
    )
    if completed.returncode == 0:
        output = (completed.stdout or "").strip().splitlines()
        if output:
            detail += f" Last output: {_sanitize_output(output[-1])[:240]}"
        return CheckResult(name, "PASS", detail)
    error = (completed.stderr or completed.stdout or "no output").strip().splitlines()
    if error:
        detail += f" Last output: {_sanitize_output(error[-1])[:240]}"
    return CheckResult(name, "FAIL", detail)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_hashes() -> dict[str, str]:
    return {
        path.name: _sha256(path)
        for path in sorted(DATA_RAW.glob("*.csv"))
        if path.is_file()
    }


def _load_hash_baseline() -> tuple[dict[str, str] | None, str | None]:
    """Read an optional committed hash baseline without creating one silently."""
    json_path = REPORTS_DIR / "data_quality" / "raw_sha256.json"
    if json_path.is_file():
        values = json.loads(json_path.read_text(encoding="utf-8"))
        return {
            str(key): str(value).lower() for key, value in values.items()
        }, json_path.relative_to(PROJECT_ROOT).as_posix()
    for path in (DATA_RAW / "SHA256SUMS", DATA_RAW / "sha256sums.txt"):
        if path.is_file():
            values: dict[str, str] = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    values[Path(parts[-1].lstrip("*")).name] = parts[0].lower()
            return values, path.relative_to(PROJECT_ROOT).as_posix()
    return None, None


def _check_raw_hashes(
    before: dict[str, str], after: dict[str, str]
) -> list[CheckResult]:
    results = [
        CheckResult(
            "Raw SHA-256 unchanged during validation",
            "PASS" if before == after else "FAIL",
            f"Compared {len(before)} raw CSV files before and after pipeline execution.",
        )
    ]
    baseline, source = _load_hash_baseline()
    if baseline is None:
        results.append(
            CheckResult(
                "Raw SHA-256 baseline",
                "WARNING",
                "No baseline hash file exists; pre/post hashes were still compared.",
            )
        )
    else:
        matches = baseline == after
        results.append(
            CheckResult(
                "Raw SHA-256 baseline",
                "PASS" if matches else "FAIL",
                f"Compared current raw hashes with `{source}`.",
            )
        )
    return results


def _check_parquets(outcome: ValidationOutcome) -> None:
    row_counts: dict[str, int] = {}
    failures: list[str] = []
    for name, path in {**INTERIM_PATHS, **PROCESSED_PATHS}.items():
        try:
            row_counts[name] = len(pd.read_parquet(path))
        except Exception as exc:  # validation must report corrupt/missing artifacts
            failures.append(f"{name}: {type(exc).__name__}")
    outcome.results.append(
        CheckResult(
            "Read all required Parquet artifacts",
            "FAIL" if failures else "PASS",
            "; ".join(failures)
            if failures
            else f"Read {len(row_counts)} Parquet files successfully.",
        )
    )
    if failures:
        return

    grain_failures: list[str] = []
    for name, path in PROCESSED_PATHS.items():
        frame = pd.read_parquet(path, columns=GRAINS[name])
        grain = GRAINS[name]
        if frame[grain].isna().any().any() or frame.duplicated(grain).any():
            grain_failures.append(name)
        outcome.reconciliations.append(
            (
                f"{name} row count",
                path.relative_to(PROJECT_ROOT).as_posix(),
                len(frame),
                "PASS",
            )
        )
    outcome.results.append(
        CheckResult(
            "Processed row count and grain",
            "FAIL" if grain_failures else "PASS",
            "Invalid grain: " + ", ".join(grain_failures)
            if grain_failures
            else "All eight processed tables have non-null, unique expected grains.",
        )
    )


def _check_reconciliation(outcome: ValidationOutcome) -> None:
    train = pd.read_parquet(INTERIM_PATHS["train_clean"], columns=["sales"])
    sales_fact = pd.read_parquet(
        PROCESSED_PATHS["fact_daily_sales"], columns=["sales"]
    )
    transactions = pd.read_parquet(
        INTERIM_PATHS["transactions_clean"], columns=["transactions"]
    )
    transaction_fact = pd.read_parquet(PROCESSED_PATHS["fact_store_transactions"])

    sales_values = {
        "interim train": float(train["sales"].sum()),
        "sales fact": float(sales_fact["sales"].sum()),
    }
    store_report = pd.read_csv(REPORTS_DIR / "tables" / "store_performance.csv")
    family_report = pd.read_csv(REPORTS_DIR / "tables" / "family_performance.csv")
    sales_values["store report"] = float(store_report["total_sales"].sum())
    sales_values["family report"] = float(family_report["total_sales"].sum())
    sales_match = all(
        np.isclose(value, sales_values["interim train"], rtol=0, atol=1e-6)
        for value in sales_values.values()
    )
    outcome.results.append(
        CheckResult(
            "Sales reconciliation",
            "PASS" if sales_match else "FAIL",
            "Interim, fact, store report, and family report totals compared with atol=1e-6.",
        )
    )
    for source, value in sales_values.items():
        outcome.reconciliations.append(("Sales volume", source, f"{value:.7f}", "PASS" if sales_match else "FAIL"))

    tx_values = {
        "interim transactions": int(transactions["transactions"].sum()),
        "transaction fact": int(transaction_fact["transactions"].sum()),
        "store report": int(store_report["total_transactions"].sum()),
    }
    transaction_store_report = REPORTS_DIR / "tables" / "transactions_store_summary.csv"
    if transaction_store_report.is_file():
        tx_frame = pd.read_csv(transaction_store_report)
        tx_values["transaction store report"] = int(tx_frame["total_transactions"].sum())
    tx_match = len(set(tx_values.values())) == 1
    outcome.results.append(
        CheckResult(
            "Transactions reconciliation",
            "PASS" if tx_match else "FAIL",
            "Interim, store-day fact, and store-level report totals compared exactly.",
        )
    )
    for source, value in tx_values.items():
        outcome.reconciliations.append(("Transactions", source, value, "PASS" if tx_match else "FAIL"))

    no_double_count = (
        len(transaction_fact) == len(transactions)
        and transaction_fact[["date_key", "store_key"]].drop_duplicates().shape[0]
        == len(transaction_fact)
        and "family_key" not in transaction_fact.columns
    )
    outcome.results.append(
        CheckResult(
            "Transactions are not double-counted by family",
            "PASS" if no_double_count else "FAIL",
            f"Interim rows={len(transactions):,}; fact rows={len(transaction_fact):,}; family_key absent={ 'family_key' not in transaction_fact.columns }.",
        )
    )


def _check_store_date_and_holidays(outcome: ValidationOutcome) -> None:
    dates = pd.read_parquet(PROCESSED_PATHS["dim_date"], columns=["date_key"])
    stores = pd.read_parquet(PROCESSED_PATHS["dim_store"], columns=["store_key"])
    store_date = pd.read_parquet(PROCESSED_PATHS["dim_store_date"])
    bridge = pd.read_parquet(PROCESSED_PATHS["bridge_store_holiday"])
    sales_keys = pd.read_parquet(
        PROCESSED_PATHS["fact_daily_sales"], columns=["date_store_key"]
    )
    transaction_keys = pd.read_parquet(
        PROCESSED_PATHS["fact_store_transactions"],
        columns=["date_key", "store_key"],
    )
    expected_rows = len(dates) * len(stores)
    flags = [
        "is_holiday",
        "is_work_day",
        "is_event",
        "has_sales_observation",
        "has_transaction_observation",
    ]
    complete = (
        len(store_date) == expected_rows
        and not store_date.duplicated(["date_key", "store_key"]).any()
        and all(set(store_date[column].unique()).issubset({0, 1}) for column in flags)
    )
    outcome.results.append(
        CheckResult(
            "dim_store_date completeness",
            "PASS" if complete else "FAIL",
            f"Rows={len(store_date):,}; expected date × store rows={expected_rows:,}.",
        )
    )
    expected_key = store_date["date_key"].astype("int64") * 100 + store_date[
        "store_key"
    ].astype("int64")
    mapped_transactions = transaction_keys.merge(
        store_date[["date_key", "store_key"]],
        on=["date_key", "store_key"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    key_valid = (
        store_date["date_store_key"].equals(expected_key)
        and store_date["date_store_key"].is_unique
        and sales_keys["date_store_key"].isin(store_date["date_store_key"]).all()
        and mapped_transactions["_merge"].eq("both").all()
    )
    outcome.results.append(
        CheckResult(
            "date_store_key contract",
            "PASS" if key_valid else "FAIL",
            "Validated formula, uniqueness, and fact foreign-key coverage.",
        )
    )
    holiday_valid = (
        not bridge.duplicated(["date_key", "store_key"]).any()
        and bridge["holiday_count"].ge(1).all()
        and len(bridge.merge(store_date[["date_key", "store_key"]], on=["date_key", "store_key"], validate="one_to_one"))
        == len(bridge)
        and int(store_date["holiday_count"].gt(0).sum()) == len(bridge)
    )
    outcome.results.append(
        CheckResult(
            "Holiday store mapping",
            "PASS" if holiday_valid else "FAIL",
            f"Validated {len(bridge):,} unique mapped store-day records against dim_store_date.",
        )
    )


def _check_forecast_readiness(outcome: ValidationOutcome) -> None:
    path = REPORTS_DIR / "tables" / "forecast_readiness.csv"
    readiness = pd.read_csv(path)
    stores = pd.read_parquet(PROCESSED_PATHS["dim_store"], columns=["store_key"])
    families = pd.read_parquet(PROCESSED_PATHS["dim_family"], columns=["family_key"])
    binary = [*RISK_COLUMNS, "is_ready"]
    valid = (
        len(readiness) == len(stores) * len(families)
        and not readiness.duplicated(["store_nbr", "family"]).any()
        and all(set(readiness[column].unique()).issubset({0, 1}) for column in binary)
        and readiness["risk_flag_count"].equals(readiness[RISK_COLUMNS].sum(axis=1))
        and readiness.loc[readiness["is_ready"].eq(1), "risk_flag_count"].eq(0).all()
    )
    outcome.results.append(
        CheckResult(
            "Forecast readiness flags",
            "PASS" if valid else "FAIL",
            f"Validated {len(readiness):,} store-family rows, binary flags, risk counts, and Ready rule.",
        )
    )


def _check_report_artifacts(outcome: ValidationOutcome) -> None:
    csv_paths = sorted(
        path
        for directory in (REPORTS_DIR / "tables", REPORTS_DIR / "data_quality")
        for path in directory.glob("*.csv")
    )
    csv_failures: list[str] = []
    for path in csv_paths:
        try:
            pd.read_csv(path)
        except Exception as exc:
            csv_failures.append(f"{path.name}: {type(exc).__name__}")
    outcome.results.append(
        CheckResult(
            "Report CSV artifacts",
            "FAIL" if csv_failures else "PASS",
            "; ".join(csv_failures)
            if csv_failures
            else f"Read {len(csv_paths)} report CSV files successfully.",
        )
    )

    png_paths = sorted((REPORTS_DIR / "figures").rglob("*.png"))
    png_failures: list[str] = []
    for path in png_paths:
        try:
            if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
                raise ValueError("invalid PNG signature")
            image = mpimg.imread(path)
            if image.size == 0:
                raise ValueError("empty decoded image")
        except Exception as exc:
            png_failures.append(f"{path.name}: {type(exc).__name__}")
    outcome.results.append(
        CheckResult(
            "PNG artifacts",
            "FAIL" if png_failures else "PASS",
            "; ".join(png_failures)
            if png_failures
            else f"Decoded {len(png_paths)} valid, non-empty PNG files.",
        )
    )
    outcome.artifacts.extend(
        [
            f"Generated by cleaning: {len(INTERIM_PATHS)} interim Parquet files",
            f"Generated by warehouse build: {len(PROCESSED_PATHS)} processed Parquet files",
            f"Validated existing artifacts: {len(csv_paths)} report CSV files",
            f"Validated existing artifacts: {len(png_paths)} report PNG files",
            "Generated by validator: "
            f"{REPORT_PATH.relative_to(PROJECT_ROOT).as_posix()}",
        ]
    )


def _tracked_files(runner: CommandRunner) -> list[Path]:
    completed = runner(["git", "ls-files"])
    if completed.returncode != 0:
        raise RuntimeError("git ls-files failed")
    return [PROJECT_ROOT / line for line in completed.stdout.splitlines() if line]


def _contains_absolute_personal_path(text: str) -> bool:
    """Detect machine paths while ignoring escaped newline/tab string literals."""
    windows = re.compile(
        r"[A-Za-z]:(?:\\|/)[A-Za-z0-9 ._-]{2,}(?:\\|/)"
    )
    posix = re.compile(r"/(?:home|Users)/[^/\s]+/")
    return bool(windows.search(text) or posix.search(text))


def _check_repository_hygiene(outcome: ValidationOutcome, runner: CommandRunner) -> None:
    try:
        tracked = _tracked_files(runner)
    except RuntimeError as exc:
        outcome.results.append(CheckResult("Git hygiene", "FAIL", str(exc)))
        return
    text_files: list[tuple[Path, str]] = []
    for path in tracked:
        if not path.is_file() or path.suffix.lower() in {".png", ".parquet", ".csv"}:
            continue
        try:
            text_files.append((path, path.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            continue

    secret_patterns = [
        re.compile(r"BEGIN (?:RSA|OPENSSH|EC|DSA) PRIVATE KEY"),
        re.compile(r"postgres(?:ql)?://[^/@\s]+:[^/@\s]+@", re.IGNORECASE),
        re.compile(r"(?:AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})"),
    ]
    secret_files = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path, text in text_files
        if any(pattern.search(text) for pattern in secret_patterns)
    )
    real_env = [path for path in PROJECT_ROOT.rglob(".env") if ".git" not in path.parts]
    secret_ok = not secret_files and not real_env
    outcome.results.append(
        CheckResult(
            "Secret scan",
            "PASS" if secret_ok else "FAIL",
            "No real .env, private key, common token, or credential-bearing database URL found."
            if secret_ok
            else f"Review: {', '.join(secret_files + [str(p) for p in real_env])}",
        )
    )

    absolute_files = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path, text in text_files
        if _contains_absolute_personal_path(text)
    )
    outcome.results.append(
        CheckResult(
            "Absolute personal path scan",
            "PASS" if not absolute_files else "FAIL",
            "No absolute personal-machine path found in tracked text files."
            if not absolute_files
            else "Found in: " + ", ".join(absolute_files),
        )
    )

    ignore_lines = {
        line.strip()
        for line in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing_rules = REQUIRED_GITIGNORE_RULES - ignore_lines
    parquet_tracked = [path for path in tracked if path.suffix.lower() == ".parquet"]
    hygiene_ok = not missing_rules and not parquet_tracked
    outcome.results.append(
        CheckResult(
            "Git hygiene rules",
            "PASS" if hygiene_ok else "FAIL",
            "Required ignore rules exist; .gitkeep is retained; no Parquet is tracked."
            if hygiene_ok
            else f"Missing rules={sorted(missing_rules)}; tracked Parquet={len(parquet_tracked)}.",
        )
    )
    raw_tracked = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in tracked
        if path.parent == DATA_RAW and path.suffix.lower() == ".csv"
    )
    large_ignored_tracked = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in tracked
        if path.relative_to(PROJECT_ROOT).as_posix()
        in {
            "reports/tables/holiday_analysis.csv",
            "reports/tables/promotion_analysis_matched.csv",
            "reports/tables/transactions_analysis.csv",
        }
    )
    if raw_tracked or large_ignored_tracked:
        outcome.results.append(
            CheckResult(
                "Ignored artifacts still tracked",
                "WARNING",
                f"Raw CSV tracked={len(raw_tracked)}; large reproducible report CSV tracked={len(large_ignored_tracked)}. Run documented git rm --cached commands manually.",
            )
        )
    else:
        outcome.results.append(
            CheckResult("Ignored artifacts still tracked", "PASS", "No ignored raw/report artifacts remain in the Git index.")
        )
    outcome.git_hygiene.extend(
        [
            f"Tracked files scanned: {len(tracked)}.",
            f"Raw CSV files still tracked: {len(raw_tracked)}.",
            f"Large reproducible report CSV files still tracked: {len(large_ignored_tracked)}.",
            f"Tracked Parquet files: {len(parquet_tracked)}.",
            "No files were deleted, untracked, or committed by the validator.",
        ]
    )


def _postgres_runtime_status() -> CheckResult:
    missing = [name for name in POSTGRES_ENV if not os.getenv(name)]
    if missing:
        details = "PostgreSQL runtime was not requested; missing configuration: " + ", ".join(missing) + "."
    else:
        details = "PostgreSQL variables are present, but runtime execution is intentionally outside this file-based validator."
    return CheckResult("PostgreSQL runtime", "NOT RUN", details)


def run_validation(runner: CommandRunner = _run_command) -> ValidationOutcome:
    """Execute all file-based checks and return structured evidence."""
    outcome = ValidationOutcome(timestamp=datetime.now(TIMEZONE))
    before_hashes = _raw_hashes()
    outcome.results.extend(
        [
            _command_check("Pytest suite", [sys.executable, "-m", "pytest", "-q"], runner),
            _command_check("Cleaning pipeline", [sys.executable, "-m", "src.data.run_cleaning"], runner),
            _command_check("Warehouse pipeline", [sys.executable, "-m", "src.data.run_warehouse_build"], runner),
        ]
    )
    # Artifact checks continue after command failures so the report is diagnostic.
    try:
        _check_parquets(outcome)
        _check_reconciliation(outcome)
        _check_store_date_and_holidays(outcome)
        _check_forecast_readiness(outcome)
        _check_report_artifacts(outcome)
    except Exception as exc:
        outcome.results.append(
            CheckResult(
                "File-based validation execution",
                "FAIL",
                f"{type(exc).__name__}: {str(exc)[:300]}",
            )
        )
    outcome.results.extend(_check_raw_hashes(before_hashes, _raw_hashes()))
    _check_repository_hygiene(outcome, runner)
    outcome.results.append(
        _command_check(
            "SQL quality dry-run",
            [sys.executable, "-m", "src.run_sql_quality_checks", "--dry-run"],
            runner,
        )
    )
    outcome.results.append(_postgres_runtime_status())
    outcome.powerbi_readiness.extend(
        [
            "Processed model tables and date_store_key relationships are file-validated.",
            "The expected single-direction DimStoreDate model is documented.",
            "Power BI runtime, refresh, visuals, and reconciliation are not run; no .pbix/.pbit is claimed.",
        ]
    )
    return outcome


def _result_section(outcome: ValidationOutcome, status: str) -> list[str]:
    rows = [item for item in outcome.results if item.status == status]
    return [f"- **{item.name}:** {item.details}" for item in rows] or ["- None."]


def render_report(outcome: ValidationOutcome) -> str:
    """Render the required Markdown sections from structured results."""
    environment = [
        f"- Python: `{platform.python_version()}`",
        f"- Platform: `{platform.platform()}`",
        f"- Working directory: repository root (relative paths only in report)",
        f"- PostgreSQL configured: `{'yes' if all(os.getenv(name) for name in POSTGRES_ENV) else 'no'}`",
    ]
    reconciliation_rows = [
        f"| {metric} | `{source}` | {value} | {status} |"
        for metric, source, value, status in outcome.reconciliations
    ] or ["| No reconciliation evidence | N/A | N/A | NOT RUN |"]
    lines = [
        "# DA Project Validation",
        "",
        "## Execution timestamp",
        "",
        outcome.timestamp.isoformat(),
        "",
        "## Environment",
        "",
        *environment,
        "",
        "## Passed checks",
        "",
        *_result_section(outcome, "PASS"),
        "",
        "## Warnings",
        "",
        *_result_section(outcome, "WARNING"),
        "",
        "## Failed checks",
        "",
        *_result_section(outcome, "FAIL"),
        "",
        "## Not-run checks",
        "",
        *_result_section(outcome, "NOT RUN"),
        "",
        "## Data reconciliation",
        "",
        "| Measure/check | Source | Value | Status |",
        "|---|---|---:|---|",
        *reconciliation_rows,
        "",
        "## Artifacts generated",
        "",
        *[f"- {item}" for item in outcome.artifacts],
        "",
        "## Git hygiene",
        "",
        *[f"- {item}" for item in outcome.git_hygiene],
        "",
        "## Power BI readiness",
        "",
        *[f"- {item}" for item in outcome.powerbi_readiness],
        "",
        "## Remaining work",
        "",
        "- Configure and validate PostgreSQL DDL, load, marts, and runtime SQL quality checks.",
        "- Remove ignored raw/report artifacts from the Git index manually if still tracked.",
        "- Build and validate the actual Power BI model, measures, refresh, and report pages.",
        "- Define temporal backtests and forecasting baselines; no model is approved yet.",
        "",
        "## Commands to reproduce",
        "",
        "```powershell",
        "python -m src.validate_da_project",
        "```",
        "",
        "The validator itself runs `pytest`, cleaning, warehouse build, SQL quality dry-run,",
        "and all file/artifact checks. It never runs PostgreSQL runtime, deletes files,",
        "changes business rules, modifies the Git index, or creates a commit.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    """Run validation, always write the report, and fail on serious errors."""
    del argv  # reserved for future non-business CLI options
    try:
        outcome = run_validation()
    except Exception as exc:  # ensure even an orchestration failure is reported
        outcome = ValidationOutcome(timestamp=datetime.now(TIMEZONE))
        outcome.results.append(
            CheckResult("Validator execution", "FAIL", f"{type(exc).__name__}: {str(exc)[:300]}")
        )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(outcome), encoding="utf-8")
    counts = {status: sum(item.status == status for item in outcome.results) for status in VALID_STATUSES}
    print(
        "DA validation: "
        f"{counts['PASS']} PASS, {counts['WARNING']} WARNING, "
        f"{counts['FAIL']} FAIL, {counts['NOT RUN']} NOT RUN"
    )
    print(REPORT_PATH)
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
