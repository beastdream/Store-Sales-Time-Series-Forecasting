"""Validate persisted forecasting contracts and artifacts without retraining."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROJECT_ROOT, REPORTS_DIR
from src.data.load_raw import load_test, load_train
from src.modeling.final_forecast import validate_final_test_contract
from src.modeling.predict import load_model
from src.modeling.splits import make_rolling_splits
from src.validation_common import CheckResult, VALID_STATUSES


REPORT_PATH = REPORTS_DIR / "ds_project_validation.md"
TIMEZONE = ZoneInfo("Asia/Bangkok")
MODELING_DIR = REPORTS_DIR / "modeling"
BASELINE_SCORES_PATH = MODELING_DIR / "baseline_scores.csv"
BASELINE_SUMMARY_PATH = MODELING_DIR / "baseline_summary.csv"
CONFIG_PATH = MODELS_DIR / "global_lightgbm_chosen_config.json"
FINAL_MODEL_PATH = MODELS_DIR / "final_global_lightgbm.txt"
FINAL_METADATA_PATH = MODELS_DIR / "final_global_lightgbm_metadata.json"
SUBMISSION_PATH = MODELING_DIR / "final_submission.csv"
CURRENT_REPORTS = [
    MODELING_DIR / "recursive_vs_previous_strategy.md",
    MODELING_DIR / "ablation_summary.md",
    MODELING_DIR / "tuning_summary.md",
    MODELING_DIR / "final_forecast_report.md",
]
EXPECTED_FOLDS = [
    ("2017-06-12", "2017-06-13", "2017-06-28"),
    ("2017-06-28", "2017-06-29", "2017-07-14"),
    ("2017-07-14", "2017-07-15", "2017-07-30"),
    ("2017-07-30", "2017-07-31", "2017-08-15"),
]


@dataclass
class DSValidationOutcome:
    """Structured evidence for the forecasting artifact validator."""

    timestamp: datetime
    results: list[CheckResult] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return 1 if any(item.status == "FAIL" for item in self.results) else 0


def _temporal_split_check(train: pd.DataFrame) -> CheckResult:
    splits = make_rolling_splits(train["date"].max(), horizon=16, n_folds=4)
    actual = [
        (
            split.train_end.date().isoformat(),
            split.validation_start.date().isoformat(),
            split.validation_end.date().isoformat(),
        )
        for split in splits
    ]
    return CheckResult(
        "Temporal split configuration",
        "PASS" if actual == EXPECTED_FOLDS else "FAIL",
        f"Observed rolling-origin folds: {actual}.",
    )


def _baseline_check() -> CheckResult:
    scores = pd.read_csv(BASELINE_SCORES_PATH)
    summary = pd.read_csv(BASELINE_SUMMARY_PATH)
    required_metrics = ["rmsle_mean", "rmsle_std", "mae_mean", "wape_mean"]
    strongest = summary.sort_values("rmsle_mean", kind="stable").iloc[0]
    valid = (
        not scores.empty
        and scores.groupby("model")["fold"].nunique().eq(4).all()
        and summary[required_metrics].notna().all().all()
        and np.isfinite(summary[required_metrics].to_numpy(dtype="float64")).all()
    )
    return CheckResult(
        "Baseline artifacts",
        "PASS" if valid else "FAIL",
        "Validated four-fold baseline structure and finite recorded metrics; "
        f"minimum recorded mean RMSLE belongs to {strongest['model']}. This is an "
        "artifact check, not an independent model-quality claim.",
    )


def _modeling_reports_check() -> CheckResult:
    missing = [path for path in CURRENT_REPORTS if not path.is_file() or path.stat().st_size == 0]
    return CheckResult(
        "Current modeling reports",
        "PASS" if not missing else "FAIL",
        (
            "Current recursive backtest, ablation, tuning, and final forecast reports exist."
            if not missing
            else "Missing: " + ", ".join(path.name for path in missing)
        ),
    )


def _load_metadata() -> tuple[dict[str, object], dict[str, object]]:
    return (
        json.loads(CONFIG_PATH.read_text(encoding="utf-8")),
        json.loads(FINAL_METADATA_PATH.read_text(encoding="utf-8")),
    )


def _selected_metadata_check() -> CheckResult:
    config, metadata = _load_metadata()
    config_required = {
        "chosen_experiment", "feature_set_name", "feature_list", "parameters",
        "temporal_folds", "chosen_mean_rmsle", "chosen_rmsle_std",
        "final_test_used_for_selection", "strongest_baseline",
    }
    metadata_required = {
        "model_type", "feature_set_name", "feature_list", "inference_strategy",
        "training_cutoff", "forecast_start", "forecast_end", "target_transform",
        "selected_parameters", "validation_metrics", "baseline_comparison",
    }
    valid = (
        config_required <= set(config)
        and metadata_required <= set(metadata)
        and config["feature_list"] == metadata["feature_list"]
        and config["parameters"] == metadata["selected_parameters"]
        and config["final_test_used_for_selection"] is False
        and metadata.get("final_test_used_for_model_selection") is False
    )
    return CheckResult(
        "Selected model metadata",
        "PASS" if valid else "FAIL",
        "Chosen configuration and final metadata are complete and mutually consistent.",
    )


def _final_model_check() -> CheckResult:
    config, _ = _load_metadata()
    exists = FINAL_MODEL_PATH.is_file() and FINAL_MODEL_PATH.stat().st_size > 0
    if not exists:
        return CheckResult("Final model artifact", "FAIL", "Final model is missing or empty.")
    model = load_model(FINAL_MODEL_PATH)
    valid = (
        model.num_trees() == int(config["num_boost_round"])
        and model.feature_name() == list(config["feature_list"])
    )
    return CheckResult(
        "Final model artifact",
        "PASS" if valid else "FAIL",
        f"Loaded {FINAL_MODEL_PATH.relative_to(PROJECT_ROOT).as_posix()} with "
        f"{model.num_trees()} trees and {len(model.feature_name())} features.",
    )


def _submission_checks(train: pd.DataFrame, test: pd.DataFrame) -> list[CheckResult]:
    if not SUBMISSION_PATH.is_file():
        return [CheckResult("Final submission existence", "FAIL", "Submission is missing.")]
    submission = pd.read_csv(SUBMISSION_PATH)
    validate_final_test_contract(test, train)
    schema_valid = list(submission.columns) == ["id", "sales"] and len(submission) == len(test)
    ids_valid = (
        submission["id"].notna().all()
        and submission["id"].is_unique
        and np.array_equal(submission["id"].to_numpy(), test["id"].to_numpy())
    )
    try:
        predictions = pd.to_numeric(submission["sales"], errors="raise").to_numpy(dtype="float64")
        values_valid = np.isfinite(predictions).all() and (predictions >= 0).all()
    except (TypeError, ValueError):
        values_valid = False
    return [
        CheckResult(
            "Final submission existence",
            "PASS",
            f"Found {SUBMISSION_PATH.relative_to(PROJECT_ROOT).as_posix()}.",
        ),
        CheckResult(
            "Final submission schema and row count",
            "PASS" if schema_valid else "FAIL",
            f"Observed columns={list(submission.columns)}, rows={len(submission):,}; "
            f"test rows={len(test):,}.",
        ),
        CheckResult(
            "Final submission IDs",
            "PASS" if ids_valid else "FAIL",
            "Submission IDs are unique and match exact original test order."
            if ids_valid else "Submission IDs are missing, duplicated, or misordered.",
        ),
        CheckResult(
            "Final submission predictions",
            "PASS" if values_valid else "FAIL",
            "Predictions are numeric, finite, non-missing, and nonnegative."
            if values_valid else "Predictions violate numeric/finite/nonnegative constraints.",
        ),
    ]


def _record(outcome: DSValidationOutcome, name: str, check: Callable[[], CheckResult | list[CheckResult]]) -> None:
    try:
        result = check()
        outcome.results.extend(result if isinstance(result, list) else [result])
    except Exception as exc:
        outcome.results.append(
            CheckResult(name, "FAIL", f"{type(exc).__name__}: {str(exc)[:300]}")
        )


def run_validation() -> DSValidationOutcome:
    """Inspect current DS artifacts without training, tuning, or forecasting."""
    outcome = DSValidationOutcome(timestamp=datetime.now(TIMEZONE))
    try:
        train = load_train()
        test = load_test()
    except Exception as exc:
        outcome.results.append(
            CheckResult("Raw forecasting inputs", "FAIL", f"{type(exc).__name__}: {exc}")
        )
        return outcome
    _record(outcome, "Temporal split configuration", lambda: _temporal_split_check(train))
    _record(outcome, "Baseline artifacts", _baseline_check)
    _record(outcome, "Current modeling reports", _modeling_reports_check)
    _record(outcome, "Selected model metadata", _selected_metadata_check)
    _record(outcome, "Final model artifact", _final_model_check)
    _record(outcome, "Final submission", lambda: _submission_checks(train, test))
    return outcome


def _section(outcome: DSValidationOutcome, status: str) -> list[str]:
    rows = [item for item in outcome.results if item.status == status]
    return [f"- **{item.name}:** {item.details}" for item in rows] or ["- None."]


def render_report(outcome: DSValidationOutcome) -> str:
    lines = [
        "# Data Science Project Validation",
        "",
        "> Scope: temporal-validation configuration and persisted forecasting "
        "artifacts. Existence and structural checks do not independently prove "
        "model quality; recorded CV metrics remain the comparison evidence.",
        "",
        "## Execution timestamp",
        "",
        outcome.timestamp.isoformat(),
        "",
        "## Environment",
        "",
        f"- Python: `{platform.python_version()}`",
        "- Working directory: repository root",
        "",
        "## Passed checks",
        "",
        *_section(outcome, "PASS"),
        "",
        "## Warnings",
        "",
        *_section(outcome, "WARNING"),
        "",
        "## Failed checks",
        "",
        *_section(outcome, "FAIL"),
        "",
        "## Not-run checks",
        "",
        *_section(outcome, "NOT RUN"),
        "",
        "## Interpretation boundary",
        "",
        "- Final-horizon accuracy is unknown because competition test targets are unavailable.",
        "- Legacy error-segmentation, specialist-routing, and interval reports are historical evidence, not diagnostics of the current recursive selected model.",
        "- This validator never retrains, tunes, forecasts, or changes artifacts.",
        "",
        "## Command to reproduce",
        "",
        "```powershell",
        "python -m src.validate_ds_project",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    del argv
    outcome = run_validation()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(outcome), encoding="utf-8")
    counts = {
        status: sum(item.status == status for item in outcome.results)
        for status in VALID_STATUSES
    }
    print(
        "DS validation: "
        f"{counts['PASS']} PASS, {counts['WARNING']} WARNING, "
        f"{counts['FAIL']} FAIL, {counts['NOT RUN']} NOT RUN"
    )
    print(REPORT_PATH.relative_to(PROJECT_ROOT).as_posix())
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
