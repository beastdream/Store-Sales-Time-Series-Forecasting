"""Contracts for recursive feature-ablation definitions and artifacts."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.modeling.ablation import (
    EXPERIMENT_FEATURES,
    HOLIDAY_FEATURES,
    summarize_ablation,
)
from src.modeling.train_global import FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "modeling"
EXPECTED_EXPERIMENTS = [
    "M0", "M1", "M2", "M3", "M4", "M5", "M6", "M6_NO_HOLIDAY"
]


def test_m6_no_holiday_is_exact_full_model_subtraction() -> None:
    expected = [feature for feature in FEATURE_COLUMNS if feature not in HOLIDAY_FEATURES]
    assert EXPERIMENT_FEATURES["M6"] == FEATURE_COLUMNS
    assert EXPERIMENT_FEATURES["M6_NO_HOLIDAY"] == expected
    assert not set(HOLIDAY_FEATURES) & set(EXPERIMENT_FEATURES["M6_NO_HOLIDAY"])


def test_summary_compares_no_holiday_directly_with_m6() -> None:
    means = {
        "M0": 0.50,
        "M1": 0.45,
        "M2": 0.44,
        "M3": 0.43,
        "M4": 0.42,
        "M5": 0.4215,
        "M6": 0.41,
        "M6_NO_HOLIDAY": 0.4095,
    }
    rows = []
    for experiment, mean in means.items():
        for fold in range(1, 5):
            rows.append(
                {
                    "experiment": experiment,
                    "model": "model",
                    "added_group": "group",
                    "fold": fold,
                    "rmsle": mean,
                    "mae": 1.0,
                    "wape": 1.0,
                }
            )
    summary = summarize_ablation(pd.DataFrame(rows)).set_index("experiment")
    assert summary.loc["M5", "effect"] == "degraded"
    assert summary.loc["M6_NO_HOLIDAY", "comparison_experiment"] == "M6"
    assert summary.loc["M6_NO_HOLIDAY", "effect"] == "negligible effect"


def test_recursive_ablation_artifacts_are_complete_and_causal() -> None:
    scores = pd.read_csv(
        REPORT_DIR / "ablation_scores.csv",
        parse_dates=["train_end", "validation_start", "validation_end"],
    )
    assert scores.columns.tolist() == [
        "experiment", "model", "added_group", "inference_strategy", "fold",
        "train_end", "validation_start", "validation_end", "rmsle", "mae", "wape",
    ]
    assert scores["experiment"].drop_duplicates().tolist() == EXPECTED_EXPERIMENTS
    assert len(scores) == 32
    assert not scores.duplicated(["experiment", "fold"]).any()
    assert scores.groupby("experiment").size().eq(4).all()
    assert scores.loc[scores["experiment"].ne("M0"), "inference_strategy"].eq(
        "recursive_untuned"
    ).all()
    assert scores[["rmsle", "mae", "wape"]].notna().all().all()
    assert np.isfinite(scores[["rmsle", "mae", "wape"]]).all().all()
    assert (scores["validation_start"] == scores["train_end"] + pd.Timedelta(days=1)).all()
    assert (scores["validation_end"] - scores["validation_start"]).dt.days.eq(15).all()
    expected_folds = {
        1: ("2017-06-12", "2017-06-13", "2017-06-28"),
        2: ("2017-06-28", "2017-06-29", "2017-07-14"),
        3: ("2017-07-14", "2017-07-15", "2017-07-30"),
        4: ("2017-07-30", "2017-07-31", "2017-08-15"),
    }
    actual_folds = {
        fold: tuple(value.date().isoformat() for value in row)
        for fold, row in scores.groupby("fold")[[
            "train_end", "validation_start", "validation_end"
        ]].first().iterrows()
    }
    assert actual_folds == expected_folds

    recursive = pd.read_csv(REPORT_DIR / "recursive_backtest_scores.csv")
    m6 = scores.loc[scores["experiment"].eq("M6")].sort_values("fold")
    recursive = recursive.sort_values("fold")
    assert np.allclose(
        m6[["rmsle", "mae", "wape"]],
        recursive[["rmsle", "mae", "wape"]],
    )


def test_recursive_ablation_report_and_runner_exclude_oil_tuning_and_test() -> None:
    report = (REPORT_DIR / "ablation_summary.md").read_text(encoding="utf-8")
    source = (PROJECT_ROOT / "notebooks" / "15_feature_ablation.py").read_text(
        encoding="utf-8"
    )
    assert "M6_NO_HOLIDAY" in report
    assert "Oil features (M7): not run" in report
    assert "Recommended feature set" in report
    assert "recursive_forecast(" in source
    assert "DEFAULT_PARAMETERS" in source
    assert "src.modeling.tuning" not in source
    assert "load_test" not in source
    assert "final_submission" not in source
