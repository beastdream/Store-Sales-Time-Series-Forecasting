from pathlib import Path

import pandas as pd

from src.modeling.baselines import BASELINE_MODELS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORES_PATH = PROJECT_ROOT / "reports" / "modeling" / "baseline_scores.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "modeling" / "baseline_summary.csv"


def test_baseline_scores_cover_every_model_and_fold() -> None:
    scores = pd.read_csv(SCORES_PATH, parse_dates=["train_end", "validation_start", "validation_end"])

    assert list(scores.columns) == [
        "model",
        "fold",
        "train_end",
        "validation_start",
        "validation_end",
        "rmsle",
        "mae",
        "wape",
    ]
    assert len(scores) == len(BASELINE_MODELS) * 4
    assert set(scores["model"]) == set(BASELINE_MODELS)
    assert scores.groupby("model")["fold"].nunique().eq(4).all()
    assert (scores["validation_start"] == scores["train_end"] + pd.Timedelta(days=1)).all()
    assert (scores["validation_end"] - scores["validation_start"]).dt.days.eq(15).all()
    assert scores[["rmsle", "mae", "wape"]].notna().all().all()
    assert scores[["rmsle", "mae", "wape"]].ge(0).all().all()


def test_baseline_summary_contains_mean_and_std_for_four_folds() -> None:
    summary = pd.read_csv(SUMMARY_PATH)

    assert list(summary.columns) == [
        "model",
        "fold_count",
        "rmsle_mean",
        "rmsle_std",
        "mae_mean",
        "mae_std",
        "wape_mean",
        "wape_std",
    ]
    assert len(summary) == len(BASELINE_MODELS)
    assert set(summary["model"]) == set(BASELINE_MODELS)
    assert summary["fold_count"].eq(4).all()
    metric_columns = [column for column in summary if column.endswith(("_mean", "_std"))]
    assert summary[metric_columns].notna().all().all()
    assert summary[metric_columns].ge(0).all().all()
