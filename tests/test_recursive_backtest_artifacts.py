"""Contracts for the separate recursive base-model evaluation artifacts."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "modeling"


def test_recursive_scores_and_oof_are_complete() -> None:
    scores = pd.read_csv(
        REPORT_DIR / "recursive_backtest_scores.csv",
        parse_dates=["train_end", "validation_start", "validation_end"],
    )
    oof = pd.read_parquet(REPORT_DIR / "recursive_global_lgbm_oof_predictions.parquet")

    assert scores["fold"].tolist() == [1, 2, 3, 4]
    assert scores["strategy"].eq("recursive_global_lightgbm_untuned").all()
    assert scores[["rmsle", "mae", "wape"]].notna().all().all()
    assert (scores["validation_start"] == scores["train_end"] + pd.Timedelta(days=1)).all()
    assert (scores["validation_end"] - scores["validation_start"]).dt.days.eq(15).all()
    assert oof.columns.tolist() == [
        "fold", "date", "store_nbr", "family", "actual", "prediction"
    ]
    assert len(oof) == 4 * 16 * 1_782 == 114_048
    assert not oof.duplicated(["fold", "date", "store_nbr", "family"]).any()
    assert oof.groupby("fold").size().eq(16 * 1_782).all()
    assert oof[["actual", "prediction"]].notna().all().all()
    assert np.isfinite(oof["prediction"]).all()
    assert oof["prediction"].ge(0).all()


def test_recursive_comparison_preserves_previous_evidence_and_no_tuning() -> None:
    report = (REPORT_DIR / "recursive_vs_previous_strategy.md").read_text(
        encoding="utf-8"
    )
    source = (
        PROJECT_ROOT / "notebooks" / "21_recursive_global_backtest.py"
    ).read_text(encoding="utf-8")

    assert "old_strategy_rmsle" in report
    assert "new_recursive_rmsle" in report
    assert "Fold 4" in report
    assert "DEFAULT_PARAMETERS" in source
    assert "recursive_forecast(" in source
    assert "src.modeling.tuning" not in source
    assert "SEARCH_CONFIGS" not in source
    assert "global_lightgbm_chosen_config" not in source
    assert "final_submission.csv" not in source
