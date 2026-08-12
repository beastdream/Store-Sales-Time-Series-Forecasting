from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modeling.intermittent import (
    ROUTING_MINIMUM_RMSLE_IMPROVEMENT,
    summarize_intermittent_scores,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "modeling"


def test_intermittent_score_matrix_is_complete_and_uses_four_folds() -> None:
    scores = pd.read_csv(REPORT_DIR / "intermittent_model_scores.csv")
    expected_models = {
        "global_lightgbm_tuned",
        "croston",
        "sba",
        "tsb",
        "two_stage_lightgbm",
    }

    assert len(scores) == 20
    assert set(scores["model"]) == expected_models
    assert scores.groupby("model")["fold"].nunique().eq(4).all()
    assert scores[["rmsle", "mae", "wape"]].notna().all().all()
    assert scores[["rmsle", "mae", "wape"]].ge(0).all().all()


def test_global_control_matches_existing_tuned_oof_on_intermittent_cohort() -> None:
    scores = pd.read_csv(REPORT_DIR / "intermittent_model_scores.csv")
    global_scores = scores.loc[scores["model"].eq("global_lightgbm_tuned")].set_index("fold")
    oof = pd.read_parquet(REPORT_DIR / "global_lgbm_tuned_oof_predictions.parquet")
    readiness = pd.read_csv(
        PROJECT_ROOT / "reports" / "tables" / "forecast_readiness.csv",
        usecols=["store_nbr", "family", "readiness_class"],
    )
    keys = readiness.loc[
        readiness["readiness_class"].eq("Intermittent demand"),
        ["store_nbr", "family"],
    ]
    cohort = oof.merge(keys, on=["store_nbr", "family"], validate="many_to_one")
    cohort["squared_log_error"] = (
        np.log1p(cohort["prediction"]) - np.log1p(cohort["sales"])
    ) ** 2

    for fold, frame in cohort.groupby("fold"):
        assert global_scores.loc[fold, "rmsle"] == pytest.approx(
            frame["squared_log_error"].mean() ** 0.5
        )


def test_two_stage_oof_is_complete_and_routing_decision_uses_mean_improvement() -> None:
    scores = pd.read_csv(REPORT_DIR / "intermittent_model_scores.csv")
    summary = summarize_intermittent_scores(scores).set_index("model")
    oof = pd.read_parquet(REPORT_DIR / "two_stage_intermittent_oof_predictions.parquet")

    assert len(oof) == 417 * 16 * 4
    assert not oof.duplicated(["fold", "date", "store_nbr", "family"]).any()
    assert oof.groupby("fold").size().eq(417 * 16).all()
    assert oof["prediction"].ge(0).all()
    assert summary.loc["two_stage_lightgbm", "routing_eligible"]
    assert (
        summary.loc["two_stage_lightgbm", "rmsle_improvement_vs_global"]
        >= ROUTING_MINIMUM_RMSLE_IMPROVEMENT
    )
    assert not summary.loc[["croston", "sba", "tsb"], "routing_eligible"].any()


def test_readiness_is_not_a_two_stage_training_feature_and_report_is_cautious() -> None:
    source = (PROJECT_ROOT / "notebooks" / "18_intermittent_demand_models.py").read_text(
        encoding="utf-8"
    )
    reproduction = source[
        source.index("def reproduce_two_stage_oof") : source.index("def score_two_stage")
    ]
    report = (REPORT_DIR / "model_routing_analysis.md").read_text(encoding="utf-8")

    assert "readiness" not in reproduction
    assert "load_test" not in source
    assert "2 of 4 folds" in report
    assert "origin-causal cohort rule" in report
    assert "not a training feature" in report
