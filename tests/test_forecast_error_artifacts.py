from pathlib import Path

import pandas as pd
import pytest

from src.modeling.error_analysis import score_segments, validate_oof_predictions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "modeling"


def test_legacy_oof_artifact_matches_its_legacy_tuning_evidence() -> None:
    oof = pd.read_parquet(REPORT_DIR / "global_lgbm_tuned_oof_predictions.parquet")
    validate_oof_predictions(oof)

    assert len(oof) == 4 * 16 * 54 * 33
    assert oof.groupby("fold").size().eq(16 * 54 * 33).all()
    assert oof["prediction"].ge(0).all()
    fold_scores = score_segments(oof, ["fold"]).set_index("fold")
    tuning = pd.read_csv(
        REPORT_DIR / "tuning_fold_scores_legacy_fixed_strategy.csv"
    )
    tuning = tuning.loc[tuning["experiment"].eq("T2_moderate_capacity")].set_index("fold")
    for fold in range(1, 5):
        assert fold_scores.loc[fold, "rmsle"] == pytest.approx(tuning.loc[fold, "rmsle"])
        assert fold_scores.loc[fold, "mae"] == pytest.approx(tuning.loc[fold, "mae"])
        assert fold_scores.loc[fold, "wape"] == pytest.approx(tuning.loc[fold, "wape"])


def test_required_segment_artifacts_have_complete_metrics_and_expected_grains() -> None:
    stores = pd.read_csv(REPORT_DIR / "scores_by_store.csv")
    families = pd.read_csv(REPORT_DIR / "scores_by_family.csv")
    readiness = pd.read_csv(REPORT_DIR / "scores_by_readiness.csv")

    assert len(stores) == 54
    assert stores["store_nbr"].is_unique
    assert len(families) == 33
    assert families["family"].is_unique
    assert len(readiness) == 6
    assert readiness["readiness_class"].is_unique
    for frame in [stores, families, readiness]:
        assert {"observation_count", "series_count", "fold_count", "rmsle", "mae", "wape"}.issubset(frame)
        assert frame["fold_count"].eq(4).all()
        assert frame[["rmsle", "mae", "wape"]].notna().all().all()


def test_report_separates_evidence_from_speculation_and_readiness_is_posthoc() -> None:
    report = (REPORT_DIR / "error_analysis.md").read_text(encoding="utf-8")
    source = (PROJECT_ROOT / "notebooks" / "17_forecast_error_analysis.py").read_text(encoding="utf-8")

    assert "## Evidence" in report
    assert "## Speculation and hypotheses to test" in report
    assert "## Specialized-model recommendation" in report
    assert "Insufficient history** is not an observed failure" in report
    assert "post-hoc label" in report
    assert "ForecastReadiness" not in source[
        source.index("def reproduce_oof_predictions") : source.index("def markdown_table")
    ]


def test_error_analysis_plots_exist() -> None:
    figure_dir = PROJECT_ROOT / "reports" / "figures" / "modeling"
    for filename in ["oof_family_rmsle.png", "oof_readiness_rmsle.png"]:
        assert (figure_dir / filename).stat().st_size > 0
