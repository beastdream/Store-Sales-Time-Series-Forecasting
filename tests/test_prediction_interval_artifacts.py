from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modeling.uncertainty import NOMINAL_COVERAGE, score_interval_segments


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "modeling"


def test_interval_oof_is_complete_monotonic_and_preserves_point_forecasts() -> None:
    intervals = pd.read_parquet(REPORT_DIR / "global_lgbm_prediction_intervals.parquet")
    point = pd.read_parquet(REPORT_DIR / "global_lgbm_tuned_oof_predictions.parquet")

    assert len(intervals) == len(point) == 4 * 16 * 54 * 33
    grain = ["fold", "date", "store_nbr", "family"]
    assert not intervals.duplicated(grain).any()
    assert intervals["p10"].ge(0).all()
    assert intervals["p10"].le(intervals["p50"]).all()
    assert intervals["p50"].le(intervals["p90"]).all()
    comparison = intervals[grain + ["p50"]].merge(
        point[grain + ["prediction"]], on=grain, validate="one_to_one"
    )
    np.testing.assert_array_equal(comparison["p50"], comparison["prediction"])


def test_each_calibration_window_strictly_precedes_its_validation_fold() -> None:
    intervals = pd.read_parquet(REPORT_DIR / "global_lgbm_prediction_intervals.parquet")
    calibration = intervals.groupby("fold").agg(
        calibration_origin=("calibration_origin", "first"),
        calibration_start=("calibration_start", "first"),
        calibration_end=("calibration_end", "first"),
        validation_start=("date", "min"),
        validation_end=("date", "max"),
        radius=("calibration_log_radius", "first"),
    )

    assert (calibration["calibration_origin"] < calibration["calibration_start"]).all()
    assert (calibration["calibration_end"] < calibration["validation_start"]).all()
    assert (calibration["calibration_start"] + pd.Timedelta(days=15) == calibration["calibration_end"]).all()
    assert (calibration["validation_start"] + pd.Timedelta(days=15) == calibration["validation_end"]).all()
    assert calibration["radius"].gt(0).all()


def test_interval_scores_reconcile_with_interval_predictions() -> None:
    intervals = pd.read_parquet(REPORT_DIR / "global_lgbm_prediction_intervals.parquet")
    scores = pd.read_csv(REPORT_DIR / "prediction_interval_scores.csv")
    overall = scores.loc[scores["segment_type"].eq("overall")].iloc[0]
    recomputed = score_interval_segments(
        intervals.assign(segment_type="overall", segment_value="all"),
        ["segment_type", "segment_value"],
    ).iloc[0]

    for metric in [
        "empirical_coverage",
        "mean_interval_width",
        "p10_pinball_loss",
        "p50_pinball_loss",
        "p90_pinball_loss",
        "mean_pinball_loss",
        "point_rmsle",
        "point_mae",
        "point_wape",
    ]:
        assert overall[metric] == pytest.approx(recomputed[metric])
    assert overall["empirical_coverage"] == pytest.approx(NOMINAL_COVERAGE, abs=0.01)
    assert len(scores.loc[scores["segment_type"].eq("overall_fold")]) == 4
    assert len(scores.loc[scores["segment_type"].eq("readiness_class")]) == 6
    assert set(scores.loc[scores["segment_type"].eq("risk_cohort"), "segment_value"]) == {
        "high_volatility",
        "intermittent",
    }


def test_report_distinguishes_point_accuracy_from_interval_calibration() -> None:
    report = (REPORT_DIR / "prediction_intervals.md").read_text(encoding="utf-8")
    source = (PROJECT_ROOT / "notebooks" / "19_prediction_intervals.py").read_text(
        encoding="utf-8"
    )
    reproduction = source[
        source.index("def reproduce_temporal_calibration") : source.index("def build_segment_scores")
    ]

    assert "## Point accuracy versus uncertainty calibration" in report
    assert "P50 is the unchanged" in report
    assert "Intermittent demand" in report
    assert "High-volatility and intermittent cohorts" in report
    assert "forecast_readiness.csv" not in reproduction
    assert "load_test" not in source
