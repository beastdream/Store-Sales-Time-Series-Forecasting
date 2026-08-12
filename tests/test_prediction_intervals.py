import numpy as np
import pandas as pd
import pytest

from src.modeling.uncertainty import (
    NOMINAL_COVERAGE,
    build_prediction_intervals,
    conformal_log_radius,
    pinball_loss,
    score_interval_segments,
)


def test_conformal_radius_uses_absolute_log_residuals_and_is_future_invariant() -> None:
    actual = np.array([0.0, 1.0, 3.0, 7.0, 15.0])
    prediction = np.array([0.0, 1.0, 1.0, 3.0, 7.0])
    radius = conformal_log_radius(actual, prediction, alpha=0.2)
    changed_future = np.array([999_999.0, 888_888.0])

    assert radius == pytest.approx(np.max(np.abs(np.log1p(actual) - np.log1p(prediction))))
    assert conformal_log_radius(actual, prediction, alpha=0.2) == radius
    assert changed_future.sum() > 0  # future values are never accepted by the calibration API


def test_intervals_keep_point_as_p50_and_are_monotonic_nonnegative() -> None:
    points = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3),
            "store_nbr": 1,
            "family": "A",
            "prediction": [0.0, 4.0, 10.0],
        }
    )
    result = build_prediction_intervals(points, np.log(2.0))

    assert result["p50"].tolist() == points["prediction"].tolist()
    assert result["p10"].ge(0).all()
    assert result["p10"].le(result["p50"]).all()
    assert result["p50"].le(result["p90"]).all()


def test_pinball_loss_known_values() -> None:
    actual = [0.0, 10.0]
    prediction = [2.0, 8.0]

    assert pinball_loss(actual, prediction, 0.5) == pytest.approx(1.0)
    assert pinball_loss(actual, prediction, 0.1) == pytest.approx(1.0)
    assert pinball_loss(actual, prediction, 0.9) == pytest.approx(1.0)


def test_segment_scoring_reports_coverage_width_pinball_and_point_accuracy() -> None:
    frame = pd.DataFrame(
        {
            "segment": ["A", "A", "B", "B"],
            "fold": [1, 2, 1, 2],
            "store_nbr": [1, 1, 2, 2],
            "family": ["X", "X", "Y", "Y"],
            "sales": [1.0, 5.0, 2.0, 10.0],
            "p10": [0.0, 4.0, 0.0, 5.0],
            "p50": [1.0, 6.0, 3.0, 8.0],
            "p90": [2.0, 8.0, 4.0, 9.0],
        }
    )
    scores = score_interval_segments(frame, ["segment"]).set_index("segment")

    assert scores.loc["A", "empirical_coverage"] == 1.0
    assert scores.loc["B", "empirical_coverage"] == 0.5
    assert scores.loc["A", "mean_interval_width"] == 3.0
    assert scores.loc["A", "fold_count"] == 2
    assert scores.loc["A", "point_rmsle"] >= 0
    assert NOMINAL_COVERAGE == 0.8
