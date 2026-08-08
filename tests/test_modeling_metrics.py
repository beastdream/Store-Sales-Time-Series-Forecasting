"""Known-answer and edge-case tests for forecast metrics."""

import numpy as np
import pytest

from src.modeling.metrics import mae, rmsle, wape


def test_rmsle_matches_simple_log1p_example() -> None:
    result = rmsle([0.0, 3.0], [0.0, 1.0])

    assert result == pytest.approx(np.log(2.0) / np.sqrt(2.0))


def test_rmsle_clips_negative_predictions_to_zero() -> None:
    with_negative = rmsle([1.0, 4.0], [-3.0, 4.0])
    with_zero = rmsle([1.0, 4.0], [0.0, 4.0])

    assert with_negative == pytest.approx(with_zero)


def test_rmsle_rejects_negative_actual_targets() -> None:
    with pytest.raises(ValueError, match="nonnegative y_true"):
        rmsle([-1.0, 2.0], [0.0, 2.0])


def test_mae_matches_known_answer() -> None:
    assert mae([1.0, 2.0, 3.0], [2.0, 2.0, 1.0]) == pytest.approx(1.0)


def test_wape_matches_known_answer() -> None:
    assert wape([1.0, 2.0, 3.0], [2.0, 2.0, 1.0]) == pytest.approx(0.5)


def test_wape_returns_nan_for_zero_actual_denominator() -> None:
    assert np.isnan(wape([0.0, 0.0], [0.0, 1.0]))


@pytest.mark.parametrize(
    ("y_true", "y_pred", "message"),
    [
        ([], [], "must not be empty"),
        ([1.0], [1.0, 2.0], "same shape"),
        ([[1.0]], [[1.0]], "one-dimensional"),
        ([1.0, np.nan], [1.0, 2.0], "finite values"),
    ],
)
def test_metrics_reject_invalid_inputs(
    y_true: object,
    y_pred: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        mae(y_true, y_pred)
