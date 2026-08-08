"""Forecast error metrics with explicit input and edge-case handling."""

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt


ArrayLike = Iterable[float] | npt.NDArray[np.number]


def _validated_arrays(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return matching, finite, non-empty one-dimensional float arrays."""
    actual = np.asarray(y_true, dtype="float64")
    predicted = np.asarray(y_pred, dtype="float64")
    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional")
    if actual.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("y_true and y_pred must contain only finite values")
    return actual, predicted


def rmsle(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return root mean squared logarithmic error using ``log1p``.

    Actual Sales Volume must be nonnegative. Negative predictions are clipped to
    zero before applying ``log1p`` because the forecast target is constrained to be
    nonnegative; this policy is explicit and deterministic rather than allowing
    invalid logarithms or silently changing actual targets.
    """
    actual, predicted = _validated_arrays(y_true, y_pred)
    if (actual < 0).any():
        raise ValueError("RMSLE requires nonnegative y_true values")
    safe_prediction = np.clip(predicted, a_min=0.0, a_max=None)
    log_error = np.log1p(safe_prediction) - np.log1p(actual)
    return float(np.sqrt(np.mean(np.square(log_error))))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return mean absolute error."""
    actual, predicted = _validated_arrays(y_true, y_pred)
    return float(np.mean(np.abs(actual - predicted)))


def wape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return weighted absolute percentage error.

    WAPE is ``sum(abs(y_true - y_pred)) / sum(abs(y_true))``. If the actual-volume
    denominator is zero, WAPE is undefined and this function returns ``NaN`` rather
    than dividing by zero or reporting a misleading perfect score.
    """
    actual, predicted = _validated_arrays(y_true, y_pred)
    denominator = float(np.abs(actual).sum())
    if denominator == 0.0:
        return float("nan")
    numerator = float(np.abs(actual - predicted).sum())
    return numerator / denominator
