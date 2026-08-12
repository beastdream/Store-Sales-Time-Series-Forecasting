"""Temporal residual calibration for forecast prediction intervals."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.modeling.metrics import mae, rmsle, wape


INTERVAL_ALPHA = 0.20
NOMINAL_COVERAGE = 1.0 - INTERVAL_ALPHA
INTERVAL_COLUMNS = ["p10", "p50", "p90"]


def conformal_log_radius(
    actual: Sequence[float],
    prediction: Sequence[float],
    *,
    alpha: float = INTERVAL_ALPHA,
) -> float:
    """Return a finite-sample split-conformal radius on the log1p scale."""
    y = np.asarray(actual, dtype="float64")
    point = np.asarray(prediction, dtype="float64")
    if y.ndim != 1 or point.ndim != 1 or y.shape != point.shape or y.size == 0:
        raise ValueError("actual and prediction must be matching non-empty vectors")
    if not np.isfinite(y).all() or not np.isfinite(point).all() or (y < 0).any() or (point < 0).any():
        raise ValueError("actual and prediction must be finite and nonnegative")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    scores = np.abs(np.log1p(y) - np.log1p(point))
    level = min(np.ceil((scores.size + 1) * (1.0 - alpha)) / scores.size, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def build_prediction_intervals(
    point_predictions: pd.DataFrame,
    log_radius: float,
) -> pd.DataFrame:
    """Keep the validated point forecast as P50 and add nonnegative P10/P90 bounds."""
    required = ["date", "store_nbr", "family", "prediction"]
    missing = [column for column in required if column not in point_predictions]
    if missing:
        raise KeyError("point predictions are missing columns: " + ", ".join(missing))
    radius = float(log_radius)
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("log_radius must be finite and nonnegative")
    result = point_predictions.copy()
    log_point = np.log1p(result["prediction"].to_numpy(dtype="float64"))
    result["p10"] = np.clip(np.expm1(log_point - radius), 0.0, None)
    result["p50"] = result["prediction"].to_numpy(dtype="float64")
    result["p90"] = np.clip(np.expm1(log_point + radius), 0.0, None)
    result["calibration_log_radius"] = radius
    if not (result["p10"].le(result["p50"]) & result["p50"].le(result["p90"])).all():
        raise RuntimeError("prediction interval quantiles are not monotonic")
    return result.drop(columns="prediction")


def pinball_loss(actual: Sequence[float], prediction: Sequence[float], quantile: float) -> float:
    """Return mean quantile (pinball) loss."""
    y = np.asarray(actual, dtype="float64")
    pred = np.asarray(prediction, dtype="float64")
    if y.shape != pred.shape or y.size == 0 or y.ndim != 1:
        raise ValueError("actual and prediction must be matching non-empty vectors")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")
    error = y - pred
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def score_interval_segments(
    intervals: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """Score point accuracy and uncertainty calibration by post-hoc segment."""
    groups = list(group_columns)
    required = ["sales", *INTERVAL_COLUMNS, "fold", "store_nbr", "family", *groups]
    missing = [column for column in required if column not in intervals]
    if missing:
        raise KeyError("interval frame is missing columns: " + ", ".join(missing))
    if intervals.empty or intervals[["sales", *INTERVAL_COLUMNS]].isna().any().any():
        raise ValueError("interval frame must be non-empty and complete")

    rows: list[dict[str, object]] = []
    grouped = intervals.groupby(groups, observed=True, dropna=False, sort=False)
    for keys, frame in grouped:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        actual = frame["sales"].to_numpy(dtype="float64")
        p10 = frame["p10"].to_numpy(dtype="float64")
        p50 = frame["p50"].to_numpy(dtype="float64")
        p90 = frame["p90"].to_numpy(dtype="float64")
        covered = (actual >= p10) & (actual <= p90)
        row = dict(zip(groups, key_values))
        row.update(
            {
                "observation_count": len(frame),
                "series_count": frame[["store_nbr", "family"]].drop_duplicates().shape[0],
                "fold_count": frame["fold"].nunique(),
                "empirical_coverage": float(covered.mean()),
                "coverage_gap_vs_nominal": float(covered.mean() - NOMINAL_COVERAGE),
                "lower_tail_rate": float((actual < p10).mean()),
                "upper_tail_rate": float((actual > p90).mean()),
                "mean_interval_width": float(np.mean(p90 - p10)),
                "median_interval_width": float(np.median(p90 - p10)),
                "p10_pinball_loss": pinball_loss(actual, p10, 0.10),
                "p50_pinball_loss": pinball_loss(actual, p50, 0.50),
                "p90_pinball_loss": pinball_loss(actual, p90, 0.90),
                "mean_pinball_loss": float(
                    np.mean(
                        [
                            pinball_loss(actual, p10, 0.10),
                            pinball_loss(actual, p50, 0.50),
                            pinball_loss(actual, p90, 0.90),
                        ]
                    )
                ),
                "point_rmsle": rmsle(actual, p50),
                "point_mae": mae(actual, p50),
                "point_wape": wape(actual, p50),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
