"""Forecasting strategies for post-hoc intermittent-demand evaluation."""

from collections.abc import Iterable
from copy import deepcopy

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.modeling.train_global import CATEGORICAL_FEATURES, feature_matrix


SERIES_COLUMNS = ["store_nbr", "family"]
GRAIN = ["date", *SERIES_COLUMNS]
INTERMITTENT_SMOOTHING = 0.1
ROUTING_MINIMUM_RMSLE_IMPROVEMENT = 0.001


def _validate_smoothing(value: float, name: str) -> float:
    smoothing = float(value)
    if not 0 < smoothing <= 1:
        raise ValueError(f"{name} must be in (0, 1]")
    return smoothing


def _croston_level(values: np.ndarray, alpha: float, method: str, beta: float) -> float:
    positive_positions = np.flatnonzero(values > 0)
    if positive_positions.size == 0:
        return 0.0
    first = int(positive_positions[0])
    size = float(values[first])
    interval = float(first + 1)
    probability = 1.0 / interval
    elapsed = 0
    for value in values[first + 1 :]:
        elapsed += 1
        occurrence = float(value > 0)
        if method == "tsb":
            probability += beta * (occurrence - probability)
        if occurrence:
            size += alpha * (float(value) - size)
            if method != "tsb":
                interval += alpha * (elapsed - interval)
            elapsed = 0
    if method == "croston":
        return size / interval
    if method == "sba":
        return (1.0 - alpha / 2.0) * size / interval
    return probability * size


def forecast_intermittent_baseline(
    history: pd.DataFrame,
    forecast_dates: Iterable[object],
    cutoff: object,
    *,
    method: str,
    series: pd.DataFrame,
    alpha: float = INTERMITTENT_SMOOTHING,
    beta: float = INTERMITTENT_SMOOTHING,
) -> pd.DataFrame:
    """Forecast Croston, SBA, or TSB using only observations through cutoff."""
    if method not in {"croston", "sba", "tsb"}:
        raise ValueError("method must be one of: croston, sba, tsb")
    alpha_value = _validate_smoothing(alpha, "alpha")
    beta_value = _validate_smoothing(beta, "beta")
    required = [*GRAIN, "sales"]
    missing = [column for column in required if column not in history]
    if missing:
        raise KeyError("history is missing columns: " + ", ".join(missing))
    if series.duplicated(SERIES_COLUMNS).any() or series.empty:
        raise ValueError("series must be non-empty and unique by store-family")

    cutoff_date = pd.Timestamp(cutoff).normalize()
    dates = pd.DatetimeIndex(pd.to_datetime(list(forecast_dates))).normalize().unique().sort_values()
    if dates.empty or dates.min() <= cutoff_date:
        raise ValueError("forecast dates must be non-empty and after cutoff")
    prepared = history[required].copy()
    prepared["date"] = pd.to_datetime(prepared["date"]).dt.normalize()
    prepared = prepared.loc[prepared["date"].le(cutoff_date)]
    if prepared.duplicated(GRAIN).any():
        raise ValueError("history contains duplicate date-store-family rows")
    if prepared["sales"].isna().any() or prepared["sales"].lt(0).any():
        raise ValueError("historical sales must be complete and nonnegative")

    levels = []
    for keys, group in prepared.sort_values("date").groupby(SERIES_COLUMNS, observed=True):
        values = group["sales"].to_numpy(dtype="float64")
        levels.append(
            {
                "store_nbr": keys[0],
                "family": keys[1],
                "prediction": max(_croston_level(values, alpha_value, method, beta_value), 0.0),
            }
        )
    grid = series[SERIES_COLUMNS].merge(
        pd.DataFrame({"date": dates}), how="cross", validate="many_to_many"
    )
    result = grid.merge(pd.DataFrame(levels), on=SERIES_COLUMNS, how="left", validate="many_to_one")
    result["prediction"] = result["prediction"].fillna(0.0)
    return result[["date", *SERIES_COLUMNS, "prediction"]].sort_values(
        [*SERIES_COLUMNS, "date"], kind="stable"
    ).reset_index(drop=True)


def train_two_stage_models(
    feature_frame: pd.DataFrame,
    training_cutoff: object,
    *,
    parameters: dict[str, object],
    num_boost_round: int,
    feature_columns: list[str],
) -> tuple[lgb.Booster, lgb.Booster]:
    """Train global occurrence and positive-magnitude models without readiness labels."""
    cutoff = pd.Timestamp(training_cutoff).normalize()
    training = feature_frame.loc[feature_frame["date"].le(cutoff)].copy()
    if training.empty or training["sales"].isna().any() or training["sales"].lt(0).any():
        raise ValueError("two-stage training sales must be complete and nonnegative")
    categorical = [column for column in CATEGORICAL_FEATURES if column in feature_columns]

    occurrence_parameters = deepcopy(parameters)
    occurrence_parameters.update({"objective": "binary", "metric": "binary_logloss"})
    occurrence_data = lgb.Dataset(
        feature_matrix(training, feature_columns),
        label=training["sales"].gt(0).astype("uint8"),
        categorical_feature=categorical,
        free_raw_data=True,
    )
    occurrence_model = lgb.train(
        occurrence_parameters, occurrence_data, num_boost_round=num_boost_round
    )

    positive = training.loc[training["sales"].gt(0)]
    magnitude_parameters = deepcopy(parameters)
    magnitude_parameters.update({"objective": "regression", "metric": "rmse"})
    magnitude_data = lgb.Dataset(
        feature_matrix(positive, feature_columns),
        label=np.log1p(positive["sales"].to_numpy(dtype="float64")),
        categorical_feature=categorical,
        free_raw_data=True,
    )
    magnitude_model = lgb.train(
        magnitude_parameters, magnitude_data, num_boost_round=num_boost_round
    )
    return occurrence_model, magnitude_model


def predict_two_stage(
    occurrence_model: lgb.Booster,
    magnitude_model: lgb.Booster,
    feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return P(positive) multiplied by expected positive-sales magnitude."""
    occurrence = np.clip(
        occurrence_model.predict(feature_matrix(feature_frame, occurrence_model.feature_name())),
        0.0,
        1.0,
    )
    magnitude = np.clip(
        np.expm1(magnitude_model.predict(feature_matrix(feature_frame, magnitude_model.feature_name()))),
        0.0,
        None,
    )
    prediction = np.asarray(occurrence * magnitude, dtype="float64")
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise RuntimeError("two-stage prediction must be finite and nonnegative")
    result = feature_frame[GRAIN].copy()
    result["prediction"] = prediction
    return result


def summarize_intermittent_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize all four folds and apply the routing improvement threshold."""
    if not scores.groupby("model")["fold"].nunique().eq(4).all():
        raise ValueError("every intermittent strategy must cover four folds")
    summary = scores.groupby("model", as_index=False, sort=False).agg(
        fold_count=("fold", "nunique"),
        rmsle_mean=("rmsle", "mean"),
        rmsle_std=("rmsle", "std"),
        mae_mean=("mae", "mean"),
        wape_mean=("wape", "mean"),
    )
    global_rmsle = float(
        summary.loc[summary["model"].eq("global_lightgbm_tuned"), "rmsle_mean"].iloc[0]
    )
    summary["rmsle_improvement_vs_global"] = global_rmsle - summary["rmsle_mean"]
    summary["routing_eligible"] = (
        ~summary["model"].eq("global_lightgbm_tuned")
        & summary["rmsle_improvement_vs_global"].ge(ROUTING_MINIMUM_RMSLE_IMPROVEMENT)
    )
    return summary.sort_values("rmsle_mean", kind="stable").reset_index(drop=True)
