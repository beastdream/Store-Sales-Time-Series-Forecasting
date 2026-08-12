"""Training infrastructure for the first global LightGBM sales model."""

from copy import deepcopy

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.calendar_features import build_calendar_features
from src.features.exogenous_features import add_holiday_features, add_promotion_features, add_store_metadata
from src.features.lag_features import DEFAULT_LAGS, add_sales_lag_features
from src.features.rolling_features import ROLLING_FEATURE_COLUMNS, add_sales_rolling_features


MODEL_NAME = "global_lightgbm"
CATEGORICAL_FEATURES = ["store_nbr", "family", "store_type", "cluster", "city", "state"]
FEATURE_COLUMNS = [
    *CATEGORICAL_FEATURES,
    "day_of_week", "week_of_year", "month", "quarter", "year",
    "is_weekend", "is_month_start", "is_month_end", "is_payday",
    "onpromotion", "promotion_active", "holiday_count", "is_holiday",
    "is_work_day", "is_event",
    *[f"sales_lag_{lag}" for lag in DEFAULT_LAGS],
    *ROLLING_FEATURE_COLUMNS,
]
DEFAULT_PARAMETERS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
    "feature_fraction_seed": 42,
    "bagging_seed": 42,
    "num_threads": -1,
}
DEFAULT_NUM_BOOST_ROUND = 250
FORBIDDEN_FEATURES = {
    "transactions", "oil_price", "dcoilwtico", "forecast_readiness",
    "readiness_class", "sales_anomaly", "anomaly_method",
}


def add_known_features(
    sales: pd.DataFrame,
    stores: pd.DataFrame,
    holidays: pd.DataFrame,
) -> pd.DataFrame:
    """Attach only forecast-time-known calendar, promotion and metadata fields."""
    required = ["date", "store_nbr", "family", "sales", "onpromotion"]
    missing = [column for column in required if column not in sales]
    if missing:
        raise KeyError("sales is missing required columns: " + ", ".join(missing))
    result = sales.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    if result.duplicated(["date", "store_nbr", "family"]).any():
        raise ValueError("sales must have a unique date-store-family grain")
    result = result.merge(
        build_calendar_features(result["date"]),
        on="date",
        how="left",
        validate="many_to_one",
    )
    result = add_store_metadata(result, stores)
    result = add_promotion_features(result)
    result = add_holiday_features(result, holidays, stores)
    return encode_categorical_features(result)


def encode_categorical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Use stable pandas categorical columns accepted by native LightGBM."""
    result = frame.copy()
    for column in CATEGORICAL_FEATURES:
        if column not in result:
            raise KeyError(f"frame is missing categorical feature: {column}")
        result[column] = result[column].astype("category")
    return result


def build_causal_training_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build row-causal target-history features for supervised training rows."""
    return add_sales_rolling_features(add_sales_lag_features(frame))


def build_horizon_safe_features(
    frame: pd.DataFrame,
    forecast_origin: object,
    validation_start: object,
    validation_end: object,
) -> pd.DataFrame:
    """Build one horizon without allowing its actual targets into its features."""
    origin = pd.Timestamp(forecast_origin).normalize()
    start = pd.Timestamp(validation_start).normalize()
    end = pd.Timestamp(validation_end).normalize()
    if start != origin + pd.Timedelta(days=1) or start > end:
        raise ValueError("validation_start must be the day after forecast_origin")
    context_start = origin - pd.Timedelta(days=max(DEFAULT_LAGS))
    context = frame.loc[frame["date"].between(context_start, end)].copy()
    result = add_sales_lag_features(context, forecast_origin=origin)
    result = add_sales_rolling_features(result, forecast_origin=origin)
    horizon = result.loc[result["date"].between(start, end)].copy()
    if horizon.empty:
        raise ValueError("frame contains no rows in the requested forecast horizon")

    # Recursive seasonal references always resolve to an observed pre-origin date.
    # This supplies short lags without ever reading actual targets inside the horizon.
    history_lookup = context.loc[
        context["date"].le(origin), ["date", "store_nbr", "family", "sales"]
    ].rename(columns={"date": "_reference_date", "sales": "_safe_lag"})
    for lag in DEFAULT_LAGS:
        references = horizon[["date", "store_nbr", "family"]].copy()
        references["_reference_date"] = references["date"] - pd.Timedelta(days=lag)
        while references["_reference_date"].gt(origin).any():
            beyond_origin = references["_reference_date"].gt(origin)
            references.loc[beyond_origin, "_reference_date"] -= pd.Timedelta(days=lag)
        safe_values = references.merge(
            history_lookup,
            on=["_reference_date", "store_nbr", "family"],
            how="left",
            validate="many_to_one",
        )["_safe_lag"].to_numpy()
        horizon[f"sales_lag_{lag}"] = safe_values

    # Rolling statistics are a snapshot of shifted history at the fixed origin.
    # Repeating that snapshot avoids both validation-target updates and missingness
    # distribution shift on days 2-16.
    rolling_snapshot = horizon.loc[
        horizon["date"].eq(start), ["store_nbr", "family", *ROLLING_FEATURE_COLUMNS]
    ].rename(columns={column: f"_safe_{column}" for column in ROLLING_FEATURE_COLUMNS})
    horizon = horizon.merge(
        rolling_snapshot,
        on=["store_nbr", "family"],
        how="left",
        validate="many_to_one",
    )
    for column in ROLLING_FEATURE_COLUMNS:
        horizon[column] = horizon.pop(f"_safe_{column}")
    return horizon.reset_index(drop=True)


def feature_matrix(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return audited model columns and reject unsafe feature configuration."""
    selected = FEATURE_COLUMNS if feature_columns is None else list(feature_columns)
    if not selected:
        raise ValueError("feature_columns must contain at least one feature")
    missing = [column for column in selected if column not in frame]
    if missing:
        raise KeyError("frame is missing model features: " + ", ".join(missing))
    if FORBIDDEN_FEATURES.intersection(selected):
        raise RuntimeError("unsafe features are configured for the global model")
    return frame.loc[:, selected]


def train_global_model(
    feature_frame: pd.DataFrame,
    training_cutoff: object,
    *,
    parameters: dict[str, object] | None = None,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
    feature_columns: list[str] | None = None,
) -> tuple[lgb.Booster, dict[str, object]]:
    """Train one global model on ``log1p(sales)`` through one temporal cutoff."""
    cutoff = pd.Timestamp(training_cutoff).normalize()
    training = feature_frame.loc[feature_frame["date"].le(cutoff)].copy()
    if training.empty:
        raise ValueError("no training rows are available through training_cutoff")
    if training["sales"].isna().any() or training["sales"].lt(0).any():
        raise ValueError("training sales must be complete and nonnegative")
    merged_parameters = deepcopy(DEFAULT_PARAMETERS)
    if parameters:
        merged_parameters.update(parameters)
    target = np.log1p(training["sales"].to_numpy(dtype="float64"))
    selected_features = FEATURE_COLUMNS if feature_columns is None else list(feature_columns)
    categorical_features = [
        column for column in CATEGORICAL_FEATURES if column in selected_features
    ]
    dataset = lgb.Dataset(
        feature_matrix(training, selected_features),
        label=target,
        categorical_feature=categorical_features,
        free_raw_data=True,
    )
    model = lgb.train(merged_parameters, dataset, num_boost_round=num_boost_round)
    metadata: dict[str, object] = {
        "model_name": MODEL_NAME,
        "training_cutoff": cutoff.date().isoformat(),
        "feature_list": selected_features,
        "categorical_features": categorical_features,
        "parameters": merged_parameters,
        "num_boost_round": num_boost_round,
        "hyperparameter_tuning": False,
        "target_transform": "log1p(sales)",
        "prediction_inverse_transform": "clip(expm1(raw_prediction), lower=0)",
        "inference_strategy": (
            "fixed-origin horizon-safe seasonal lag references and frozen "
            "origin rolling snapshots"
        ),
    }
    return model, metadata
