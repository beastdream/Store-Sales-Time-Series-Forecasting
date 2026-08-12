"""Strict construction and validation for the final competition forecast."""

import numpy as np
import pandas as pd

from src.modeling.predict import predict_sales
from src.modeling.train_global import (
    add_known_features,
    build_causal_training_features,
    build_horizon_safe_features,
    train_global_model,
)


FINAL_TRAINING_CUTOFF = pd.Timestamp("2017-08-15")
FINAL_FORECAST_START = pd.Timestamp("2017-08-16")
FINAL_FORECAST_END = pd.Timestamp("2017-08-31")
FINAL_HORIZON_DAYS = 16
EXPECTED_SUBMISSION_ROWS = 28_512
GRAIN = ["date", "store_nbr", "family"]


def validate_final_test_contract(test: pd.DataFrame, train: pd.DataFrame) -> None:
    """Validate the immutable competition horizon before training starts."""
    required_test = ["id", *GRAIN, "onpromotion"]
    missing_test = [column for column in required_test if column not in test]
    if missing_test:
        raise KeyError("test is missing columns: " + ", ".join(missing_test))
    if len(test) != EXPECTED_SUBMISSION_ROWS:
        raise ValueError(f"test must contain exactly {EXPECTED_SUBMISSION_ROWS:,} rows")
    if test["id"].isna().any() or test["id"].duplicated().any():
        raise ValueError("test IDs must be complete and unique")
    if test.duplicated(GRAIN).any():
        raise ValueError("test contains duplicate date-store-family grain")

    test_dates = pd.DatetimeIndex(pd.to_datetime(test["date"])).normalize()
    expected_dates = pd.date_range(FINAL_FORECAST_START, FINAL_FORECAST_END)
    if not test_dates.unique().sort_values().equals(expected_dates):
        raise ValueError("test date coverage does not match the final 16-day horizon")

    store_count = test["store_nbr"].nunique()
    family_count = test["family"].nunique()
    if store_count != 54 or family_count != 33:
        raise ValueError("expected 54 stores and 33 families in final test")
    expected_per_date = store_count * family_count
    if expected_per_date != EXPECTED_SUBMISSION_ROWS // FINAL_HORIZON_DAYS:
        raise ValueError("test store-family cross-product is inconsistent")
    if not pd.Series(test_dates).value_counts().eq(expected_per_date).all():
        raise ValueError("test does not have complete store-family coverage on every date")
    if set(test["store_nbr"]) != set(train["store_nbr"]):
        raise ValueError("test store coverage does not match historical training coverage")
    if set(test["family"].astype(str)) != set(train["family"].astype(str)):
        raise ValueError("test family coverage does not match historical training coverage")


def build_final_model_inputs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    stores: pd.DataFrame,
    holidays: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build causal training features and fixed-origin test-horizon features."""
    historical = train.copy()
    future = test.copy()
    future["sales"] = pd.Series(np.nan, index=future.index, dtype="float64")
    combined = pd.concat([historical, future], ignore_index=True, sort=False)
    known = add_known_features(combined, stores, holidays)
    causal_training = build_causal_training_features(known)
    horizon = build_horizon_safe_features(
        known,
        FINAL_TRAINING_CUTOFF,
        FINAL_FORECAST_START,
        FINAL_FORECAST_END,
    )
    return causal_training, horizon


def create_submission(test: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Map predictions back to exact original test IDs and row order."""
    required_test = ["id", *GRAIN]
    missing_test = [column for column in required_test if column not in test]
    if missing_test:
        raise KeyError("test is missing columns: " + ", ".join(missing_test))
    required_prediction = [*GRAIN, "prediction"]
    missing_prediction = [
        column for column in required_prediction if column not in predictions
    ]
    if missing_prediction:
        raise KeyError(
            "predictions are missing columns: " + ", ".join(missing_prediction)
        )
    if len(predictions) != len(test):
        raise ValueError("predictions must contain exactly one row per test row")
    if predictions.duplicated(GRAIN).any():
        raise ValueError("predictions contain duplicate date-store-family grain")
    test_grain = pd.MultiIndex.from_frame(test[GRAIN])
    prediction_grain = pd.MultiIndex.from_frame(predictions[GRAIN])
    if not test_grain.difference(prediction_grain).empty:
        raise ValueError("predictions do not cover the complete test grain")
    if not prediction_grain.difference(test_grain).empty:
        raise ValueError("predictions contain rows outside the test grain")
    ordered_test = test[required_test].copy()
    ordered_test["_original_row_order"] = np.arange(len(ordered_test))
    mapped = ordered_test.merge(
        predictions[required_prediction],
        on=GRAIN,
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values("_original_row_order", kind="stable")
    return mapped[["id", "prediction"]].rename(columns={"prediction": "sales"})


def validate_final_submission(
    submission: pd.DataFrame,
    test: pd.DataFrame,
    train: pd.DataFrame,
) -> None:
    """Raise before publishing any invalid or misleading competition submission."""
    if list(submission.columns) != ["id", "sales"]:
        raise ValueError("submission schema must be exactly: id, sales")
    validate_final_test_contract(test, train)
    if len(submission) != EXPECTED_SUBMISSION_ROWS:
        raise ValueError(f"submission must contain exactly {EXPECTED_SUBMISSION_ROWS:,} rows")
    if submission["id"].isna().any() or submission["id"].duplicated().any():
        raise ValueError("submission IDs must be complete and unique")
    if not np.array_equal(submission["id"].to_numpy(), test["id"].to_numpy()):
        raise ValueError("submission IDs or row order do not match the original test")
    try:
        sales = pd.to_numeric(submission["sales"], errors="raise").to_numpy(
            dtype="float64"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("submission sales must be numeric") from exc
    if not np.isfinite(sales).all():
        raise ValueError("submission sales must be numeric, finite and non-missing")
    if (sales < 0).any():
        raise ValueError("submission sales must be nonnegative")


def train_and_predict_final(
    train: pd.DataFrame,
    test: pd.DataFrame,
    stores: pd.DataFrame,
    holidays: pd.DataFrame,
    chosen_config: dict[str, object],
) -> tuple[object, pd.DataFrame, dict[str, object]]:
    """Retrain the validation-selected strategy and return a validated submission."""
    if pd.to_datetime(train["date"]).max().normalize() != FINAL_TRAINING_CUTOFF:
        raise ValueError("historical training data must end on 2017-08-15")
    if chosen_config.get("chosen_experiment") != "T2_moderate_capacity":
        raise ValueError("final forecast requires the validation-chosen T2 configuration")
    if chosen_config.get("final_test_used_for_selection") is not False:
        raise ValueError("chosen configuration must confirm final test was not used")
    validate_final_test_contract(test, train)

    causal_training, horizon = build_final_model_inputs(
        train, test, stores, holidays
    )
    model, metadata = train_global_model(
        causal_training,
        FINAL_TRAINING_CUTOFF,
        parameters=chosen_config["parameters"],
        num_boost_round=int(chosen_config["num_boost_round"]),
        feature_columns=list(chosen_config["feature_list"]),
    )
    predictions = predict_sales(model, horizon)
    submission = create_submission(test, predictions)
    validate_final_submission(submission, test, train)
    return model, submission, metadata
