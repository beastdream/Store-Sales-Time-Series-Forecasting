"""Contracts for constructing and validating the final competition forecast."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from src.modeling.final_forecast import (
    EXPECTED_SUBMISSION_ROWS,
    create_submission,
    validate_final_submission,
    validate_final_test_contract,
)
from src.data.load_raw import load_test, load_train
from src.modeling.predict import load_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _competition_contract() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2017-08-16", "2017-08-31")
    stores = list(range(1, 55))
    families = [f"family_{number:02d}" for number in range(33)]
    grain = pd.MultiIndex.from_product(
        [dates, stores, families],
        names=["date", "store_nbr", "family"],
    ).to_frame(index=False)
    test = grain.copy()
    test.insert(0, "id", np.arange(3_000_889, 3_000_889 + len(test)))
    test["onpromotion"] = 0
    train = grain.loc[
        grain["date"].eq(pd.Timestamp("2017-08-16")),
        ["store_nbr", "family"],
    ].copy()
    train["date"] = pd.Timestamp("2017-08-15")
    train["sales"] = 1.0
    assert len(test) == EXPECTED_SUBMISSION_ROWS
    return test, train


def test_submission_preserves_original_ids_and_order() -> None:
    test, train = _competition_contract()
    predictions = test[["date", "store_nbr", "family"]].copy()
    predictions["prediction"] = np.arange(len(predictions), dtype="float64")
    predictions = predictions.iloc[::-1].reset_index(drop=True)

    submission = create_submission(test, predictions)
    validate_final_submission(submission, test, train)

    assert submission.columns.tolist() == ["id", "sales"]
    assert np.array_equal(submission["id"].to_numpy(), test["id"].to_numpy())


def test_final_test_contract_checks_date_store_family_coverage() -> None:
    test, train = _competition_contract()

    validate_final_test_contract(test, train)

    invalid = test.copy()
    invalid.loc[0, "family"] = invalid.loc[1, "family"]
    with pytest.raises(ValueError, match="duplicate date-store-family grain"):
        validate_final_test_contract(invalid, train)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -0.01, "not-numeric"])
def test_submission_rejects_invalid_predictions(invalid_value: object) -> None:
    test, train = _competition_contract()
    submission = test[["id"]].copy()
    submission["sales"] = 0.0
    if isinstance(invalid_value, str):
        submission["sales"] = submission["sales"].astype("object")
    submission.loc[0, "sales"] = invalid_value

    with pytest.raises(ValueError, match="numeric|finite|nonnegative"):
        validate_final_submission(submission, test, train)


def test_create_submission_rejects_incomplete_prediction_grain() -> None:
    test, _ = _competition_contract()
    predictions = test[["date", "store_nbr", "family"]].iloc[:-1].copy()
    predictions["prediction"] = 0.0

    with pytest.raises(ValueError, match="exactly one row per test row"):
        create_submission(test, predictions)


def test_final_entrypoint_does_not_load_submission_targets() -> None:
    source = (
        PROJECT_ROOT / "notebooks" / "20_final_competition_forecast.py"
    ).read_text(encoding="utf-8")

    assert "load_sample_submission" not in source
    assert "global_lightgbm_chosen_config.json" in source
    assert "final_test_used_for_model_selection" in source


def test_final_submission_artifact_satisfies_competition_contract() -> None:
    submission = pd.read_csv(
        PROJECT_ROOT / "reports" / "modeling" / "final_submission.csv"
    )
    test = load_test()
    train = load_train()

    validate_final_submission(submission, test, train)

    assert submission.columns.tolist() == ["id", "sales"]
    assert len(submission) == EXPECTED_SUBMISSION_ROWS
    assert submission["id"].is_unique
    assert np.array_equal(submission["id"].to_numpy(), test["id"].to_numpy())
    assert np.isfinite(submission["sales"]).all()
    assert submission["sales"].ge(0).all()


def test_final_model_and_metadata_match_validation_selected_config() -> None:
    model_path = PROJECT_ROOT / "models" / "final_global_lightgbm.txt"
    metadata = json.loads(
        (
            PROJECT_ROOT / "models" / "final_global_lightgbm_metadata.json"
        ).read_text(encoding="utf-8")
    )
    chosen = json.loads(
        (
            PROJECT_ROOT / "models" / "global_lightgbm_chosen_config.json"
        ).read_text(encoding="utf-8")
    )

    assert load_model(model_path).num_trees() == chosen["num_boost_round"] == 250
    assert metadata["model_type"] == "global LightGBM regression"
    assert metadata["chosen_experiment"] == chosen["chosen_experiment"]
    assert metadata["feature_list"] == chosen["feature_list"]
    assert metadata["parameters"] == chosen["parameters"]
    assert metadata["training_cutoff"] == "2017-08-15"
    assert metadata["forecast_start"] == "2017-08-16"
    assert metadata["forecast_end"] == "2017-08-31"
    assert metadata["forecast_horizon_days"] == 16
    assert metadata["target_transform"] == chosen["target_transform"]
    assert metadata["validation_metrics"]["mean_rmsle"] == chosen[
        "chosen_mean_rmsle"
    ]
    assert len(metadata["validation_metrics"]["temporal_folds"]) == 4
    assert metadata["final_test_used_for_model_selection"] is False
    assert metadata["submission"]["row_count"] == EXPECTED_SUBMISSION_ROWS
