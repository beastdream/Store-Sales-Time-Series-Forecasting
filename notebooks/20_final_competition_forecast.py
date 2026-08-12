# %% [markdown]
# # Final Competition Forecast
#
# This entrypoint retrains only the validation-selected T2 global LightGBM on all
# actual sales through 2017-08-15. The final horizon is generated recursively,
# feeding prior predictions into later calendar-day lag/rolling features. Final test covariates are used only for the
# 2017-08-16 through 2017-08-31 forecast, never for model/parameter selection.
# The shadow intermittent router is not used because no origin-causal routing rule
# has yet been validated.

# %%
import json
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, REPORTS_DIR
from src.data.load_raw import load_holidays, load_stores, load_test, load_train
from src.modeling.final_forecast import (
    FINAL_FORECAST_END,
    FINAL_FORECAST_START,
    FINAL_HORIZON_DAYS,
    FINAL_TRAINING_CUTOFF,
    train_and_predict_final,
    validate_final_submission,
)
from src.modeling.predict import load_model


CONFIG_PATH = MODELS_DIR / "global_lightgbm_chosen_config.json"
MODEL_PATH = MODELS_DIR / "final_global_lightgbm.txt"
METADATA_PATH = MODELS_DIR / "final_global_lightgbm_metadata.json"
SUBMISSION_PATH = REPORTS_DIR / "modeling" / "final_submission.csv"


def _publish_temp_file(source: Path, destination: Path) -> None:
    """Stage on the destination volume, then atomically replace the final file."""
    staged = destination.with_name(f".{destination.name}.staged")
    shutil.copy2(source, staged)
    staged.replace(destination)
    source.unlink(missing_ok=True)


def main() -> None:
    """Train, predict, validate, then publish final artifacts in that order."""
    chosen_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    train = load_train()
    test = load_test()
    model, submission, metadata = train_and_predict_final(
        train,
        test,
        load_stores(),
        load_holidays(),
        chosen_config,
    )

    # Final gate is repeated immediately before any output is written.
    validate_final_submission(submission, test, train)
    temporary_dir = Path(tempfile.gettempdir())
    temporary_model = temporary_dir / "store_sales_final_global_lightgbm.txt"
    temporary_submission = temporary_dir / "store_sales_final_submission.csv"
    model.save_model(str(temporary_model))
    submission.to_csv(temporary_submission, index=False)

    metadata.update(
        {
            "artifact_role": "final_competition_forecast_model",
            "model_type": "global LightGBM regression",
            "chosen_experiment": chosen_config["chosen_experiment"],
            "selection_objective": chosen_config["selection_objective"],
            "training_cutoff": FINAL_TRAINING_CUTOFF.date().isoformat(),
            "training_start": pd.to_datetime(train["date"]).min().date().isoformat(),
            "training_row_count": len(train),
            "forecast_start": FINAL_FORECAST_START.date().isoformat(),
            "forecast_end": FINAL_FORECAST_END.date().isoformat(),
            "forecast_horizon_days": FINAL_HORIZON_DAYS,
            "validation_metrics": {
                "mean_rmsle": chosen_config["chosen_mean_rmsle"],
                "rmsle_std": chosen_config["chosen_rmsle_std"],
                "rmsle_improvement_vs_untuned": chosen_config[
                    "rmsle_improvement_vs_untuned"
                ],
                "temporal_folds": chosen_config["temporal_folds"],
            },
            "submission": {
                "path": str(SUBMISSION_PATH.relative_to(PROJECT_ROOT)),
                "row_count": len(submission),
                "id_order_preserved": True,
                "minimum_prediction": float(submission["sales"].min()),
                "maximum_prediction": float(submission["sales"].max()),
            },
            "target_transform": chosen_config["target_transform"],
            "prediction_inverse_transform": chosen_config[
                "prediction_inverse_transform"
            ],
            "final_test_used_for_model_selection": False,
            "hyperparameter_tuning": True,
            "parameter_selection": "four-fold temporal validation",
            "test_id_sha256": hashlib.sha256(
                test["id"].to_numpy().tobytes()
            ).hexdigest(),
            "routing_strategy": (
                "global model for all series; intermittent two-stage routing excluded "
                "because it remains shadow-only without an origin-causal router"
            ),
        }
    )

    REPORTS_DIR.joinpath("modeling").mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    temporary_metadata = temporary_dir / "store_sales_final_metadata.json"
    temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Verify the serialized artifacts as they will be consumed before publishing.
    serialized_submission = pd.read_csv(temporary_submission)
    validate_final_submission(serialized_submission, test, train)
    serialized_metadata = json.loads(temporary_metadata.read_text(encoding="utf-8"))
    if serialized_metadata["final_test_used_for_model_selection"] is not False:
        raise RuntimeError("final metadata failed the test-selection leakage gate")
    if load_model(temporary_model).num_trees() != int(
        chosen_config["num_boost_round"]
    ):
        raise RuntimeError("serialized final model tree count is invalid")

    _publish_temp_file(temporary_model, MODEL_PATH)
    _publish_temp_file(temporary_submission, SUBMISSION_PATH)
    _publish_temp_file(temporary_metadata, METADATA_PATH)

    print(f"Final submission rows: {len(submission):,}")
    print(f"Forecast dates: {FINAL_FORECAST_START.date()}..{FINAL_FORECAST_END.date()}")
    print(f"Submission: {SUBMISSION_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Model: {MODEL_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
