# %% [markdown]
# # Final Competition Forecast
#
# This entrypoint retrains only the validation-selected global LightGBM on all
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
REPORT_PATH = REPORTS_DIR / "modeling" / "final_forecast_report.md"


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
            "model_artifact": MODEL_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "chosen_experiment": chosen_config["chosen_experiment"],
            "feature_set_name": chosen_config["feature_set_name"],
            "selected_parameters": chosen_config["parameters"],
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
                "mean_mae": chosen_config["chosen_mae_mean"],
                "mean_wape": chosen_config["chosen_wape_mean"],
                "rmsle_improvement_vs_untuned": chosen_config[
                    "rmsle_improvement_vs_untuned"
                ],
                "temporal_folds": chosen_config["temporal_folds"],
            },
            "baseline_comparison": chosen_config["strongest_baseline"],
            "submission": {
                "path": SUBMISSION_PATH.relative_to(PROJECT_ROOT).as_posix(),
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
            "hyperparameter_tuning": (
                chosen_config["chosen_experiment"] != "T0_untuned"
            ),
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

    fold_4 = chosen_config["temporal_folds"][-1]
    baseline = chosen_config["strongest_baseline"]
    report_lines = [
        "# Final Recursive Forecast",
        "",
        f"- Selected configuration: **{chosen_config['chosen_experiment']}**.",
        f"- Feature set: **{chosen_config['feature_set_name']}** "
        f"({len(chosen_config['feature_list'])} features).",
        f"- Untuned mean RMSLE: **{chosen_config['untuned_mean_rmsle']:.6f}**.",
        f"- Selected mean RMSLE: **{chosen_config['chosen_mean_rmsle']:.6f}**; "
        f"std: **{chosen_config['chosen_rmsle_std']:.6f}**.",
        f"- Mean MAE: **{chosen_config['chosen_mae_mean']:.6f}**; "
        f"mean WAPE: **{chosen_config['chosen_wape_mean']:.6f}**.",
        f"- Fold 4 RMSLE: **{fold_4['rmsle']:.6f}**; MAE: "
        f"**{fold_4['mae']:.6f}**; WAPE: **{fold_4['wape']:.6f}**.",
        f"- Strongest baseline: **{baseline['model']}**, mean RMSLE "
        f"**{baseline['mean_rmsle']:.6f}**.",
        "- Inference: recursive calendar-day forecasting; no final-test target "
        "was used for feature or parameter selection.",
        f"- Submission: **{len(submission):,} rows**, exact original test ID order, "
        "unique IDs, finite and nonnegative predictions.",
        f"- Model artifact: `{MODEL_PATH.relative_to(PROJECT_ROOT).as_posix()}`.",
        f"- Metadata: `{METADATA_PATH.relative_to(PROJECT_ROOT).as_posix()}`.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Final submission rows: {len(submission):,}")
    print(f"Forecast dates: {FINAL_FORECAST_START.date()}..{FINAL_FORECAST_END.date()}")
    print(f"Submission: {SUBMISSION_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Model: {MODEL_PATH.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
