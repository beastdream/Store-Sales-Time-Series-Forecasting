# %% [markdown]
# # Controlled Global LightGBM Tuning
#
# This is a small, reproducible search over three candidates plus the validated
# untuned control. Every candidate uses the same four rolling 16-day folds, M6
# features, log target, fixed-origin inference, seeds and 250 boosting rounds.
# The final competition test is never loaded. Selection uses mean four-fold RMSLE,
# never the best individual fold.

# %%
import gc
import json
from pathlib import Path
import shutil
import sys
import tempfile

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, REPORTS_DIR
from src.data.load_raw import load_holidays, load_stores, load_train
from src.modeling.evaluate import score_predictions
from src.modeling.predict import predict_sales
from src.modeling.splits import make_rolling_splits
from src.modeling.train_global import (
    DEFAULT_NUM_BOOST_ROUND,
    FEATURE_COLUMNS,
    add_known_features,
    build_causal_training_features,
    build_horizon_safe_features,
    train_global_model,
)
from src.modeling.tuning import (
    MINIMUM_RMSLE_IMPROVEMENT,
    SEARCH_CONFIGS,
    chosen_result,
    resolved_parameters,
    summarize_tuning,
)


REPORT_DIR = REPORTS_DIR / "modeling"
RESULTS_PATH = REPORT_DIR / "tuning_results.csv"
FOLD_SCORES_PATH = REPORT_DIR / "tuning_fold_scores.csv"
UNTUNED_SCORES_PATH = REPORT_DIR / "global_lgbm_scores.csv"
CONFIG_PATH = MODELS_DIR / "global_lightgbm_chosen_config.json"
TUNED_MODEL_PATH = MODELS_DIR / "global_lightgbm_tuned.txt"
TUNED_METADATA_PATH = MODELS_DIR / "global_lightgbm_tuned_metadata.json"
UNTUNED_MODEL_PATH = MODELS_DIR / "global_lightgbm.txt"
HORIZON_DAYS = 16
N_FOLDS = 4


def seed_untuned_control() -> pd.DataFrame:
    """Reuse the authoritative validated T0 scores without retraining them."""
    scores = pd.read_csv(
        UNTUNED_SCORES_PATH,
        parse_dates=["train_end", "validation_start", "validation_end"],
    )
    scores.insert(0, "experiment", "T0_untuned")
    return scores


def run_search() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run only the three predeclared candidates and checkpoint every fold."""
    control = seed_untuned_control()
    if FOLD_SCORES_PATH.exists():
        existing = pd.read_csv(
            FOLD_SCORES_PATH,
            parse_dates=["train_end", "validation_start", "validation_end"],
        )
        existing = existing.loc[existing["experiment"].isin(SEARCH_CONFIGS)]
        existing = existing.loc[~existing["experiment"].eq("T0_untuned")]
        fold_scores = pd.concat([control, existing], ignore_index=True)
    else:
        fold_scores = control

    train = load_train()
    known = add_known_features(train, load_stores(), load_holidays())
    causal = build_causal_training_features(known)
    splits = make_rolling_splits(train["date"].max(), HORIZON_DAYS, N_FOLDS)
    expected_rows = train[["store_nbr", "family"]].drop_duplicates().shape[0] * HORIZON_DAYS

    for experiment, overrides in SEARCH_CONFIGS.items():
        if experiment == "T0_untuned":
            continue
        parameters = resolved_parameters(overrides)
        for fold, split in enumerate(splits, start=1):
            completed = fold_scores["experiment"].eq(experiment) & fold_scores["fold"].eq(fold)
            if completed.any():
                print(f"{experiment} fold {fold}: reused checkpoint")
                continue
            horizon = build_horizon_safe_features(
                known, split.train_end, split.validation_start, split.validation_end
            )
            if len(horizon) != expected_rows:
                raise AssertionError(f"{experiment} fold {fold}: incomplete horizon")
            model, metadata = train_global_model(
                causal,
                split.train_end,
                parameters=parameters,
                num_boost_round=DEFAULT_NUM_BOOST_ROUND,
                feature_columns=FEATURE_COLUMNS,
            )
            if metadata["feature_list"] != FEATURE_COLUMNS:
                raise AssertionError("feature set changed during tuning")
            metrics = score_predictions(horizon, predict_sales(model, horizon))
            row = pd.DataFrame(
                [{
                    "experiment": experiment,
                    "model": "global_lightgbm",
                    "fold": fold,
                    "train_end": split.train_end,
                    "validation_start": split.validation_start,
                    "validation_end": split.validation_end,
                    **metrics,
                }]
            )
            fold_scores = pd.concat([fold_scores, row], ignore_index=True)
            fold_scores.sort_values(["experiment", "fold"], kind="stable").to_csv(
                FOLD_SCORES_PATH, index=False
            )
            print(f"{experiment} fold {fold}: RMSLE={metrics['rmsle']:.6f}")
            del model, horizon
            gc.collect()
    results = summarize_tuning(fold_scores)
    return fold_scores, results


def save_selection(
    causal_features: pd.DataFrame,
    fold_scores: pd.DataFrame,
    results: pd.DataFrame,
) -> None:
    """Persist a reproducible config and train a new artifact only if tuned wins."""
    chosen = chosen_result(results)
    experiment = str(chosen["experiment"])
    overrides = SEARCH_CONFIGS[experiment]
    parameters = resolved_parameters(overrides)
    selected_scores = fold_scores.loc[fold_scores["experiment"].eq(experiment)]
    config = {
        "chosen_experiment": experiment,
        "selection_objective": "minimum mean RMSLE across all four temporal folds",
        "minimum_required_rmsle_improvement": MINIMUM_RMSLE_IMPROVEMENT,
        "untuned_mean_rmsle": float(
            results.loc[results["experiment"].eq("T0_untuned"), "rmsle_mean"].iloc[0]
        ),
        "chosen_mean_rmsle": float(chosen["rmsle_mean"]),
        "chosen_rmsle_std": float(chosen["rmsle_std"]),
        "rmsle_improvement_vs_untuned": float(chosen["rmsle_improvement_vs_untuned"]),
        "parameters": parameters,
        "num_boost_round": DEFAULT_NUM_BOOST_ROUND,
        "feature_list": FEATURE_COLUMNS,
        "target_transform": "log1p(sales)",
        "prediction_inverse_transform": "clip(expm1(raw_prediction), lower=0)",
        "temporal_folds": json.loads(
            selected_scores.to_json(orient="records", date_format="iso")
        ),
        "final_test_used_for_selection": False,
        "selection_uses_single_best_fold": False,
        "search_size_including_control": len(SEARCH_CONFIGS),
        "random_seed": 42,
    }

    if experiment == "T0_untuned":
        config["model_artifact"] = str(UNTUNED_MODEL_PATH.relative_to(PROJECT_ROOT))
        config["selection_reason"] = (
            "No tuned candidate improved four-fold mean RMSLE by at least "
            f"{MINIMUM_RMSLE_IMPROVEMENT:.3f}; retained validated untuned model."
        )
    else:
        cutoff = causal_features["date"].max().normalize()
        model, metadata = train_global_model(
            causal_features,
            cutoff,
            parameters=parameters,
            num_boost_round=DEFAULT_NUM_BOOST_ROUND,
            feature_columns=FEATURE_COLUMNS,
        )
        temporary = Path(tempfile.gettempdir()) / "store_sales_global_lightgbm_tuned.txt"
        model.save_model(str(temporary))
        shutil.move(str(temporary), TUNED_MODEL_PATH)
        metadata["hyperparameter_tuning"] = True
        metadata["chosen_experiment"] = experiment
        metadata["metric_results"] = config["temporal_folds"]
        metadata["mean_metrics"] = {
            "rmsle": float(chosen["rmsle_mean"]),
            "mae": float(chosen["mae_mean"]),
            "wape": float(chosen["wape_mean"]),
        }
        TUNED_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        config["model_artifact"] = str(TUNED_MODEL_PATH.relative_to(PROJECT_ROOT))
        config["selection_reason"] = (
            "Tuned candidate improved mean four-fold RMSLE beyond the predefined "
            f"{MINIMUM_RMSLE_IMPROVEMENT:.3f} threshold."
        )
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fold_scores, results = run_search()
    results.to_csv(RESULTS_PATH, index=False)

    # Rebuild once for final chosen training only when a tuned candidate wins.
    chosen = chosen_result(results)
    if chosen["experiment"] == "T0_untuned":
        causal = pd.DataFrame()
    else:
        train = load_train()
        known = add_known_features(train, load_stores(), load_holidays())
        causal = build_causal_training_features(known)
    save_selection(causal, fold_scores, results)
    print(results.to_string(index=False))
    print(f"Chosen configuration: {chosen['experiment']}")


if __name__ == "__main__":
    main()
