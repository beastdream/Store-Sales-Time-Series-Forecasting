# %% [markdown]
# # Controlled Global LightGBM Tuning
#
# This is a small, reproducible search over three candidates plus the validated
# untuned control. Every candidate uses the same four rolling 16-day folds,
# the ablation-selected M6_NO_HOLIDAY feature set,
# features, log target, recursive calendar-day inference, seeds and 250 boosting rounds.
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
from src.modeling.ablation import M6_NO_HOLIDAY_FEATURES
from src.modeling.evaluate import score_predictions
from src.modeling.recursive import recursive_forecast
from src.modeling.splits import make_rolling_splits
from src.modeling.train_global import (
    DEFAULT_NUM_BOOST_ROUND,
    add_known_features,
    build_causal_training_features,
    train_global_model,
)
from src.modeling.tuning import (
    MAXIMUM_RMSLE_STD_DEGRADATION,
    MINIMUM_RMSLE_IMPROVEMENT,
    NEAR_TIE_RMSLE,
    SEARCH_CONFIGS,
    chosen_result,
    resolved_parameters,
    summarize_tuning,
)


REPORT_DIR = REPORTS_DIR / "modeling"
RESULTS_PATH = REPORT_DIR / "tuning_results.csv"
FOLD_SCORES_PATH = REPORT_DIR / "tuning_fold_scores.csv"
SUMMARY_PATH = REPORT_DIR / "tuning_summary.md"
ABLATION_SCORES_PATH = REPORT_DIR / "ablation_scores.csv"
BASELINE_SUMMARY_PATH = REPORT_DIR / "baseline_summary.csv"
CONFIG_PATH = MODELS_DIR / "global_lightgbm_chosen_config.json"
TUNED_MODEL_PATH = MODELS_DIR / "global_lightgbm_tuned.txt"
TUNED_METADATA_PATH = MODELS_DIR / "global_lightgbm_tuned_metadata.json"
HORIZON_DAYS = 16
N_FOLDS = 4
EXPECTED_SERIES = 1_782
EXPECTED_FOLD_ROWS = EXPECTED_SERIES * HORIZON_DAYS
FEATURE_SET_NAME = "M6_NO_HOLIDAY"
INFERENCE_STRATEGY = "recursive_untuned_or_tuned"
FOLD_SCORE_COLUMNS = [
    "experiment", "model", "feature_set", "inference_strategy", "fold",
    "train_end", "validation_start", "validation_end", "rmsle", "mae", "wape",
]


def seed_untuned_control() -> pd.DataFrame:
    """Reuse the authoritative validated T0 scores without retraining them."""
    scores = pd.read_csv(
        ABLATION_SCORES_PATH,
        parse_dates=["train_end", "validation_start", "validation_end"],
    )
    scores = scores.loc[
        scores["experiment"].eq(FEATURE_SET_NAME)
        & scores["inference_strategy"].eq("recursive_untuned")
    ].copy()
    if len(scores) != N_FOLDS:
        raise RuntimeError("M6_NO_HOLIDAY control must contain exactly four folds")
    scores["feature_set"] = FEATURE_SET_NAME
    scores["inference_strategy"] = INFERENCE_STRATEGY
    scores["model"] = "global_lightgbm"
    scores["experiment"] = "T0_untuned"
    scores = scores[FOLD_SCORE_COLUMNS]
    return scores


def validate_splits(splits: tuple[object, ...]) -> None:
    expected = [
        ("2017-06-12", "2017-06-13", "2017-06-28"),
        ("2017-06-28", "2017-06-29", "2017-07-14"),
        ("2017-07-14", "2017-07-15", "2017-07-30"),
        ("2017-07-30", "2017-07-31", "2017-08-15"),
    ]
    actual = [
        (
            split.train_end.date().isoformat(),
            split.validation_start.date().isoformat(),
            split.validation_end.date().isoformat(),
        )
        for split in splits
    ]
    if actual != expected:
        raise RuntimeError(f"unexpected temporal folds: {actual}")


def run_search() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run only the three predeclared candidates and checkpoint every fold."""
    control = seed_untuned_control()
    if FOLD_SCORES_PATH.exists():
        existing = pd.read_csv(
            FOLD_SCORES_PATH,
            parse_dates=["train_end", "validation_start", "validation_end"],
        )
        if set(FOLD_SCORE_COLUMNS).issubset(existing.columns):
            existing = existing.loc[
                existing["experiment"].isin(set(SEARCH_CONFIGS) - {"T0_untuned"})
                & existing["feature_set"].eq(FEATURE_SET_NAME)
                & existing["inference_strategy"].eq(INFERENCE_STRATEGY),
                FOLD_SCORE_COLUMNS,
            ]
        else:
            existing = pd.DataFrame(columns=FOLD_SCORE_COLUMNS)
        if existing.duplicated(["experiment", "fold"]).any():
            raise RuntimeError("tuning checkpoint contains duplicate experiment-fold rows")
        fold_scores = (
            control.copy()
            if existing.empty
            else pd.concat([control, existing], ignore_index=True)
        )
    else:
        fold_scores = control

    train = load_train()
    known = add_known_features(train, load_stores(), load_holidays())
    causal = build_causal_training_features(known)
    splits = make_rolling_splits(train["date"].max(), HORIZON_DAYS, N_FOLDS)
    validate_splits(splits)
    if train[["store_nbr", "family"]].drop_duplicates().shape[0] != EXPECTED_SERIES:
        raise RuntimeError("unexpected store-family series count")

    for experiment, overrides in SEARCH_CONFIGS.items():
        if experiment == "T0_untuned":
            continue
        parameters = resolved_parameters(overrides)
        for fold, split in enumerate(splits, start=1):
            completed = fold_scores["experiment"].eq(experiment) & fold_scores["fold"].eq(fold)
            if completed.any():
                print(f"{experiment} fold {fold}: reused checkpoint")
                continue
            horizon = known.loc[known["date"].between(
                split.validation_start, split.validation_end
            )].copy()
            if len(horizon) != EXPECTED_FOLD_ROWS:
                raise AssertionError(f"{experiment} fold {fold}: incomplete horizon")
            model, metadata = train_global_model(
                causal,
                split.train_end,
                parameters=parameters,
                num_boost_round=DEFAULT_NUM_BOOST_ROUND,
                feature_columns=M6_NO_HOLIDAY_FEATURES,
            )
            if metadata["feature_list"] != M6_NO_HOLIDAY_FEATURES:
                raise AssertionError("feature set changed during tuning")
            if model.feature_name() != M6_NO_HOLIDAY_FEATURES:
                raise AssertionError("trained model feature names changed during tuning")
            predictions = recursive_forecast(
                model, known, split.train_end,
                split.validation_start, split.validation_end,
            )
            metrics = score_predictions(horizon, predictions)
            row = pd.DataFrame(
                [{
                    "experiment": experiment,
                    "model": "global_lightgbm",
                    "feature_set": FEATURE_SET_NAME,
                    "inference_strategy": INFERENCE_STRATEGY,
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
    baseline = pd.read_csv(BASELINE_SUMMARY_PATH).sort_values(
        "rmsle_mean", kind="stable"
    ).iloc[0]
    untuned = results.loc[results["experiment"].eq("T0_untuned")].iloc[0]
    config = {
        "chosen_experiment": experiment,
        "selection_objective": "minimum mean RMSLE across all four temporal folds",
        "minimum_required_rmsle_improvement": MINIMUM_RMSLE_IMPROVEMENT,
        "maximum_allowed_rmsle_std_degradation": MAXIMUM_RMSLE_STD_DEGRADATION,
        "near_tie_rmsle": NEAR_TIE_RMSLE,
        "untuned_mean_rmsle": float(untuned["rmsle_mean"]),
        "untuned_rmsle_std": float(untuned["rmsle_std"]),
        "chosen_mean_rmsle": float(chosen["rmsle_mean"]),
        "chosen_rmsle_std": float(chosen["rmsle_std"]),
        "rmsle_improvement_vs_untuned": float(chosen["rmsle_improvement_vs_untuned"]),
        "parameters": parameters,
        "num_boost_round": DEFAULT_NUM_BOOST_ROUND,
        "chosen_mae_mean": float(chosen["mae_mean"]),
        "chosen_wape_mean": float(chosen["wape_mean"]),
        "feature_set_name": FEATURE_SET_NAME,
        "feature_selection_source": "recursive feature ablation",
        "feature_list": M6_NO_HOLIDAY_FEATURES,
        "target_transform": "log1p(sales)",
        "prediction_inverse_transform": "clip(expm1(raw_prediction), lower=0)",
        "temporal_folds": json.loads(
            selected_scores.to_json(orient="records", date_format="iso")
        ),
        "final_test_used_for_selection": False,
        "selection_uses_single_best_fold": False,
        "search_size_including_control": len(SEARCH_CONFIGS),
        "random_seed": 42,
        "inference_strategy": (
            "recursive calendar-day forecasting; each prior prediction updates "
            "later lag and rolling features"
        ),
        "strongest_baseline": {
            "model": str(baseline["model"]),
            "mean_rmsle": float(baseline["rmsle_mean"]),
            "rmsle_std": float(baseline["rmsle_std"]),
            "mae_mean": float(baseline["mae_mean"]),
            "wape_mean": float(baseline["wape_mean"]),
        },
    }

    if experiment == "T0_untuned":
        config["model_artifact"] = None
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
            feature_columns=M6_NO_HOLIDAY_FEATURES,
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
        config["model_artifact"] = TUNED_MODEL_PATH.relative_to(
            PROJECT_ROOT
        ).as_posix()
        config["selection_reason"] = (
            "Tuned candidate improved mean four-fold RMSLE beyond the predefined "
            f"{MINIMUM_RMSLE_IMPROVEMENT:.3f} threshold."
        )
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def write_summary(results: pd.DataFrame) -> None:
    """Write the controlled search comparison and selection rationale."""
    chosen = chosen_result(results)
    control = results.loc[results["experiment"].eq("T0_untuned")].iloc[0]
    rows = [
        "| experiment | mean RMSLE | RMSLE std | mean MAE | mean WAPE | fold 4 RMSLE | chosen |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    rows.extend(
        f"| {row.experiment} | {row.rmsle_mean:.6f} | {row.rmsle_std:.6f} | "
        f"{row.mae_mean:.6f} | {row.wape_mean:.6f} | {row.fold_4_rmsle:.6f} | "
        f"{'yes' if row.is_chosen else 'no'} |"
        for row in results.itertuples(index=False)
    )
    lines = [
        "# Controlled Recursive LightGBM Tuning",
        "",
        "All candidates use M6_NO_HOLIDAY, the same four rolling 16-day folds, "
        "recursive inference, 250 boosting rounds, and deterministic seeds. The "
        "final competition test is not loaded by this entrypoint.",
        "",
        *rows,
        "",
        "## Selection",
        "",
        f"**{chosen['experiment']}** was selected. Improvement versus untuned is "
        f"**{chosen['rmsle_improvement_vs_untuned']:.6f} RMSLE**. A tuned candidate "
        f"must improve mean RMSLE by at least {MINIMUM_RMSLE_IMPROVEMENT:.3f} and "
        f"may not degrade fold std by more than {MAXIMUM_RMSLE_STD_DEGRADATION:.3f}. "
        f"Candidates within {NEAR_TIE_RMSLE:.4f} are resolved by stability and then "
        "parameter simplicity.",
        "",
        f"Untuned control: mean RMSLE {control['rmsle_mean']:.6f}, std "
        f"{control['rmsle_std']:.6f}. Selected: mean RMSLE "
        f"{chosen['rmsle_mean']:.6f}, std {chosen['rmsle_std']:.6f}.",
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


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
    write_summary(results)
    print(results.to_string(index=False))
    print(f"Chosen configuration: {chosen['experiment']}")


if __name__ == "__main__":
    main()
