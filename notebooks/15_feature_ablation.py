# %% [markdown]
# # Controlled Global LightGBM Feature Ablation
#
# Every LightGBM experiment uses the same four 16-day folds, fixed parameters,
# 250 boosting rounds, log target and recursive calendar-day inference. M7 is gated off:
# the current oil interpolation reads future values and has not passed a causal
# availability scenario. No final test target is loaded or used.

# %%
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import REPORTS_DIR
from src.data.load_raw import load_holidays, load_stores, load_train
from src.modeling.ablation import (
    ADDED_GROUP,
    EXPERIMENT_FEATURES,
    NEGLIGIBLE_RMSLE_THRESHOLD,
    recommended_experiment,
    summarize_ablation,
)
from src.modeling.evaluate import score_predictions
from src.modeling.recursive import recursive_forecast
from src.modeling.splits import make_rolling_splits
from src.modeling.train_global import (
    DEFAULT_NUM_BOOST_ROUND,
    DEFAULT_PARAMETERS,
    MODEL_NAME,
    add_known_features,
    build_causal_training_features,
    train_global_model,
)


REPORT_DIR = REPORTS_DIR / "modeling"
SCORES_PATH = REPORT_DIR / "ablation_scores.csv"
SUMMARY_PATH = REPORT_DIR / "ablation_summary.md"
BASELINE_SCORES_PATH = REPORT_DIR / "baseline_scores.csv"
BASELINE_SUMMARY_PATH = REPORT_DIR / "baseline_summary.csv"
FULL_MODEL_SCORES_PATH = REPORT_DIR / "global_lgbm_scores.csv"
HORIZON_DAYS = 16
N_FOLDS = 4


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an extra dependency."""
    headers = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(headers) + " |"]
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(rows)


def seed_control_rows() -> pd.DataFrame:
    """Reuse already validated M0 and identical-configuration M6 fold scores."""
    baseline_summary = pd.read_csv(BASELINE_SUMMARY_PATH)
    strongest_model = baseline_summary.loc[baseline_summary["rmsle_mean"].idxmin(), "model"]
    m0 = pd.read_csv(BASELINE_SCORES_PATH, parse_dates=["train_end", "validation_start", "validation_end"])
    m0 = m0.loc[m0["model"].eq(strongest_model)].copy()
    m0["experiment"] = "M0"
    m0["added_group"] = ADDED_GROUP["M0"]

    m6 = pd.read_csv(FULL_MODEL_SCORES_PATH, parse_dates=["train_end", "validation_start", "validation_end"])
    m6["experiment"] = "M6"
    m6["added_group"] = ADDED_GROUP["M6"]
    columns = [
        "experiment", "model", "added_group", "fold", "train_end",
        "validation_start", "validation_end", "rmsle", "mae", "wape",
    ]
    return pd.concat([m0[columns], m6[columns]], ignore_index=True)


def run_experiments() -> pd.DataFrame:
    """Run M1-M5 and checkpoint deterministic fold results after every fold."""
    control = seed_control_rows()
    existing = pd.read_csv(SCORES_PATH, parse_dates=["train_end", "validation_start", "validation_end"]) if SCORES_PATH.exists() else control
    valid_experiments = {"M0", "M1", "M2", "M3", "M4", "M5", "M6"}
    existing = existing.loc[existing["experiment"].isin(valid_experiments)]
    # M0 and M6 are replaced from authoritative existing backtests each run.
    existing = existing.loc[~existing["experiment"].isin(["M0", "M6"])]
    scores = pd.concat([control, existing], ignore_index=True)

    train = load_train()
    known = add_known_features(train, load_stores(), load_holidays())
    causal = build_causal_training_features(known)
    splits = make_rolling_splits(train["date"].max(), HORIZON_DAYS, N_FOLDS)
    expected_rows = train[["store_nbr", "family"]].drop_duplicates().shape[0] * HORIZON_DAYS

    for experiment, features in EXPERIMENT_FEATURES.items():
        if experiment == "M6":
            continue
        for fold, split in enumerate(splits, start=1):
            completed = scores["experiment"].eq(experiment) & scores["fold"].eq(fold)
            if completed.any():
                print(f"{experiment} fold {fold}: reused checkpoint")
                continue
            horizon = known.loc[known["date"].between(
                split.validation_start, split.validation_end
            )].copy()
            if len(horizon) != expected_rows:
                raise AssertionError(f"{experiment} fold {fold}: incomplete horizon")
            model, metadata = train_global_model(
                causal,
                split.train_end,
                parameters=DEFAULT_PARAMETERS,
                num_boost_round=DEFAULT_NUM_BOOST_ROUND,
                feature_columns=features,
            )
            if metadata["parameters"] != DEFAULT_PARAMETERS:
                raise AssertionError("ablation parameters changed across experiments")
            predictions = recursive_forecast(
                model, known, split.train_end,
                split.validation_start, split.validation_end,
            )
            metrics = score_predictions(horizon, predictions)
            row = pd.DataFrame(
                [{
                    "experiment": experiment,
                    "model": MODEL_NAME,
                    "added_group": ADDED_GROUP[experiment],
                    "fold": fold,
                    "train_end": split.train_end,
                    "validation_start": split.validation_start,
                    "validation_end": split.validation_end,
                    **metrics,
                }]
            )
            scores = pd.concat([scores, row], ignore_index=True)
            scores.sort_values(["experiment", "fold"], kind="stable").to_csv(SCORES_PATH, index=False)
            print(f"{experiment} fold {fold}: RMSLE={metrics['rmsle']:.6f}")
    return scores.sort_values(["experiment", "fold"], kind="stable").reset_index(drop=True)


def write_summary(scores: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_ablation(scores)
    best = recommended_experiment(summary)
    best_order = int(str(best.experiment).removeprefix("M"))
    recommended_features: list[str] = []
    previous_features: list[str] = []
    excluded_groups: list[str] = []
    for experiment, current_features in EXPERIMENT_FEATURES.items():
        if int(experiment.removeprefix("M")) > best_order:
            break
        added_features = [feature for feature in current_features if feature not in previous_features]
        effect = summary.loc[summary["experiment"].eq(experiment), "effect"].iloc[0]
        if effect == "improved":
            recommended_features.extend(added_features)
        else:
            excluded_groups.append(ADDED_GROUP[experiment])
        previous_features = current_features
    table = summary.copy()
    numeric = ["rmsle_mean", "rmsle_std", "mae_mean", "wape_mean", "delta_rmsle_vs_previous"]
    table[numeric] = table[numeric].round(6)
    lines = [
        "# Controlled Global LightGBM Feature Ablation",
        "",
        "All M1-M6 experiments use the same four rolling 16-day folds, the same fixed "
        f"LightGBM parameters, and {DEFAULT_NUM_BOOST_ROUND} boosting rounds. No "
        "hyperparameter tuning or final test target is used.",
        "",
        f"Effects use mean RMSLE relative to the immediately preceding experiment. "
        f"Absolute changes below {NEGLIGIBLE_RMSLE_THRESHOLD:.3f} are `negligible effect`.",
        "",
        markdown_table(table),
        "",
        "## Feature-group conclusions",
        "",
    ]
    for row in summary.itertuples(index=False):
        if row.experiment == "M0":
            continue
        lines.append(
            f"- **{row.added_group} ({row.experiment}): {row.effect}.** "
            f"Mean RMSLE change versus previous = {row.delta_rmsle_vs_previous:+.6f}."
        )
    lines.extend(
        [
            "- **Oil features (M7): not run.** The current oil cleaner uses future-aware "
            "linear interpolation and `bfill`; no leakage-safe availability scenario has passed.",
            "",
            "## Recommended feature set",
            "",
            f"**{best.experiment}** is the lowest-scoring complete experiment, with mean "
            f"RMSLE {best.rmsle_mean:.6f}. For the next model, recommend only feature "
            "groups whose incremental effect improved validation RMSLE. The resulting "
            f"feature set is: `{', '.join(recommended_features)}`.",
            "",
            f"Excluded despite appearing in the cumulative best experiment: "
            f"**{', '.join(excluded_groups)}**, because its measured effect was not an "
            "improvement. This reduced combination has not itself been backtested and must "
            "pass a confirmation run before replacing the validated M6 artifact. Features "
            "added after the best experiment are not recommended without evidence. M7 "
            "remains prohibited.",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scores = run_experiments()
    scores.to_csv(SCORES_PATH, index=False)
    summary = write_summary(scores)
    print(summary.to_string(index=False))
    print("M7 not run: oil has not passed a leakage-safe availability scenario.")


if __name__ == "__main__":
    main()
