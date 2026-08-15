# %% [markdown]
# # Recursive Untuned Global LightGBM Backtest
#
# This evaluation uses the existing four rolling 16-day folds, the base untuned
# LightGBM parameters, the current M6 feature list, and shared recursive inference.
# It writes separate recursive artifacts and never overwrites legacy scores,
# tuned/final models, or the final submission. No tuning occurs here.

# %%
import gc
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import REPORTS_DIR
from src.data.load_raw import load_holidays, load_stores, load_train
from src.modeling.evaluate import score_predictions
from src.modeling.recursive import recursive_forecast
from src.modeling.splits import make_rolling_splits
from src.modeling.train_global import (
    DEFAULT_NUM_BOOST_ROUND,
    DEFAULT_PARAMETERS,
    FEATURE_COLUMNS,
    add_known_features,
    build_causal_training_features,
    train_global_model,
)


REPORT_DIR = REPORTS_DIR / "modeling"
SCORES_PATH = REPORT_DIR / "recursive_backtest_scores.csv"
OOF_PATH = REPORT_DIR / "recursive_global_lgbm_oof_predictions.parquet"
COMPARISON_PATH = REPORT_DIR / "recursive_vs_previous_strategy.md"
LEGACY_SCORES_PATH = REPORT_DIR / "global_lgbm_scores.csv"
BASELINE_SUMMARY_PATH = REPORT_DIR / "baseline_summary.csv"
HORIZON_DAYS = 16
N_FOLDS = 4
EXPECTED_SERIES = 1_782
EXPECTED_FOLD_ROWS = EXPECTED_SERIES * HORIZON_DAYS
EXPECTED_OOF_ROWS = EXPECTED_FOLD_ROWS * N_FOLDS


def _validate_splits(splits: tuple[object, ...]) -> None:
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


def _load_checkpoints() -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(SCORES_PATH) if SCORES_PATH.is_file() else pd.DataFrame()
    oof = pd.read_parquet(OOF_PATH) if OOF_PATH.is_file() else pd.DataFrame()
    return scores, oof


def _validate_oof(oof: pd.DataFrame) -> None:
    required = ["fold", "date", "store_nbr", "family", "actual", "prediction"]
    if list(oof.columns) != required:
        raise RuntimeError("recursive OOF schema is invalid")
    if len(oof) != EXPECTED_OOF_ROWS:
        raise RuntimeError(
            f"recursive OOF must contain {EXPECTED_OOF_ROWS:,} rows, got {len(oof):,}"
        )
    if oof.duplicated(["fold", "date", "store_nbr", "family"]).any():
        raise RuntimeError("recursive OOF contains duplicate grain")
    values = oof["prediction"].to_numpy(dtype="float64")
    if not np.isfinite(values).all() or (values < 0).any():
        raise RuntimeError("recursive OOF predictions must be finite and nonnegative")
    if not oof.groupby("fold").size().eq(EXPECTED_FOLD_ROWS).all():
        raise RuntimeError("recursive OOF fold coverage is incomplete")


def run_recursive_backtest() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run or resume the four-fold base-model recursive evaluation."""
    train = load_train()
    known = add_known_features(train, load_stores(), load_holidays())
    causal = build_causal_training_features(known)
    splits = make_rolling_splits(train["date"].max(), HORIZON_DAYS, N_FOLDS)
    _validate_splits(splits)
    if known[["store_nbr", "family"]].drop_duplicates().shape[0] != EXPECTED_SERIES:
        raise RuntimeError("unexpected store-family series count")

    scores, oof = _load_checkpoints()
    for fold, split in enumerate(splits, start=1):
        if not scores.empty and scores["fold"].eq(fold).any():
            if oof.empty or len(oof.loc[oof["fold"].eq(fold)]) != EXPECTED_FOLD_ROWS:
                raise RuntimeError(f"fold {fold} score exists without complete OOF rows")
            print(f"Fold {fold}/{N_FOLDS}: reused recursive checkpoint")
            continue

        model, metadata = train_global_model(
            causal,
            split.train_end,
            parameters=DEFAULT_PARAMETERS,
            num_boost_round=DEFAULT_NUM_BOOST_ROUND,
            feature_columns=FEATURE_COLUMNS,
        )
        if metadata["parameters"] != DEFAULT_PARAMETERS:
            raise RuntimeError("base parameters changed during recursive backtest")
        prediction = recursive_forecast(
            model,
            known,
            split.train_end,
            split.validation_start,
            split.validation_end,
        )
        actual = known.loc[
            known["date"].between(split.validation_start, split.validation_end),
            ["date", "store_nbr", "family", "sales"],
        ].rename(columns={"sales": "actual"})
        fold_oof = actual.merge(
            prediction,
            on=["date", "store_nbr", "family"],
            how="left",
            validate="one_to_one",
        )
        fold_oof.insert(0, "fold", fold)
        if len(fold_oof) != EXPECTED_FOLD_ROWS:
            raise RuntimeError(f"fold {fold}: incomplete recursive OOF rows")
        metrics = score_predictions(
            fold_oof.rename(columns={"actual": "sales"}), prediction
        )
        score = pd.DataFrame(
            [
                {
                    "strategy": "recursive_global_lightgbm_untuned",
                    "fold": fold,
                    "train_end": split.train_end,
                    "validation_start": split.validation_start,
                    "validation_end": split.validation_end,
                    **metrics,
                }
            ]
        )
        scores = pd.concat([scores, score], ignore_index=True)
        oof = pd.concat([oof, fold_oof], ignore_index=True)
        scores.sort_values("fold", kind="stable").to_csv(SCORES_PATH, index=False)
        oof.sort_values(
            ["fold", "date", "store_nbr", "family"], kind="stable"
        ).to_parquet(OOF_PATH, index=False)
        print(
            f"Fold {fold}/{N_FOLDS}: RMSLE={metrics['rmsle']:.6f}, "
            f"MAE={metrics['mae']:.6f}, WAPE={metrics['wape']:.6f}"
        )
        del model, prediction, actual, fold_oof
        gc.collect()

    scores = scores.sort_values("fold", kind="stable").reset_index(drop=True)
    oof = oof.sort_values(
        ["fold", "date", "store_nbr", "family"], kind="stable"
    ).reset_index(drop=True)
    if len(scores) != N_FOLDS or scores["fold"].nunique() != N_FOLDS:
        raise RuntimeError("recursive score table must contain exactly four folds")
    _validate_oof(oof)
    return scores, oof


def write_comparison(scores: pd.DataFrame, oof: pd.DataFrame) -> None:
    """Compare recursive results with preserved legacy and baseline evidence."""
    legacy = pd.read_csv(LEGACY_SCORES_PATH)
    baselines = pd.read_csv(BASELINE_SUMMARY_PATH).sort_values(
        "rmsle_mean", kind="stable"
    ).reset_index(drop=True)
    baseline = baselines.iloc[0]
    comparison = legacy[["fold", "rmsle"]].rename(
        columns={"rmsle": "old_strategy_rmsle"}
    ).merge(
        scores[["fold", "rmsle"]].rename(
            columns={"rmsle": "new_recursive_rmsle"}
        ),
        on="fold",
        validate="one_to_one",
    )
    comparison["recursive_minus_previous"] = (
        comparison["new_recursive_rmsle"] - comparison["old_strategy_rmsle"]
    )
    means = scores[["rmsle", "mae", "wape"]].agg(["mean", "std"])
    mean_rmsle = float(means.loc["mean", "rmsle"])
    std_rmsle = float(means.loc["std", "rmsle"])
    beats_baseline = mean_rmsle < float(baseline["rmsle_mean"])
    fold_4 = scores.loc[scores["fold"].eq(4)].iloc[0]

    table_lines = [
        "| fold | old_strategy_rmsle | new_recursive_rmsle | recursive_minus_previous |",
        "| ---: | ---: | ---: | ---: |",
    ]
    table_lines.extend(
        f"| {int(row.fold)} | {row.old_strategy_rmsle:.6f} | "
        f"{row.new_recursive_rmsle:.6f} | {row.recursive_minus_previous:.6f} |"
        for row in comparison.itertuples(index=False)
    )
    baseline_lines = [
        "| model | folds | rmsle_mean | rmsle_std | mae_mean | wape_mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    baseline_lines.extend(
        f"| {row.model} | {int(row.fold_count)} | {row.rmsle_mean:.6f} | "
        f"{row.rmsle_std:.6f} | {row.mae_mean:.6f} | {row.wape_mean:.6f} |"
        for row in baselines.itertuples(index=False)
    )
    lines = [
        "# Recursive Global LightGBM versus Previous Strategy",
        "",
        "## Evaluation contract",
        "",
        "The base untuned global LightGBM uses the unchanged M6 feature list, 250 "
        "boosting rounds, and the repository's fixed default parameters. Every "
        "16-day fold uses shared recursive inference. No tuning or final-test data "
        "is used. Legacy evidence is preserved separately.",
        "",
        "Baselines were verified rather than rerun: they cut all targets after the "
        "origin, use calendar-date references, and cannot consume validation actuals. "
        "The four June-August folds contain no missing Christmas closure date.",
        "",
        "## Verified baseline leaderboard",
        "",
        *baseline_lines,
        "",
        "## Fold comparison",
        "",
        *table_lines,
        "",
        "## Recursive summary",
        "",
        f"- OOF rows: **{len(oof):,}** ({EXPECTED_FOLD_ROWS:,} per fold).",
        f"- Mean RMSLE: **{mean_rmsle:.6f}**; fold std: **{std_rmsle:.6f}**.",
        f"- Mean MAE: **{means.loc['mean', 'mae']:.6f}**; std: **{means.loc['std', 'mae']:.6f}**.",
        f"- Mean WAPE: **{means.loc['mean', 'wape']:.6f}**; std: **{means.loc['std', 'wape']:.6f}**.",
        f"- Strongest baseline: **{baseline['model']}**, mean RMSLE **{baseline['rmsle_mean']:.6f} +/- {baseline['rmsle_std']:.6f}**.",
        f"- Recursive model {'beats' if beats_baseline else 'does not beat'} the strongest baseline by mean RMSLE.",
        f"- Fold 4: RMSLE **{fold_4['rmsle']:.6f}**, MAE **{fold_4['mae']:.6f}**, WAPE **{fold_4['wape']:.6f}**.",
        "- Predictions are complete, finite, non-missing, and nonnegative.",
        "",
        "## Methodology and recommendation",
        "",
        "Correct recursive semantics are retained regardless of whether the metric "
        "improves. The next step is controlled feature ablation under this same "
        "recursive contract. Do not reuse the old tuning selection or tune parameters "
        "until recursive ablation evidence has been regenerated.",
        "",
    ]
    COMPARISON_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scores, oof = run_recursive_backtest()
    write_comparison(scores, oof)
    print(scores.to_string(index=False))
    print(f"Scores: {SCORES_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"OOF: {OOF_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Comparison: {COMPARISON_PATH.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
