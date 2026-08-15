# %% [markdown]
# # Horizon-Safe Temporal Backtesting of Statistical Baselines
#
# Every 16-day validation horizon is forecast from one cutoff. Actual targets
# inside that horizon are used only for scoring, never to construct predictions.
# No machine-learning model or final competition prediction is created here.

# %%
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_RAW, REPORTS_DIR
from src.modeling.baselines import BASELINE_MODELS, forecast_baseline
from src.modeling.metrics import mae, rmsle, wape
from src.modeling.splits import make_rolling_splits


MODELING_REPORTS_DIR = REPORTS_DIR / "modeling"
SCORES_PATH = MODELING_REPORTS_DIR / "baseline_scores.csv"
SUMMARY_PATH = MODELING_REPORTS_DIR / "baseline_summary.csv"
HORIZON_DAYS = 16
N_FOLDS = 4


def load_history() -> pd.DataFrame:
    """Load the historical target at its validated daily store-family grain."""
    history = pd.read_csv(
        DATA_RAW / "train.csv",
        usecols=["date", "store_nbr", "family", "sales"],
        parse_dates=["date"],
    )
    grain = ["date", "store_nbr", "family"]
    if history.duplicated(grain).any():
        raise AssertionError("historical sales grain is not unique")
    if history["sales"].isna().any() or history["sales"].lt(0).any():
        raise AssertionError("historical sales must be complete and nonnegative")
    return history


def run_backtest(history: pd.DataFrame) -> pd.DataFrame:
    """Score every baseline on rolling folds without horizon target updates."""
    last_actual_date = history["date"].max().normalize()
    splits = make_rolling_splits(
        last_actual_date, horizon=HORIZON_DAYS, n_folds=N_FOLDS
    )
    series = history[["store_nbr", "family"]].drop_duplicates()
    expected_rows = len(series) * HORIZON_DAYS
    rows: list[dict[str, object]] = []

    for fold_number, split in enumerate(splits, start=1):
        # This is the only target data supplied to a baseline for the fold.
        fold_history = history.loc[history["date"].le(split.train_end)]
        validation = history.loc[
            history["date"].between(
                split.validation_start, split.validation_end
            ),
            ["date", "store_nbr", "family", "sales"],
        ]
        forecast_dates = pd.date_range(
            split.validation_start, split.validation_end, freq="D"
        )
        if len(validation) != expected_rows:
            raise AssertionError(
                f"fold {fold_number}: validation rows do not match full forecast grain"
            )

        for model in BASELINE_MODELS:
            predictions = forecast_baseline(
                fold_history,
                forecast_dates,
                cutoff=split.train_end,
                model=model,
                series=series,
            )
            if len(predictions) != expected_rows:
                raise AssertionError(
                    f"fold {fold_number} {model}: incomplete predictions"
                )
            scored = validation.merge(
                predictions,
                on=["date", "store_nbr", "family"],
                how="left",
                validate="one_to_one",
            )
            if scored["prediction"].isna().any():
                raise AssertionError(
                    f"fold {fold_number} {model}: missing predictions"
                )
            rows.append(
                {
                    "model": model,
                    "fold": fold_number,
                    "train_end": split.train_end,
                    "validation_start": split.validation_start,
                    "validation_end": split.validation_end,
                    "rmsle": rmsle(scored["sales"], scored["prediction"]),
                    "mae": mae(scored["sales"], scored["prediction"]),
                    "wape": wape(scored["sales"], scored["prediction"]),
                }
            )
    scores = pd.DataFrame(rows)
    expected_score_rows = len(BASELINE_MODELS) * N_FOLDS
    if len(scores) != expected_score_rows:
        raise AssertionError("baseline score row count is incomplete")
    return scores


def summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Return mean and sample standard deviation across temporal folds."""
    summary = (
        scores.groupby("model", as_index=False, observed=True)
        .agg(
            fold_count=("fold", "nunique"),
            rmsle_mean=("rmsle", "mean"),
            rmsle_std=("rmsle", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            wape_mean=("wape", "mean"),
            wape_std=("wape", "std"),
        )
        .sort_values(["rmsle_mean", "mae_mean"], kind="stable")
        .reset_index(drop=True)
    )
    if not summary["fold_count"].eq(N_FOLDS).all():
        raise AssertionError("every baseline summary must contain all folds")
    return summary


def main() -> None:
    """Run baseline backtests and persist score artifacts only."""
    history = load_history()
    scores = run_backtest(history)
    summary = summarize_scores(scores)
    MODELING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(SCORES_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("Baseline temporal backtesting completed.")
    print(summary.to_string(index=False))
    print(f"Scores: {SCORES_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Summary: {SUMMARY_PATH.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
