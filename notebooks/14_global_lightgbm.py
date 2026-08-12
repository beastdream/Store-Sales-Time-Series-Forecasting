# %% [markdown]
# # First Global LightGBM Forecasting Model
#
# One model is trained across every store-family series. The target is
# `log1p(sales)` and predictions are transformed with `expm1` then clipped at
# zero. Each 16-day fold is predicted recursively: an earlier prediction updates
# later lag and rolling features. This is an untuned initial configuration. The final Kaggle test set is
# never used for parameter selection or backtest evaluation.

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
from src.modeling.evaluate import (
    compare_with_baselines,
    score_predictions,
    summarize_model_scores,
)
from src.modeling.recursive import recursive_forecast
from src.modeling.splits import make_rolling_splits
from src.modeling.train_global import (
    DEFAULT_NUM_BOOST_ROUND,
    FEATURE_COLUMNS,
    MODEL_NAME,
    add_known_features,
    build_causal_training_features,
    train_global_model,
)


REPORT_DIR = REPORTS_DIR / "modeling"
SCORES_PATH = REPORT_DIR / "global_lgbm_scores.csv"
COMPARISON_PATH = REPORT_DIR / "global_lgbm_vs_baselines.csv"
IMPORTANCE_PATH = REPORT_DIR / "feature_importance.csv"
BASELINE_SUMMARY_PATH = REPORT_DIR / "baseline_summary.csv"
MODEL_PATH = MODELS_DIR / "global_lightgbm.txt"
METADATA_PATH = MODELS_DIR / "global_lightgbm_metadata.json"
HORIZON_DAYS = 16
N_FOLDS = 4


def run_backtest(
    known_frame: pd.DataFrame,
    causal_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and score one global model at each of the four fixed origins."""
    splits = make_rolling_splits(
        known_frame["date"].max(), horizon=HORIZON_DAYS, n_folds=N_FOLDS
    )
    expected_rows = known_frame[["store_nbr", "family"]].drop_duplicates().shape[0] * HORIZON_DAYS
    rows: list[dict[str, object]] = []
    fold_metadata: list[dict[str, object]] = []

    for fold, split in enumerate(splits, start=1):
        horizon = known_frame.loc[known_frame["date"].between(
            split.validation_start, split.validation_end
        )].copy()
        if len(horizon) != expected_rows:
            raise AssertionError(f"fold {fold}: incomplete validation horizon")
        model, metadata = train_global_model(causal_features, split.train_end)
        predictions = recursive_forecast(
            model, known_frame, split.train_end,
            split.validation_start, split.validation_end,
        )
        metrics = score_predictions(horizon, predictions)
        rows.append(
            {
                "model": MODEL_NAME,
                "fold": fold,
                "train_end": split.train_end,
                "validation_start": split.validation_start,
                "validation_end": split.validation_end,
                **metrics,
            }
        )
        metadata["fold"] = fold
        metadata["metrics"] = metrics
        fold_metadata.append(metadata)
        del model, horizon, predictions
        gc.collect()
        print(f"Fold {fold}/{N_FOLDS}: RMSLE={metrics['rmsle']:.6f}")

    return pd.DataFrame(rows), pd.DataFrame(fold_metadata)


def train_and_save_final(
    causal_features: pd.DataFrame,
    scores: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    """Train on all history and save only after the complete backtest succeeds."""
    cutoff = causal_features["date"].max().normalize()
    model, metadata = train_global_model(causal_features, cutoff)
    temporary_model = Path(tempfile.gettempdir()) / "store_sales_global_lightgbm.txt"
    model.save_model(str(temporary_model))
    shutil.move(str(temporary_model), MODEL_PATH)
    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_gain": model.feature_importance(importance_type="gain"),
            "importance_split": model.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False, kind="stable")
    importance.insert(0, "rank", range(1, len(importance) + 1))
    importance.to_csv(IMPORTANCE_PATH, index=False)

    candidate = comparison.loc[comparison["model"].eq(MODEL_NAME)].iloc[0]
    metadata["metric_results"] = json.loads(
        scores.to_json(orient="records", date_format="iso")
    )
    metadata["mean_metrics"] = {
        "rmsle": float(candidate["rmsle_mean"]),
        "mae": float(candidate["mae_mean"]),
        "wape": float(candidate["wape_mean"]),
    }
    metadata["strongest_baseline_rmsle"] = float(candidate["strongest_baseline_rmsle"])
    metadata["beats_strongest_baseline"] = bool(candidate["beats_strongest_baseline"])
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    """Execute untuned four-fold backtesting and persist successful artifacts."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train = load_train()
    known = add_known_features(train, load_stores(), load_holidays())
    causal = build_causal_training_features(known)
    scores, _ = run_backtest(known, causal)
    summary = summarize_model_scores(scores)
    baselines = pd.read_csv(BASELINE_SUMMARY_PATH)
    comparison = compare_with_baselines(summary, baselines)

    scores.to_csv(SCORES_PATH, index=False)
    comparison.to_csv(COMPARISON_PATH, index=False)
    train_and_save_final(causal, scores, comparison)

    candidate = comparison.loc[comparison["model"].eq(MODEL_NAME)].iloc[0]
    outcome = "BEAT" if candidate["beats_strongest_baseline"] else "DID NOT BEAT"
    print(comparison.to_string(index=False))
    print(
        f"Untuned {MODEL_NAME} {outcome} the strongest baseline by mean RMSLE: "
        f"{candidate['rmsle_mean']:.6f} vs {candidate['strongest_baseline_rmsle']:.6f}."
    )


if __name__ == "__main__":
    main()
