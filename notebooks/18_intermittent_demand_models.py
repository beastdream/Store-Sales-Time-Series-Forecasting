# %% [markdown]
# # Intermittent-Demand Forecasting Strategies
#
# ForecastReadiness identifies the post-hoc evaluation cohort only. It is never
# merged into training features. Croston-family baselines use pre-origin history;
# the two-stage strategy trains globally on all series, predicts recursively,
# and is evaluated on the
# intermittent cohort only after prediction.

# %%
import gc
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import REPORTS_DIR, TABLES_DIR
from src.data.load_raw import load_holidays, load_stores, load_train
from src.modeling.evaluate import score_predictions
from src.modeling.intermittent import (
    INTERMITTENT_SMOOTHING,
    ROUTING_MINIMUM_RMSLE_IMPROVEMENT,
    forecast_intermittent_baseline,
    predict_two_stage,
    summarize_intermittent_scores,
    train_two_stage_models,
)
from src.modeling.splits import make_rolling_splits
from src.modeling.recursive import recursive_forecast
from src.modeling.train_global import (
    add_known_features,
    build_causal_training_features,
)


REPORT_DIR = REPORTS_DIR / "modeling"
CONFIG_PATH = PROJECT_ROOT / "models" / "global_lightgbm_chosen_config.json"
GLOBAL_OOF_PATH = REPORT_DIR / "global_lgbm_tuned_oof_predictions.parquet"
TWO_STAGE_OOF_PATH = REPORT_DIR / "two_stage_intermittent_oof_predictions.parquet"
SCORES_PATH = REPORT_DIR / "intermittent_model_scores.csv"
REPORT_PATH = REPORT_DIR / "model_routing_analysis.md"
HORIZON_DAYS = 16
N_FOLDS = 4
INTERMITTENT_CLASS = "Intermittent demand"


def load_intermittent_series() -> pd.DataFrame:
    """Load only post-hoc routing keys, never readiness statistics as features."""
    readiness = pd.read_csv(
        TABLES_DIR / "forecast_readiness.csv",
        usecols=["store_nbr", "family", "readiness_class"],
    )
    series = readiness.loc[
        readiness["readiness_class"].eq(INTERMITTENT_CLASS),
        ["store_nbr", "family"],
    ].copy()
    if series.empty or series.duplicated(["store_nbr", "family"]).any():
        raise ValueError("intermittent evaluation series must be non-empty and unique")
    return series


def score_global_control(series: pd.DataFrame) -> list[dict[str, object]]:
    oof = pd.read_parquet(GLOBAL_OOF_PATH).merge(
        series, on=["store_nbr", "family"], how="inner", validate="many_to_one"
    )
    rows = []
    for fold, fold_oof in oof.groupby("fold", sort=True):
        metrics = score_predictions(
            fold_oof,
            fold_oof[["date", "store_nbr", "family", "prediction"]],
        )
        rows.append({"model": "global_lightgbm_tuned", "fold": fold, **metrics})
    return rows


def score_intermittent_baselines(
    train: pd.DataFrame,
    series: pd.DataFrame,
) -> list[dict[str, object]]:
    """Score fixed-alpha Croston variants on the identical temporal folds."""
    rows = []
    splits = make_rolling_splits(train["date"].max(), HORIZON_DAYS, N_FOLDS)
    for fold, split in enumerate(splits, start=1):
        actual = train.loc[
            train["date"].between(split.validation_start, split.validation_end)
        ].merge(series, on=["store_nbr", "family"], how="inner", validate="many_to_one")
        dates = pd.date_range(split.validation_start, split.validation_end)
        for method in ["croston", "sba", "tsb"]:
            prediction = forecast_intermittent_baseline(
                train,
                dates,
                split.train_end,
                method=method,
                series=series,
                alpha=INTERMITTENT_SMOOTHING,
                beta=INTERMITTENT_SMOOTHING,
            )
            metrics = score_predictions(actual, prediction)
            rows.append({"model": method, "fold": fold, **metrics})
            print(f"{method} fold {fold}: RMSLE={metrics['rmsle']:.6f}")
    return rows


def reproduce_two_stage_oof(
    train: pd.DataFrame,
    series: pd.DataFrame,
) -> pd.DataFrame:
    """Train globally, predict full horizons, then filter the evaluation cohort."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    known = add_known_features(train, load_stores(), load_holidays())
    causal = build_causal_training_features(known)
    splits = make_rolling_splits(train["date"].max(), HORIZON_DAYS, N_FOLDS)
    rows = []
    for fold, split in enumerate(splits, start=1):
        horizon = known.loc[known["date"].between(
            split.validation_start, split.validation_end
        )].copy()
        occurrence, magnitude = train_two_stage_models(
            causal,
            split.train_end,
            parameters=config["parameters"],
            num_boost_round=config["num_boost_round"],
            feature_columns=config["feature_list"],
        )
        prediction = recursive_forecast(
            (occurrence, magnitude),
            known,
            split.train_end,
            split.validation_start,
            split.validation_end,
            prediction_function=lambda models, features: predict_two_stage(
                models[0], models[1], features
            ),
        )
        intermittent = horizon[["date", "store_nbr", "family", "sales"]].merge(
            prediction,
            on=["date", "store_nbr", "family"],
            validate="one_to_one",
        ).merge(
            series,
            on=["store_nbr", "family"],
            how="inner",
            validate="many_to_one",
        )
        intermittent.insert(0, "fold", fold)
        rows.append(intermittent)
        print(f"two-stage fold {fold}: {len(intermittent):,} cohort predictions")
        del occurrence, magnitude, horizon, prediction
        gc.collect()
    return pd.concat(rows, ignore_index=True)


def score_two_stage(oof: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for fold, fold_oof in oof.groupby("fold", sort=True):
        metrics = score_predictions(
            fold_oof,
            fold_oof[["date", "store_nbr", "family", "prediction"]],
        )
        rows.append({"model": "two_stage_lightgbm", "fold": fold, **metrics})
    return rows


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include="number"):
        display[column] = display[column].map(lambda value: f"{value:.6f}")
    headers = list(display.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in display.itertuples(index=False, name=None))
    return "\n".join(lines)


def write_report(scores: pd.DataFrame, series_count: int) -> pd.DataFrame:
    summary = summarize_intermittent_scores(scores)
    global_row = summary.loc[summary["model"].eq("global_lightgbm_tuned")].iloc[0]
    best = summary.iloc[0]
    eligible = summary.loc[summary["routing_eligible"]]
    comparison = scores.pivot(index="fold", columns="model", values="rmsle")
    two_stage_fold_wins = int(
        comparison["two_stage_lightgbm"].lt(comparison["global_lightgbm_tuned"]).sum()
    )
    if eligible.empty:
        recommendation = (
            "Do **not** route intermittent series away from the tuned global model. "
            "No specialized strategy improved four-fold mean RMSLE by at least "
            f"{ROUTING_MINIMUM_RMSLE_IMPROVEMENT:.3f}."
        )
    else:
        routed = eligible.sort_values("rmsle_mean", kind="stable").iloc[0]
        recommendation = (
            f"Recommend a controlled validation/shadow routing experiment for the post-hoc "
            f"intermittent cohort using "
            f"**{routed['model']}**: mean RMSLE improves by "
            f"{routed['rmsle_improvement_vs_global']:.6f} versus the tuned global model. "
            "Do not deploy a router based directly on full-history readiness labels; first "
            "implement an origin-causal cohort rule. Keep the global model for all other "
            "series and monitor pooled metrics."
        )
    lines = [
        "# Intermittent-Demand Model Routing Analysis",
        "",
        "## Evaluation contract",
        "",
        f"The evaluation cohort contains {series_count:,} series labeled `Intermittent demand`. "
        "ForecastReadiness is used only to select and score this post-hoc cohort; it is not a "
        "training feature. All strategies use the same four rolling 16-day folds. Croston, SBA "
        f"and TSB use fixed alpha/beta {INTERMITTENT_SMOOTHING:.1f}; no smoothing tuning was performed. "
        "The two-stage model uses the already chosen LightGBM configuration without further tuning.",
        "",
        "## Evidence",
        "",
        markdown_table(summary[["model", "fold_count", "rmsle_mean", "rmsle_std", "mae_mean", "wape_mean", "rmsle_improvement_vs_global", "routing_eligible"]]),
        "",
        f"The lowest observed mean RMSLE is **{best['model']}** at **{best['rmsle_mean']:.6f}**. "
        f"The tuned global control is **{global_row['rmsle_mean']:.6f}**.",
        "",
        f"Two-stage LightGBM improves RMSLE in **{two_stage_fold_wins} of 4 folds**. Its mean "
        "advantage is concentrated in the final fold; folds 1 and 3 are approximately flat or "
        "slightly worse. MAE and WAPE nevertheless improve on their four-fold means.",
        "",
        "## Interpretation boundary",
        "",
        "The comparison establishes predictive performance on these temporal folds, not a causal "
        "explanation for intermittent demand. Readiness labels were computed on full history and "
        "therefore remain routing/evaluation labels only. Any production router must reproduce its "
        "cohort classification from origin-available history.",
        "",
        "## Routing recommendation",
        "",
        recommendation,
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    series = load_intermittent_series()
    train = load_train()
    rows = score_global_control(series)
    rows.extend(score_intermittent_baselines(train, series))
    if TWO_STAGE_OOF_PATH.exists():
        two_stage_oof = pd.read_parquet(TWO_STAGE_OOF_PATH)
        print(f"Reused {len(two_stage_oof):,} two-stage OOF predictions")
    else:
        two_stage_oof = reproduce_two_stage_oof(train, series)
        two_stage_oof.to_parquet(TWO_STAGE_OOF_PATH, index=False)
    rows.extend(score_two_stage(two_stage_oof))
    scores = pd.DataFrame(rows).sort_values(["model", "fold"], kind="stable")
    if len(scores) != 5 * N_FOLDS:
        raise AssertionError("intermittent strategy score matrix is incomplete")
    scores.to_csv(SCORES_PATH, index=False)
    summary = write_report(scores, len(series))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
