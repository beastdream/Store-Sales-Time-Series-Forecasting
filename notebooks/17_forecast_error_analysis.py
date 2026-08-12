# %% [markdown]
# # Detailed Out-of-Fold Forecast Error Analysis
#
# OOF predictions are reproduced with the chosen tuned configuration and the
# original four temporal folds. ForecastReadiness is loaded only after all model
# predictions have been created; it is a post-hoc segmentation label and never a
# training feature.

# %%
import gc
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import REPORTS_DIR, TABLES_DIR
from src.data.load_raw import load_holidays, load_stores, load_train
from src.modeling.error_analysis import (
    attach_readiness_labels,
    score_failure_flags,
    score_segments,
    validate_oof_predictions,
)
from src.modeling.predict import predict_sales
from src.modeling.splits import make_rolling_splits
from src.modeling.train_global import (
    add_known_features,
    build_causal_training_features,
    build_horizon_safe_features,
    train_global_model,
)


REPORT_DIR = REPORTS_DIR / "modeling"
FIGURE_DIR = REPORTS_DIR / "figures" / "modeling"
CONFIG_PATH = PROJECT_ROOT / "models" / "global_lightgbm_chosen_config.json"
OOF_PATH = REPORT_DIR / "global_lgbm_tuned_oof_predictions.parquet"
STORE_PATH = REPORT_DIR / "scores_by_store.csv"
FAMILY_PATH = REPORT_DIR / "scores_by_family.csv"
READINESS_PATH = REPORT_DIR / "scores_by_readiness.csv"
REPORT_PATH = REPORT_DIR / "error_analysis.md"
HORIZON_DAYS = 16
N_FOLDS = 4


def reproduce_oof_predictions() -> pd.DataFrame:
    """Recreate row-level OOF predictions without accepting readiness inputs."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    train = load_train()
    known = add_known_features(train, load_stores(), load_holidays())
    causal = build_causal_training_features(known)
    splits = make_rolling_splits(train["date"].max(), HORIZON_DAYS, N_FOLDS)
    expected_rows = train[["store_nbr", "family"]].drop_duplicates().shape[0] * HORIZON_DAYS
    rows: list[pd.DataFrame] = []

    for fold, split in enumerate(splits, start=1):
        horizon = build_horizon_safe_features(
            known, split.train_end, split.validation_start, split.validation_end
        )
        model, _ = train_global_model(
            causal,
            split.train_end,
            parameters=config["parameters"],
            num_boost_round=config["num_boost_round"],
            feature_columns=config["feature_list"],
        )
        predictions = predict_sales(model, horizon)
        oof_fold = horizon[
            ["date", "store_nbr", "family", "store_type", "promotion_active", "is_holiday", "sales"]
        ].merge(
            predictions,
            on=["date", "store_nbr", "family"],
            validate="one_to_one",
        )
        if len(oof_fold) != expected_rows:
            raise AssertionError(f"fold {fold}: incomplete OOF prediction grain")
        oof_fold.insert(0, "fold", fold)
        rows.append(oof_fold)
        print(f"OOF fold {fold}/{N_FOLDS}: {len(oof_fold):,} predictions")
        del model, horizon, predictions
        gc.collect()
    result = pd.concat(rows, ignore_index=True)
    validate_oof_predictions(result)
    return result


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include="number"):
        display[column] = display[column].map(lambda value: f"{value:.6f}" if pd.notna(value) else "NA")
    headers = [str(column) for column in display]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in display.itertuples(index=False, name=None))
    return "\n".join(lines)


def make_plots(family_scores: pd.DataFrame, readiness_scores: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    family_plot = pd.concat(
        [family_scores.nsmallest(10, "rmsle"), family_scores.nlargest(10, "rmsle")]
    ).drop_duplicates("family").sort_values("rmsle")
    fig, axis = plt.subplots(figsize=(10, 8))
    axis.barh(family_plot["family"], family_plot["rmsle"], color="#4472C4")
    axis.set(title="Best and worst family OOF RMSLE", xlabel="RMSLE", ylabel="Family")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "oof_family_rmsle.png", dpi=160)
    plt.close(fig)

    ordered = readiness_scores.sort_values("rmsle")
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.barh(ordered["readiness_class"], ordered["rmsle"], color="#ED7D31")
    axis.set(title="OOF RMSLE by post-hoc readiness class", xlabel="RMSLE", ylabel="Readiness class")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "oof_readiness_rmsle.png", dpi=160)
    plt.close(fig)


def write_report(oof: pd.DataFrame, labeled: pd.DataFrame, store_scores: pd.DataFrame, family_scores: pd.DataFrame, readiness_scores: pd.DataFrame) -> None:
    overall = score_segments(oof.assign(segment="All OOF rows"), ["segment"])
    store_type = score_segments(oof, ["store_type"]).sort_values("rmsle")
    promotion = score_segments(oof, ["promotion_active"]).sort_values("promotion_active")
    holiday = score_segments(oof, ["is_holiday"]).sort_values("is_holiday")
    flags = score_failure_flags(labeled)
    active_flags = flags.loc[flags["flag_active"].eq(1)].sort_values("rmsle", ascending=False)
    best_stores = store_scores.nsmallest(5, "rmsle")[["store_nbr", "store_type", "rmsle", "mae", "wape"]]
    worst_stores = store_scores.nlargest(5, "rmsle")[["store_nbr", "store_type", "rmsle", "mae", "wape"]]
    best_families = family_scores.nsmallest(5, "rmsle")[["family", "rmsle", "mae", "wape"]]
    worst_families = family_scores.nlargest(5, "rmsle")[["family", "rmsle", "mae", "wape"]]
    overall_rmsle = float(overall.iloc[0]["rmsle"])
    worst_readiness = readiness_scores.loc[readiness_scores["rmsle"].idxmax()]
    best_readiness = readiness_scores.loc[readiness_scores["rmsle"].idxmin()]
    largest_flag = active_flags.iloc[0]
    promotion_class = readiness_scores.loc[
        readiness_scores["readiness_class"].eq("Promotion dependent")
    ].iloc[0]
    insufficient_class = readiness_scores.loc[
        readiness_scores["readiness_class"].eq("Insufficient history")
    ].iloc[0]

    evidence = [
        "# Tuned Global LightGBM OOF Error Analysis",
        "",
        "## Scope and leakage boundary",
        "",
        "These results use row-level predictions from four independently trained temporal folds. "
        "Each 16-day horizon is generated from one fixed origin. `ForecastReadiness` is joined "
        "only after prediction as a post-hoc label; none of its full-history statistics enters training.",
        "",
        "## Evidence",
        "",
        "### Overall",
        "",
        markdown_table(overall[["observation_count", "fold_count", "rmsle", "mae", "wape"]]),
        "",
        "### Store type",
        "",
        markdown_table(store_type[["store_type", "observation_count", "rmsle", "mae", "wape"]]),
        "",
        "### Promotion status",
        "",
        markdown_table(promotion[["promotion_active", "observation_count", "rmsle", "mae", "wape"]]),
        "",
        "### Holiday status",
        "",
        markdown_table(holiday[["is_holiday", "observation_count", "rmsle", "mae", "wape"]]),
        "",
        "### Readiness class (post-hoc only)",
        "",
        markdown_table(readiness_scores[["readiness_class", "series_count", "observation_count", "rmsle", "mae", "wape"]].sort_values("rmsle")),
        "",
        "### Overlapping readiness failure flags",
        "",
        markdown_table(active_flags[["risk_flag", "series_count", "observation_count", "rmsle", "mae", "wape"]]),
        "",
        "### Best and worst stores",
        "",
        "Best by RMSLE:", "", markdown_table(best_stores), "", "Worst by RMSLE:", "", markdown_table(worst_stores),
        "",
        "### Best and worst families",
        "",
        "Best by RMSLE:", "", markdown_table(best_families), "", "Worst by RMSLE:", "", markdown_table(worst_families),
        "",
        "## Evidence-based findings",
        "",
        f"- Overall pooled OOF RMSLE is **{overall_rmsle:.6f}** across {len(oof):,} predictions.",
        f"- The worst readiness class by RMSLE is **{worst_readiness['readiness_class']}** "
        f"at **{worst_readiness['rmsle']:.6f}** across {int(worst_readiness['series_count']):,} series.",
        f"- The best readiness class by RMSLE is **{best_readiness['readiness_class']}** "
        f"at **{best_readiness['rmsle']:.6f}**; class names alone therefore do not establish failure.",
        f"- The highest-error overlapping risk cohort is **{largest_flag['risk_flag']}** "
        f"at RMSLE **{largest_flag['rmsle']:.6f}**.",
        f"- **Promotion dependent** has low proportional error (RMSLE "
        f"**{promotion_class['rmsle']:.6f}**) but the largest readiness-class MAE "
        f"(**{promotion_class['mae']:.6f}**), consistent with its high sales volume; this is "
        "an absolute-error burden, not an RMSLE failure.",
        f"- **Insufficient history** is not an observed failure in these folds: RMSLE "
        f"**{insufficient_class['rmsle']:.6f}** and WAPE **{insufficient_class['wape']:.6f}** "
        "are the lowest among readiness classes.",
        f"- Holiday rows have RMSLE **{holiday.loc[holiday['is_holiday'].eq(1), 'rmsle'].iloc[0]:.6f}** "
        f"versus **{holiday.loc[holiday['is_holiday'].eq(0), 'rmsle'].iloc[0]:.6f}** on regular rows, "
        f"but include only {int(holiday.loc[holiday['is_holiday'].eq(1), 'observation_count'].iloc[0]):,} observations.",
        "- RMSLE, MAE and WAPE answer different questions: RMSLE emphasizes proportional/log-scale "
        "error, while MAE and WAPE are dominated more strongly by high-volume segments.",
        "",
        "## Speculation and hypotheses to test",
        "",
        "- Larger intermittent-demand errors may reflect zero/nonzero occurrence difficulty. "
        "This is a hypothesis, not a causal conclusion.",
        "- Promotion-status gaps may reflect promotion intensity, assortment, or unmodeled timing; "
        "the post-hoc comparison does not estimate promotion effects.",
        "- Holiday gaps may be unstable because relatively few OOF rows are holidays and event "
        "types are heterogeneous.",
        "- High-volatility cohorts may benefit from robust objectives or uncertainty modeling, but "
        "their label was computed over full history and is diagnostic only.",
        "",
        "## Specialized-model recommendation",
        "",
        "Do **not** replace the global model solely from this segmentation. First run controlled, "
        "fold-identical experiments for the worst sufficiently large cohorts. A specialized model "
        "is warranted only if it improves cohort RMSLE without materially degrading pooled RMSLE, "
        "MAE or WAPE. Prioritize an intermittent-demand occurrence/size experiment and a "
        "high-volatility robust-loss or uncertainty experiment. There is no current evidence "
        "for an insufficient-history specialist. Keep "
        "ForecastReadiness labels outside training unless they are recomputed causally per fold.",
        "",
        "Plots: `reports/figures/modeling/oof_family_rmsle.png` and "
        "`reports/figures/modeling/oof_readiness_rmsle.png`.",
    ]
    REPORT_PATH.write_text("\n".join(evidence), encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if OOF_PATH.exists():
        oof = pd.read_parquet(OOF_PATH)
        validate_oof_predictions(oof)
        print(f"Reused {len(oof):,} cached OOF predictions")
    else:
        oof = reproduce_oof_predictions()
        oof.to_parquet(OOF_PATH, index=False)

    # POST-HOC ONLY: readiness is intentionally loaded after OOF predictions exist.
    readiness = pd.read_csv(TABLES_DIR / "forecast_readiness.csv")
    labeled = attach_readiness_labels(oof, readiness)
    store_scores = score_segments(labeled, ["store_nbr", "store_type"]).sort_values("rmsle")
    family_scores = score_segments(labeled, ["family"]).sort_values("rmsle")
    readiness_scores = score_segments(labeled, ["readiness_class"]).sort_values("rmsle")
    store_scores.to_csv(STORE_PATH, index=False)
    family_scores.to_csv(FAMILY_PATH, index=False)
    readiness_scores.to_csv(READINESS_PATH, index=False)
    make_plots(family_scores, readiness_scores)
    write_report(oof, labeled, store_scores, family_scores, readiness_scores)
    print(f"OOF rows: {len(oof):,}")
    print(readiness_scores[["readiness_class", "rmsle", "mae", "wape"]].to_string(index=False))


if __name__ == "__main__":
    main()
