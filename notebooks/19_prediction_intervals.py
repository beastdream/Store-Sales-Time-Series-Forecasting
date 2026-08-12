# %% [markdown]
# # Temporally Calibrated Prediction Intervals
#
# P50 is the unchanged tuned LightGBM point forecast. For each validation fold,
# an independent 16-day calibration horizon immediately preceding that fold is
# forecast from an earlier origin. An 80% split-conformal radius is estimated on
# absolute log residuals and applied to P50 to create nonnegative P10/P90 bounds.
# Current-fold validation targets never enter their own interval calibration.

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
from src.modeling.error_analysis import attach_readiness_labels
from src.modeling.predict import predict_sales
from src.modeling.splits import make_rolling_splits
from src.modeling.train_global import (
    add_known_features,
    build_causal_training_features,
    build_horizon_safe_features,
    train_global_model,
)
from src.modeling.uncertainty import (
    NOMINAL_COVERAGE,
    build_prediction_intervals,
    conformal_log_radius,
    score_interval_segments,
)


REPORT_DIR = REPORTS_DIR / "modeling"
CONFIG_PATH = PROJECT_ROOT / "models" / "global_lightgbm_chosen_config.json"
POINT_OOF_PATH = REPORT_DIR / "global_lgbm_tuned_oof_predictions.parquet"
INTERVAL_OOF_PATH = REPORT_DIR / "global_lgbm_prediction_intervals.parquet"
SCORES_PATH = REPORT_DIR / "prediction_interval_scores.csv"
REPORT_PATH = REPORT_DIR / "prediction_intervals.md"
HORIZON_DAYS = 16
N_FOLDS = 4


def reproduce_temporal_calibration() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build fold intervals using only a prior calibration horizon per fold."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    train = load_train()
    known = add_known_features(train, load_stores(), load_holidays())
    causal = build_causal_training_features(known)
    point_oof = pd.read_parquet(POINT_OOF_PATH)
    splits = make_rolling_splits(train["date"].max(), HORIZON_DAYS, N_FOLDS)
    interval_rows: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, object]] = []

    for fold, split in enumerate(splits, start=1):
        calibration_end = split.train_end
        calibration_start = calibration_end - pd.Timedelta(days=HORIZON_DAYS - 1)
        calibration_origin = calibration_start - pd.Timedelta(days=1)
        calibration_frame = build_horizon_safe_features(
            known,
            calibration_origin,
            calibration_start,
            calibration_end,
        )
        model, _ = train_global_model(
            causal,
            calibration_origin,
            parameters=config["parameters"],
            num_boost_round=config["num_boost_round"],
            feature_columns=config["feature_list"],
        )
        calibration_prediction = predict_sales(model, calibration_frame)
        calibration_scored = calibration_frame[
            ["date", "store_nbr", "family", "sales"]
        ].merge(
            calibration_prediction,
            on=["date", "store_nbr", "family"],
            validate="one_to_one",
        )
        radius = conformal_log_radius(
            calibration_scored["sales"], calibration_scored["prediction"]
        )

        validation_point = point_oof.loc[point_oof["fold"].eq(fold)].copy()
        intervals = build_prediction_intervals(
            validation_point[["date", "store_nbr", "family", "prediction"]],
            radius,
        )
        fold_intervals = validation_point.drop(columns="prediction").merge(
            intervals,
            on=["date", "store_nbr", "family"],
            validate="one_to_one",
        )
        fold_intervals["calibration_origin"] = calibration_origin
        fold_intervals["calibration_start"] = calibration_start
        fold_intervals["calibration_end"] = calibration_end
        interval_rows.append(fold_intervals)
        calibration_rows.append(
            {
                "fold": fold,
                "calibration_origin": calibration_origin,
                "calibration_start": calibration_start,
                "calibration_end": calibration_end,
                "validation_start": split.validation_start,
                "validation_end": split.validation_end,
                "calibration_observation_count": len(calibration_scored),
                "calibration_log_radius": radius,
            }
        )
        print(
            f"Fold {fold}: calibration {calibration_start.date()}.."
            f"{calibration_end.date()}, log radius={radius:.6f}"
        )
        del model, calibration_frame, calibration_prediction, calibration_scored
        gc.collect()
    return pd.concat(interval_rows, ignore_index=True), pd.DataFrame(calibration_rows)


def build_segment_scores(intervals: pd.DataFrame) -> pd.DataFrame:
    """Score intervals overall and by post-hoc readiness/risk groups."""
    overall = score_interval_segments(
        intervals.assign(segment_type="overall", segment_value="all"),
        ["segment_type", "segment_value"],
    )
    fold_scores = score_interval_segments(intervals, ["fold"])
    fold_scores.insert(0, "segment_type", "overall_fold")
    fold_scores.insert(1, "segment_value", fold_scores["fold"].astype(str))
    fold_scores = fold_scores.drop(columns="fold")

    # POST-HOC ONLY: full-history readiness labels are attached after all intervals exist.
    readiness = pd.read_csv(TABLES_DIR / "forecast_readiness.csv")
    labeled = attach_readiness_labels(intervals, readiness)
    by_readiness = score_interval_segments(labeled, ["readiness_class"])
    by_readiness.insert(0, "segment_type", "readiness_class")
    by_readiness.insert(1, "segment_value", by_readiness["readiness_class"])
    by_readiness = by_readiness.drop(columns="readiness_class")

    risk_rows = []
    for flag, label in [
        ("is_high_volatility", "high_volatility"),
        ("is_intermittent", "intermittent"),
    ]:
        active = labeled.loc[labeled[flag].eq(1)].assign(
            segment_type="risk_cohort", segment_value=label
        )
        risk_rows.append(
            score_interval_segments(active, ["segment_type", "segment_value"])
        )
    return pd.concat([overall, fold_scores, by_readiness, *risk_rows], ignore_index=True)


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include="number"):
        display[column] = display[column].map(lambda value: f"{value:.6f}")
    headers = list(display.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in display.itertuples(index=False, name=None))
    return "\n".join(lines)


def write_report(scores: pd.DataFrame, calibration: pd.DataFrame) -> None:
    metric_columns = [
        "segment_value", "observation_count", "series_count", "fold_count",
        "empirical_coverage", "coverage_gap_vs_nominal", "mean_interval_width",
        "p10_pinball_loss", "p50_pinball_loss", "p90_pinball_loss",
        "mean_pinball_loss", "point_rmsle", "point_mae", "point_wape",
    ]
    overall = scores.loc[scores["segment_type"].eq("overall")].iloc[0]
    readiness = scores.loc[scores["segment_type"].eq("readiness_class")]
    risks = scores.loc[scores["segment_type"].eq("risk_cohort")]
    fold_scores = scores.loc[scores["segment_type"].eq("overall_fold")]
    widest = readiness.loc[readiness["mean_interval_width"].idxmax()]
    lowest_coverage = readiness.loc[readiness["empirical_coverage"].idxmin()]
    lines = [
        "# Prediction Interval Evaluation",
        "",
        "## Method and temporal contract",
        "",
        "P50 is the unchanged tuned LightGBM point forecast. P10/P90 use an 80% "
        "split-conformal interval on the log1p scale. For every validation fold, the "
        "calibration residuals come from a separate 16-day horizon ending before the "
        "validation horizon and forecast from an earlier origin. No current-fold target "
        "is used to calibrate its interval. The method is global and does not use readiness "
        "labels during calibration or prediction.",
        "",
        "The P10/P90 labels denote lower/upper bounds of a nominal 80% conformal interval; "
        "they are not independently trained conditional quantile models.",
        "",
        "## Calibration windows",
        "",
        markdown_table(calibration),
        "",
        "## Uncertainty calibration evidence",
        "",
        f"Nominal P10/P90 coverage is **{NOMINAL_COVERAGE:.0%}**. Pooled empirical coverage "
        f"is **{overall['empirical_coverage']:.2%}**, with mean interval width "
        f"**{overall['mean_interval_width']:.3f}** and mean three-quantile pinball loss "
        f"**{overall['mean_pinball_loss']:.6f}**.",
        "",
        "### Overall by temporal fold",
        "",
        markdown_table(fold_scores[metric_columns]),
        "",
        "### Readiness class (post-hoc only)",
        "",
        markdown_table(readiness[metric_columns].sort_values("empirical_coverage")),
        "",
        "### High-volatility and intermittent cohorts",
        "",
        markdown_table(risks[metric_columns]),
        "",
        "## Point accuracy versus uncertainty calibration",
        "",
        f"Point accuracy is unchanged by construction: pooled P50 RMSLE is "
        f"**{overall['point_rmsle']:.6f}**, MAE **{overall['point_mae']:.6f}**, and WAPE "
        f"**{overall['point_wape']:.6f}**. These metrics evaluate central predictions. "
        "Coverage, width and pinball loss evaluate interval calibration and sharpness; a "
        "wider interval can improve coverage without improving point accuracy.",
        "",
        "## Segment findings",
        "",
        f"- Lowest readiness coverage: **{lowest_coverage['segment_value']}** at "
        f"**{lowest_coverage['empirical_coverage']:.2%}**.",
        f"- Widest readiness interval: **{widest['segment_value']}**, mean width "
        f"**{widest['mean_interval_width']:.3f}**.",
        "- Full-history readiness labels are diagnostic only. Segment coverage differences "
        "do not authorize using those labels as model or calibration features.",
        "",
        "## Recommendation",
        "",
        "Keep the validated point model unchanged. Treat these intervals as an initial global "
        "calibration layer. Before production use, require stable near-80% coverage across future "
        "origins and investigate segment-conditional or adaptive conformal calibration only when "
        "implemented from origin-available information. Do not narrow intervals merely to improve "
        "sharpness if empirical coverage deteriorates.",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if INTERVAL_OOF_PATH.exists():
        intervals = pd.read_parquet(INTERVAL_OOF_PATH)
        calibration = (
            intervals.groupby("fold", as_index=False)
            .agg(
                calibration_origin=("calibration_origin", "first"),
                calibration_start=("calibration_start", "first"),
                calibration_end=("calibration_end", "first"),
                validation_start=("date", "min"),
                validation_end=("date", "max"),
                calibration_observation_count=("sales", "size"),
                calibration_log_radius=("calibration_log_radius", "first"),
            )
        )
        print(f"Reused {len(intervals):,} cached prediction intervals")
    else:
        intervals, calibration = reproduce_temporal_calibration()
        intervals.to_parquet(INTERVAL_OOF_PATH, index=False)
    scores = build_segment_scores(intervals)
    scores.to_csv(SCORES_PATH, index=False)
    write_report(scores, calibration)
    print(scores.loc[scores["segment_type"].isin(["overall", "risk_cohort"]), [
        "segment_type", "segment_value", "empirical_coverage", "mean_interval_width",
        "mean_pinball_loss", "point_rmsle",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
