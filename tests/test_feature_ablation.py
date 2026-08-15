from pathlib import Path

import pandas as pd

from src.modeling.ablation import (
    ADDED_GROUP,
    EXPERIMENT_FEATURES,
    recommended_experiment,
    summarize_ablation,
)
from src.modeling.train_global import FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_experiment_feature_sets_are_strictly_cumulative() -> None:
    experiments = list(EXPERIMENT_FEATURES)
    assert experiments == [
        "M1", "M2", "M3", "M4", "M5", "M6", "M6_NO_HOLIDAY"
    ]
    for previous, current in zip(experiments[:5], experiments[1:6]):
        assert set(EXPERIMENT_FEATURES[previous]) < set(EXPERIMENT_FEATURES[current])
    assert EXPERIMENT_FEATURES["M6"] == FEATURE_COLUMNS
    assert set(EXPERIMENT_FEATURES["M6_NO_HOLIDAY"]) < set(FEATURE_COLUMNS)
    assert "M7" not in EXPERIMENT_FEATURES
    assert ADDED_GROUP["M7"] == "oil features"


def test_effect_classification_and_recommendation_use_validation_rmsle() -> None:
    rows = []
    means = {"M0": 0.50, "M1": 0.45, "M2": 0.4505, "M3": 0.46}
    for experiment, mean in means.items():
        for fold in range(1, 5):
            rows.append(
                {
                    "experiment": experiment,
                    "model": "baseline" if experiment == "M0" else "global_lightgbm",
                    "added_group": ADDED_GROUP[experiment],
                    "fold": fold,
                    "rmsle": mean,
                    "mae": mean * 100,
                    "wape": mean / 2,
                }
            )
    summary = summarize_ablation(pd.DataFrame(rows)).set_index("experiment")

    assert summary.loc["M0", "effect"] == "reference"
    assert summary.loc["M1", "effect"] == "improved"
    assert summary.loc["M2", "effect"] == "negligible effect"
    assert summary.loc["M3", "effect"] == "degraded"
    assert recommended_experiment(summary.reset_index())["experiment"] == "M1"


def test_ablation_runner_has_no_test_target_tuning_or_oil_experiment() -> None:
    source = (PROJECT_ROOT / "notebooks" / "15_feature_ablation.py").read_text(encoding="utf-8")

    assert "load_test" not in source
    assert "hyperparameter tuning" in source
    assert "EXPERIMENT_FEATURES" in source
    assert '"M7":' not in source
    assert "M7 not run" in source


def test_ablation_artifacts_cover_identical_four_folds_and_match_controls() -> None:
    report_dir = PROJECT_ROOT / "reports" / "modeling"
    scores = pd.read_csv(
        report_dir / "ablation_scores.csv",
        parse_dates=["train_end", "validation_start", "validation_end"],
    )
    assert len(scores) == 32
    assert set(scores["experiment"]) == {
        "M0", "M1", "M2", "M3", "M4", "M5", "M6", "M6_NO_HOLIDAY"
    }
    assert scores.groupby("experiment")["fold"].nunique().eq(4).all()
    boundaries = scores.groupby("experiment")[
        ["train_end", "validation_start", "validation_end"]
    ].apply(lambda frame: tuple(map(tuple, frame.to_numpy())))
    assert boundaries.nunique() == 1
    assert scores[["rmsle", "mae", "wape"]].notna().all().all()

    baseline = pd.read_csv(report_dir / "baseline_scores.csv")
    baseline = baseline.loc[baseline["model"].eq("rolling_historical_median_28d")]
    m0 = scores.loc[scores["experiment"].eq("M0")]
    pd.testing.assert_series_equal(
        m0["rmsle"].reset_index(drop=True),
        baseline["rmsle"].reset_index(drop=True),
        check_names=False,
    )
    full_model = pd.read_csv(report_dir / "recursive_backtest_scores.csv")
    m6 = scores.loc[scores["experiment"].eq("M6")]
    pd.testing.assert_series_equal(
        m6["rmsle"].reset_index(drop=True),
        full_model["rmsle"].reset_index(drop=True),
        check_names=False,
    )


def test_ablation_report_matches_fold_scores_and_excludes_unsupported_groups() -> None:
    report_dir = PROJECT_ROOT / "reports" / "modeling"
    scores = pd.read_csv(report_dir / "ablation_scores.csv")
    summary = summarize_ablation(scores).set_index("experiment")
    report = (report_dir / "ablation_summary.md").read_text(encoding="utf-8")

    assert summary.loc["M1", "effect"] == "improved"
    assert summary.loc["M2", "effect"] == "improved"
    assert summary.loc["M3", "effect"] == "negligible effect"
    assert summary.loc["M4", "effect"] == "improved"
    assert summary.loc["M5", "effect"] == "improved"
    assert summary.loc["M6", "effect"] == "improved"
    assert recommended_experiment(summary.reset_index())["experiment"] == "M6_NO_HOLIDAY"
    assert "M7): not run" in report
    assert "M6 holiday removal decision" in report
    assert "M6_NO_HOLIDAY" in report
