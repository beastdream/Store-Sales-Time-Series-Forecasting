import json
from pathlib import Path

import pandas as pd

from src.modeling.predict import load_model
from src.modeling.train_global import DEFAULT_NUM_BOOST_ROUND, FEATURE_COLUMNS
from src.modeling.tuning import (
    MINIMUM_RMSLE_IMPROVEMENT,
    SEARCH_CONFIGS,
    TUNABLE_PARAMETERS,
    chosen_result,
    resolved_parameters,
    summarize_tuning,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _scores(means: dict[str, float]) -> pd.DataFrame:
    rows = []
    for experiment in SEARCH_CONFIGS:
        for fold in range(1, 5):
            value = means[experiment] + fold * 0.00001
            rows.append(
                {
                    "experiment": experiment,
                    "fold": fold,
                    "rmsle": value,
                    "mae": value * 100,
                    "wape": value / 2,
                }
            )
    return pd.DataFrame(rows)


def test_search_is_small_reproducible_and_uses_only_allowed_parameters() -> None:
    assert list(SEARCH_CONFIGS)[0] == "T0_untuned"
    assert len(SEARCH_CONFIGS) == 4
    assert set().union(*(set(config) for config in SEARCH_CONFIGS.values())).issubset(
        TUNABLE_PARAMETERS
    )
    for config in SEARCH_CONFIGS.values():
        resolved = resolved_parameters(config)
        assert resolved["seed"] == 42
        assert resolved["feature_fraction_seed"] == 42
        assert resolved["bagging_seed"] == 42


def test_selection_requires_mean_four_fold_improvement_threshold() -> None:
    insufficient = {
        "T0_untuned": 0.410,
        "T1_compact_regularized": 0.4095,
        "T2_moderate_capacity": 0.420,
        "T3_robust_subsample": 0.430,
    }
    results = summarize_tuning(_scores(insufficient))
    assert chosen_result(results)["experiment"] == "T0_untuned"

    sufficient = dict(insufficient)
    sufficient["T2_moderate_capacity"] = 0.405
    results = summarize_tuning(_scores(sufficient))
    assert chosen_result(results)["experiment"] == "T2_moderate_capacity"
    assert chosen_result(results)["fold_count"] == 4


def test_runner_does_not_load_test_or_select_single_fold() -> None:
    source = (PROJECT_ROOT / "notebooks" / "16_global_lightgbm_tuning.py").read_text(
        encoding="utf-8"
    )

    assert "load_test" not in source
    assert "mean four-fold RMSLE" in source
    assert "selection_uses_single_best_fold" in source
    assert "num_boost_round=DEFAULT_NUM_BOOST_ROUND" in source
    assert DEFAULT_NUM_BOOST_ROUND == 250
    assert "feature_columns=FEATURE_COLUMNS" in source


def test_completed_tuning_artifacts_record_all_experiments_and_selection() -> None:
    results_path = PROJECT_ROOT / "reports" / "modeling" / "tuning_results.csv"
    config_path = PROJECT_ROOT / "models" / "global_lightgbm_chosen_config.json"
    assert results_path.is_file()
    assert config_path.is_file()

    results = pd.read_csv(results_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert results["experiment"].tolist() == list(SEARCH_CONFIGS)
    assert results["fold_count"].eq(4).all()
    assert results["is_chosen"].sum() == 1
    assert results[["rmsle_mean", "rmsle_std", "mae_mean", "wape_mean"]].notna().all().all()
    assert all(column in results for column in TUNABLE_PARAMETERS)
    assert config["chosen_experiment"] == chosen_result(results)["experiment"]
    assert config["feature_list"] == FEATURE_COLUMNS
    assert config["num_boost_round"] == DEFAULT_NUM_BOOST_ROUND
    assert config["final_test_used_for_selection"] is False
    assert config["selection_uses_single_best_fold"] is False
    assert len(config["temporal_folds"]) == 4
    assert config["minimum_required_rmsle_improvement"] == MINIMUM_RMSLE_IMPROVEMENT

    fold_scores = pd.read_csv(
        PROJECT_ROOT / "reports" / "modeling" / "tuning_fold_scores.csv",
        parse_dates=["train_end", "validation_start", "validation_end"],
    )
    assert len(fold_scores) == len(SEARCH_CONFIGS) * 4
    boundaries = fold_scores.groupby("experiment")[
        ["train_end", "validation_start", "validation_end"]
    ].apply(lambda frame: tuple(map(tuple, frame.sort_values("validation_start").to_numpy())))
    assert boundaries.nunique() == 1
    recomputed = fold_scores.groupby("experiment")["rmsle"].mean()
    recorded = results.set_index("experiment")["rmsle_mean"]
    pd.testing.assert_series_equal(
        recomputed.sort_index(), recorded.sort_index(), check_names=False
    )

    chosen = chosen_result(results)
    assert chosen["rmsle_improvement_vs_untuned"] >= MINIMUM_RMSLE_IMPROVEMENT
    tuned_model = PROJECT_ROOT / config["model_artifact"]
    assert tuned_model.is_file() and tuned_model.stat().st_size > 0
    assert load_model(tuned_model).num_trees() == DEFAULT_NUM_BOOST_ROUND
