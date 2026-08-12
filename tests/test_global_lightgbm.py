from pathlib import Path
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.modeling.evaluate import compare_with_baselines, score_predictions
from src.modeling.predict import load_model, predict_sales
from src.modeling.train_global import (
    FEATURE_COLUMNS,
    FORBIDDEN_FEATURES,
    MODEL_NAME,
    add_known_features,
    build_causal_training_features,
    build_horizon_safe_features,
    train_global_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2023-01-01", periods=400, freq="D")
    rows = []
    row_id = 1
    for store_nbr, family, offset in [(1, "A", 0.0), (2, "B", 10.0)]:
        for day, date in enumerate(dates):
            rows.append(
                {
                    "id": row_id,
                    "date": date,
                    "store_nbr": store_nbr,
                    "family": family,
                    "sales": float((day % 14) + offset),
                    "onpromotion": day % 3,
                }
            )
            row_id += 1
    stores = pd.DataFrame(
        {
            "store_nbr": [1, 2],
            "city": ["Quito", "Cuenca"],
            "state": ["Pichincha", "Azuay"],
            "type": ["A", "B"],
            "cluster": [1, 2],
        }
    )
    holidays = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "type": ["Holiday"],
            "locale": ["National"],
            "locale_name": ["Ecuador"],
            "description": ["New year"],
            "transferred": [False],
        }
    )
    return pd.DataFrame(rows), stores, holidays


def test_model_feature_contract_contains_only_audited_groups() -> None:
    assert FORBIDDEN_FEATURES.isdisjoint(FEATURE_COLUMNS)
    assert {"store_nbr", "family", "onpromotion", "is_holiday"}.issubset(FEATURE_COLUMNS)
    assert "sales_lag_364" in FEATURE_COLUMNS
    assert "rolling_mean_28" in FEATURE_COLUMNS


def test_horizon_features_are_invariant_to_validation_actuals() -> None:
    sales, stores, holidays = _inputs()
    known = add_known_features(sales, stores, holidays)
    origin = pd.Timestamp("2024-01-19")
    start = origin + pd.Timedelta(days=1)
    end = origin + pd.Timedelta(days=16)
    contaminated = known.copy()
    contaminated.loc[contaminated["date"].gt(origin), "sales"] = 99_999_999.0

    clean = build_horizon_safe_features(known, origin, start, end)
    changed = build_horizon_safe_features(contaminated, origin, start, end)
    target_features = [column for column in FEATURE_COLUMNS if "lag" in column or "rolling" in column]
    pd.testing.assert_frame_equal(clean[target_features], changed[target_features])
    assert clean[target_features].notna().all().all()
    first_series = clean.loc[clean["store_nbr"].eq(1)].sort_values("date")
    assert first_series["rolling_mean_7"].nunique() == 1


def test_global_model_uses_log_target_and_predicts_complete_nonnegative_grain() -> None:
    sales, stores, holidays = _inputs()
    known = add_known_features(sales, stores, holidays)
    features = build_causal_training_features(known)
    origin = pd.Timestamp("2024-01-19")
    horizon = build_horizon_safe_features(
        known, origin, origin + pd.Timedelta(days=1), origin + pd.Timedelta(days=16)
    )
    model, metadata = train_global_model(
        features,
        origin,
        parameters={"num_threads": 1},
        num_boost_round=5,
    )
    predictions = predict_sales(model, horizon)

    assert isinstance(model, lgb.Booster)
    assert metadata["model_name"] == MODEL_NAME
    assert metadata["target_transform"] == "log1p(sales)"
    assert metadata["hyperparameter_tuning"] is False
    assert metadata["training_cutoff"] == "2024-01-19"
    assert metadata["feature_list"] == FEATURE_COLUMNS
    assert len(predictions) == 32
    assert predictions["prediction"].ge(0).all()
    assert np.isfinite(predictions["prediction"]).all()
    assert not predictions.duplicated(["date", "store_nbr", "family"]).any()


def test_evaluation_and_baseline_comparison_report_acceptance_decision() -> None:
    actual = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "store_nbr": [1, 2],
            "family": ["A", "B"],
            "sales": [0.0, 10.0],
        }
    )
    predictions = actual.drop(columns="sales").assign(prediction=[0.0, 9.0])
    metrics = score_predictions(actual, predictions)
    candidate = pd.DataFrame(
        {"model": [MODEL_NAME], "fold_count": [4], "rmsle_mean": [0.4],
         "rmsle_std": [0.01], "mae_mean": [1.0], "mae_std": [0.1],
         "wape_mean": [0.1], "wape_std": [0.01]}
    )
    baseline = candidate.assign(model="baseline", rmsle_mean=0.5)
    comparison = compare_with_baselines(candidate, baseline)
    model_row = comparison.loc[comparison["model"].eq(MODEL_NAME)].iloc[0]

    assert metrics["rmsle"] >= 0
    assert bool(model_row["beats_strongest_baseline"])
    assert model_row["rmsle_rank"] == 1


def test_dependency_and_notebook_do_not_use_final_test_for_selection() -> None:
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    notebook = (PROJECT_ROOT / "notebooks" / "14_global_lightgbm.py").read_text(encoding="utf-8")

    assert requirements.count("lightgbm") == 1
    assert "scikit-learn" not in requirements
    assert "load_test" not in notebook
    assert "hyperparameter" not in notebook.lower()


def test_successful_training_artifacts_are_complete_and_beat_baseline() -> None:
    report_dir = PROJECT_ROOT / "reports" / "modeling"
    scores = pd.read_csv(report_dir / "global_lgbm_scores.csv")
    comparison = pd.read_csv(report_dir / "global_lgbm_vs_baselines.csv")
    importance = pd.read_csv(report_dir / "feature_importance.csv")
    model_path = PROJECT_ROOT / "models" / "global_lightgbm.txt"
    metadata_path = PROJECT_ROOT / "models" / "global_lightgbm_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert len(scores) == 4
    assert scores["fold"].tolist() == [1, 2, 3, 4]
    assert scores[["rmsle", "mae", "wape"]].notna().all().all()
    candidate = comparison.loc[comparison["model"].eq(MODEL_NAME)].iloc[0]
    assert candidate["fold_count"] == 4
    assert bool(candidate["beats_strongest_baseline"])
    assert candidate["rmsle_mean"] < candidate["strongest_baseline_rmsle"]
    assert importance["feature"].tolist() == importance.sort_values(
        "importance_gain", ascending=False, kind="stable"
    )["feature"].tolist()
    assert set(importance["feature"]) == set(FEATURE_COLUMNS)
    assert model_path.stat().st_size > 0
    assert metadata["training_cutoff"] == "2017-08-15"
    assert metadata["feature_list"] == FEATURE_COLUMNS
    assert metadata["target_transform"] == "log1p(sales)"
    assert len(metadata["metric_results"]) == 4
    assert metadata["beats_strongest_baseline"] is True

    loaded_model = load_model(model_path)
    assert loaded_model.num_trees() > 0
