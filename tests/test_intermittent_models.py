import numpy as np
import pandas as pd
import pytest

from src.modeling.intermittent import (
    ROUTING_MINIMUM_RMSLE_IMPROVEMENT,
    forecast_intermittent_baseline,
    predict_two_stage,
    summarize_intermittent_scores,
    train_two_stage_models,
)
from src.modeling.recursive import recursive_forecast
from src.modeling.train_global import (
    FEATURE_COLUMNS,
    add_known_features,
    build_causal_training_features,
)


def test_croston_sba_and_tsb_known_values() -> None:
    history = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5),
            "store_nbr": 1,
            "family": "A",
            "sales": [0.0, 2.0, 0.0, 0.0, 4.0],
        }
    )
    series = pd.DataFrame({"store_nbr": [1], "family": ["A"]})
    common = (history, ["2024-01-06", "2024-01-07"], "2024-01-05")

    croston = forecast_intermittent_baseline(*common, method="croston", series=series)
    sba = forecast_intermittent_baseline(*common, method="sba", series=series)
    tsb = forecast_intermittent_baseline(*common, method="tsb", series=series)

    assert croston["prediction"].eq(2.2 / 2.1).all()
    assert sba["prediction"].eq(0.95 * 2.2 / 2.1).all()
    assert tsb["prediction"].eq(0.4645 * 2.2).all()


@pytest.mark.parametrize("method", ["croston", "sba", "tsb"])
def test_intermittent_baselines_are_horizon_safe_grouped_and_nonnegative(method: str) -> None:
    history = pd.DataFrame(
        {
            "date": list(pd.date_range("2024-01-01", periods=8)) * 2,
            "store_nbr": [1] * 8 + [2] * 8,
            "family": ["A"] * 8 + ["B"] * 8,
            "sales": [0, 2, 0, 0, 3, 0, 0, 1] + [0, 20, 0, 0, 30, 0, 0, 10],
        }
    )
    future = pd.DataFrame(
        {
            "date": list(pd.date_range("2024-01-09", periods=3)) * 2,
            "store_nbr": [1] * 3 + [2] * 3,
            "family": ["A"] * 3 + ["B"] * 3,
            "sales": 999_999.0,
        }
    )
    series = history[["store_nbr", "family"]].drop_duplicates()
    dates = pd.date_range("2024-01-09", periods=3)
    clean = forecast_intermittent_baseline(history, dates, "2024-01-08", method=method, series=series)
    contaminated = forecast_intermittent_baseline(
        pd.concat([history, future]), dates, "2024-01-08", method=method, series=series
    )

    pd.testing.assert_frame_equal(clean, contaminated)
    assert len(clean) == 6
    assert not clean.duplicated(["date", "store_nbr", "family"]).any()
    assert clean["prediction"].ge(0).all()
    store_one = clean.loc[clean["store_nbr"].eq(1), "prediction"].iloc[0]
    store_two = clean.loc[clean["store_nbr"].eq(2), "prediction"].iloc[0]
    assert store_two == pytest.approx(store_one * 10)


def _ml_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for store_nbr, family in [(1, "A"), (2, "B")]:
        for day, date in enumerate(pd.date_range("2023-01-01", periods=400)):
            rows.append(
                {
                    "date": date,
                    "store_nbr": store_nbr,
                    "family": family,
                    "sales": 0.0 if day % 3 else float(day % 10 + 1),
                    "onpromotion": day % 2,
                }
            )
    stores = pd.DataFrame(
        {"store_nbr": [1, 2], "city": ["Q", "C"], "state": ["P", "A"], "type": ["D", "B"], "cluster": [1, 2]}
    )
    holidays = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-01"]), "type": ["Holiday"], "locale": ["National"],
         "locale_name": ["Ecuador"], "description": ["Event"], "transferred": [False]}
    )
    return pd.DataFrame(rows), stores, holidays


def test_two_stage_global_prediction_is_complete_and_nonnegative() -> None:
    sales, stores, holidays = _ml_inputs()
    known = add_known_features(sales, stores, holidays)
    causal = build_causal_training_features(known)
    origin = pd.Timestamp("2024-01-19")
    parameters = {
        "learning_rate": 0.05, "num_leaves": 8, "verbosity": -1,
        "seed": 42, "num_threads": 1,
    }
    occurrence, magnitude = train_two_stage_models(
        causal, origin, parameters=parameters, num_boost_round=5, feature_columns=FEATURE_COLUMNS
    )
    prediction = recursive_forecast(
        (occurrence, magnitude),
        known,
        origin,
        origin + pd.Timedelta(days=1),
        origin + pd.Timedelta(days=16),
        prediction_function=lambda models, features: predict_two_stage(
            models[0], models[1], features
        ),
    )

    assert len(prediction) == 32
    assert prediction["prediction"].ge(0).all()
    assert np.isfinite(prediction["prediction"]).all()
    assert not prediction.duplicated(["date", "store_nbr", "family"]).any()


def test_routing_requires_four_fold_mean_rmsle_improvement() -> None:
    rows = []
    means = {"global_lightgbm_tuned": 0.55, "croston": 0.54, "two_stage_lightgbm": 0.5495}
    for model, mean in means.items():
        for fold in range(1, 5):
            rows.append({"model": model, "fold": fold, "rmsle": mean, "mae": 1.0, "wape": 0.2})
    summary = summarize_intermittent_scores(pd.DataFrame(rows)).set_index("model")

    assert summary.loc["croston", "routing_eligible"]
    assert not summary.loc["two_stage_lightgbm", "routing_eligible"]
    assert ROUTING_MINIMUM_RMSLE_IMPROVEMENT == 0.001
