"""Behavioral proof for recursive calendar-day multi-step forecasting."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modeling.recursive import recursive_forecast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAIN = ["date", "store_nbr", "family"]


def _dense_frame(series_count: int = 1) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=96, freq="D")
    rows = []
    for number in range(series_count):
        for day, date in enumerate(dates, start=1):
            rows.append(
                {
                    "date": date,
                    "store_nbr": number + 1,
                    "family": f"F{number}",
                    "sales": float(day + number * 100),
                    "sales_observed": 1,
                }
            )
    return pd.DataFrame(rows)


class RecordingPredictor:
    """Record daily features and predict lag_1 + 10 deterministically."""

    def __init__(self) -> None:
        self.feature_days: list[pd.DataFrame] = []

    def __call__(self, _: object, features: pd.DataFrame) -> pd.DataFrame:
        self.feature_days.append(features.copy(deep=True))
        result = features[GRAIN].copy()
        result["prediction"] = features["sales_lag_1"].fillna(0).to_numpy() + 10.0
        return result


def _run(
    frame: pd.DataFrame,
    recorder: RecordingPredictor | None = None,
) -> tuple[pd.DataFrame, RecordingPredictor]:
    origin = pd.Timestamp("2024-03-20")
    predictor = recorder or RecordingPredictor()
    prediction = recursive_forecast(
        object(),
        frame,
        origin,
        origin + pd.Timedelta(days=1),
        origin + pd.Timedelta(days=16),
        prediction_function=predictor,
    )
    return prediction, predictor


def test_d1_uses_last_historical_value_and_d2_uses_d1_prediction() -> None:
    frame = _dense_frame()
    prediction, recorder = _run(frame)

    assert recorder.feature_days[0]["sales_lag_1"].iloc[0] == 80.0
    assert prediction["prediction"].iloc[0] == 90.0
    assert recorder.feature_days[1]["sales_lag_1"].iloc[0] == 90.0


def test_d2_never_uses_actual_d1_and_future_target_changes_are_invariant() -> None:
    frame = _dense_frame()
    contaminated = frame.copy()
    contaminated.loc[contaminated["date"].gt("2024-03-20"), "sales"] = 999_999.0

    clean_prediction, clean_recorder = _run(frame)
    changed_prediction, changed_recorder = _run(contaminated)

    assert changed_recorder.feature_days[1]["sales_lag_1"].iloc[0] == 90.0
    assert changed_recorder.feature_days[1]["sales_lag_1"].iloc[0] != 999_999.0
    pd.testing.assert_frame_equal(clean_prediction, changed_prediction)
    for clean, changed in zip(
        clean_recorder.feature_days, changed_recorder.feature_days, strict=True
    ):
        feature_columns = [
            column
            for column in clean
            if column.startswith("sales_lag_") or column.startswith("rolling_")
        ]
        pd.testing.assert_frame_equal(
            clean[feature_columns].reset_index(drop=True),
            changed[feature_columns].reset_index(drop=True),
        )


def test_lag_7_is_calendar_day_lag_and_missing_is_not_zero() -> None:
    frame = _dense_frame()
    missing_date = pd.Timestamp("2024-03-14")
    missing = frame["date"].eq(missing_date)
    frame.loc[missing, "sales"] = np.nan
    frame.loc[missing, "sales_observed"] = 0
    original = frame.copy(deep=True)

    _, recorder = _run(frame)
    d1 = recorder.feature_days[0].iloc[0]

    assert pd.isna(d1["sales_lag_7"])
    assert d1["sales_lag_7"] != 0
    pd.testing.assert_frame_equal(frame, original)


def test_later_rolling_window_is_updated_with_prior_prediction() -> None:
    _, recorder = _run(_dense_frame())

    assert recorder.feature_days[0]["rolling_mean_7"].iloc[0] == pytest.approx(77.0)
    assert recorder.feature_days[1]["rolling_mean_7"].iloc[0] == pytest.approx(
        (75 + 76 + 77 + 78 + 79 + 80 + 90) / 7
    )


def test_recursive_output_covers_16_days_times_every_series() -> None:
    prediction, recorder = _run(_dense_frame(series_count=2))

    assert len(prediction) == 16 * 2
    assert prediction["date"].nunique() == 16
    assert not prediction.duplicated(GRAIN).any()
    assert len(recorder.feature_days) == 16
    assert all(len(day) == 2 for day in recorder.feature_days)


def test_sparse_calendar_input_is_rejected() -> None:
    sparse = _dense_frame().loc[lambda frame: ~frame["date"].eq("2024-02-01")]

    with pytest.raises(ValueError, match="dense calendar"):
        _run(sparse)


def test_backtest_and_final_inference_call_same_recursive_core() -> None:
    backtest = (PROJECT_ROOT / "notebooks" / "14_global_lightgbm.py").read_text(
        encoding="utf-8"
    )
    final_module = (
        PROJECT_ROOT / "src" / "modeling" / "final_forecast.py"
    ).read_text(encoding="utf-8")

    assert "from src.modeling.recursive import recursive_forecast" in backtest
    assert "from src.modeling.recursive import recursive_forecast" in final_module
    assert "recursive_forecast(" in backtest
    assert "recursive_forecast(" in final_module
    assert "build_horizon_safe_features(" not in backtest
    assert "build_horizon_safe_features(" not in final_module

    for notebook_name in [
        "15_feature_ablation.py",
        "16_global_lightgbm_tuning.py",
        "17_forecast_error_analysis.py",
        "18_intermittent_demand_models.py",
        "19_prediction_intervals.py",
    ]:
        source = (PROJECT_ROOT / "notebooks" / notebook_name).read_text(
            encoding="utf-8"
        )
        assert "from src.modeling.recursive import recursive_forecast" in source
        assert "recursive_forecast(" in source
        assert "build_horizon_safe_features(" not in source
