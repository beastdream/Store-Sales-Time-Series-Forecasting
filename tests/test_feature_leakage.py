"""End-to-end temporal leakage contracts for the forecasting feature pipeline."""

from pathlib import Path

import pandas as pd

from src.data.clean_oil import clean_oil
from src.features.build_forecast_frame import build_forecast_frame
from src.features.lag_features import add_horizon_safe_sales_lags
from src.features.rolling_features import add_horizon_safe_sales_rolling_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = PROJECT_ROOT / "reports" / "modeling" / "feature_leakage_audit.md"
FORBIDDEN_FULL_HISTORY_FEATURES = {
    "transactions",
    "forecast_readiness",
    "readiness_class",
    "sales_anomaly",
    "anomaly_method",
    "dcoilwtico",
    "oil_price",
}


def _target_history() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "store_nbr": 1,
            "family": "A",
            "sales": range(1, 51),
        }
    )


def _canonical_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "store_nbr": [1, 1, 1],
            "family": ["A", "A", "A"],
            "sales": [0.0, pd.NA, 5.0],
            "onpromotion": [0, 0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "id": [100, 101],
            "date": pd.to_datetime(["2024-01-04", "2024-01-05"]),
            "store_nbr": [1, 1],
            "family": ["A", "A"],
            "onpromotion": [2, 0],
        }
    )
    stores = pd.DataFrame(
        {
            "store_nbr": [1],
            "city": ["Quito"],
            "state": ["Pichincha"],
            "type": ["A"],
            "cluster": [1],
        }
    )
    holidays = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-04"]),
            "type": ["Holiday"],
            "locale": ["National"],
            "locale_name": ["Ecuador"],
            "description": ["Known event"],
            "transferred": [False],
        }
    )
    return train, test, stores, holidays


def _horizon_features(frame: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    result = add_horizon_safe_sales_lags(frame, cutoff)
    return add_horizon_safe_sales_rolling_features(result, cutoff)


def test_origin_mask_keeps_d1_audit_independent_of_future_actuals() -> None:
    history = _target_history()
    cutoff = "2024-02-03"
    future = history["date"].gt(cutoff)
    contaminated = history.copy()
    contaminated.loc[future, "sales"] = 9_999_999

    clean = _horizon_features(history, cutoff)
    changed = _horizon_features(contaminated, cutoff)
    feature_columns = [
        column
        for column in clean
        if column.startswith("sales_lag_") or column.startswith("rolling_")
    ]
    first_future = history.index[future][0]
    pd.testing.assert_series_equal(
        clean.loc[first_future, feature_columns],
        changed.loc[first_future, feature_columns],
    )


def test_low_level_origin_mask_is_not_used_as_multistep_strategy() -> None:
    recursive_source = (
        PROJECT_ROOT / "src" / "modeling" / "recursive.py"
    ).read_text(encoding="utf-8")
    final_source = (
        PROJECT_ROOT / "src" / "modeling" / "final_forecast.py"
    ).read_text(encoding="utf-8")

    assert "recursive_forecast" in recursive_source
    assert "recursive_forecast(" in final_source
    assert "build_horizon_safe_features(" not in final_source


def test_test_rows_have_no_target_or_forbidden_full_history_features() -> None:
    frame = build_forecast_frame(*_canonical_inputs())
    test_rows = frame.loc[frame["is_future"].eq(1)]

    assert test_rows["sales"].isna().all()
    assert test_rows["sales_observed"].eq(0).all()
    assert FORBIDDEN_FULL_HISTORY_FEATURES.isdisjoint(frame.columns)


def test_calendar_promotion_holiday_and_static_features_do_not_depend_on_sales() -> None:
    train, test, stores, holidays = _canonical_inputs()
    changed_train = train.copy()
    changed_train["sales"] = [1_000_000.0, 2_000_000.0, 3_000_000.0]
    original = build_forecast_frame(train, test, stores, holidays)
    changed = build_forecast_frame(changed_train, test, stores, holidays)
    safe_columns = [
        "date", "store_nbr", "family", "day_of_week", "week_of_year",
        "month", "quarter", "year", "is_weekend", "is_month_start",
        "is_month_end", "is_payday", "onpromotion", "promotion_active",
        "holiday_count", "is_holiday", "is_work_day", "is_event",
        "store_type", "cluster", "city", "state",
    ]

    pd.testing.assert_frame_equal(original[safe_columns], changed[safe_columns])


def test_missing_sales_observation_is_distinct_from_observed_zero() -> None:
    frame = build_forecast_frame(*_canonical_inputs()).set_index("date")
    zero = frame.loc[pd.Timestamp("2024-01-01")]
    missing = frame.loc[pd.Timestamp("2024-01-02")]

    assert zero["sales"] == 0
    assert zero["sales_observed"] == 1
    assert pd.isna(missing["sales"])
    assert missing["sales_observed"] == 0


def test_current_oil_imputation_reads_future_values_and_stays_excluded() -> None:
    oil = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            "dcoilwtico": [10.0, 30.0],
        }
    )
    changed_oil = oil.copy()
    changed_oil.loc[1, "dcoilwtico"] = 300.0
    original = clean_oil(oil, "2024-01-01", "2024-01-03")
    changed = clean_oil(changed_oil, "2024-01-01", "2024-01-03")

    assert original.loc[1, "oil_price"] == 20.0
    assert changed.loc[1, "oil_price"] == 155.0
    assert original.loc[1, "oil_price"] != changed.loc[1, "oil_price"]
    frame = build_forecast_frame(*_canonical_inputs())
    assert {"oil_price", "dcoilwtico"}.isdisjoint(frame.columns)


def test_audit_report_records_all_feature_groups_and_decisions() -> None:
    report = AUDIT_PATH.read_text(encoding="utf-8")
    groups = [
        "Sales lags", "Rolling statistics", "Calendar", "Promotion",
        "Holiday/event", "Store metadata", "Family metadata", "Oil",
        "Transactions", "ForecastReadiness", "SalesAnomalies",
    ]

    assert all(group in report for group in groups)
    assert all(decision in report for decision in ["SAFE", "CONDITIONALLY SAFE", "UNSAFE"])
    assert "No-go gate" in report
