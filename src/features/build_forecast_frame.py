"""Build the canonical leakage-aware store-family-date modeling frame."""

import pandas as pd

from src.features.calendar_features import build_calendar_features
from src.features.exogenous_features import (
    add_holiday_features,
    add_promotion_features,
    add_store_metadata,
    validate_known_future_features,
)


GRAIN = ["date", "store_nbr", "family"]


def _require_columns(
    frame: pd.DataFrame, required: list[str], table_name: str
) -> None:
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"{table_name}: required columns not found: {', '.join(missing)}")


def _prepare_source_rows(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Timestamp]:
    _require_columns(
        train, ["id", "date", "store_nbr", "family", "sales", "onpromotion"], "train"
    )
    _require_columns(test, ["id", "date", "store_nbr", "family", "onpromotion"], "test")

    historical = train[["id", *GRAIN, "sales", "onpromotion"]].copy()
    future = test[["id", *GRAIN, "onpromotion"]].copy()
    historical["date"] = pd.to_datetime(historical["date"]).dt.normalize()
    future["date"] = pd.to_datetime(future["date"]).dt.normalize()
    historical["is_historical"] = 1
    historical["is_future"] = 0
    future["sales"] = pd.Series(float("nan"), index=future.index, dtype="float64")
    future["is_historical"] = 0
    future["is_future"] = 1

    if historical.duplicated(GRAIN).any() or future.duplicated(GRAIN).any():
        raise ValueError("train and test must each have a unique store-family-date grain")
    overlap = historical[GRAIN].merge(future[GRAIN], on=GRAIN, how="inner")
    if not overlap.empty:
        raise ValueError("train and test grains must not overlap")
    return pd.concat([historical, future], ignore_index=True), historical["date"].max()


def build_dense_known_frame(
    sales: pd.DataFrame,
    stores: pd.DataFrame,
    holidays: pd.DataFrame,
) -> pd.DataFrame:
    """Build the canonical dense calendar frame used by training and inference.

    Every store-family series receives one row for every calendar date between
    the input bounds. An absent source observation retains a null sale and
    sales_observed=0; it is never converted to an observed zero. Target lags and
    rolling features are deliberately added later from this same representation.
    """
    required = [*GRAIN, "sales", "onpromotion"]
    _require_columns(sales, required, "sales")
    source = sales.copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce").dt.normalize()
    if source.empty or source["date"].isna().any():
        raise ValueError("sales must contain valid dated rows")
    if source.duplicated(GRAIN).any():
        raise ValueError("sales must have a unique store-family-date grain")

    dates = pd.DataFrame(
        {"date": pd.date_range(source["date"].min(), source["date"].max(), freq="D")}
    )
    series = source[["store_nbr", "family"]].drop_duplicates()
    frame = dates.merge(series, how="cross", validate="many_to_many").merge(
        source,
        on=GRAIN,
        how="left",
        validate="one_to_one",
    )
    frame["sales_observed"] = frame["sales"].notna().astype("uint8")
    frame["source_row_observed"] = frame["onpromotion"].notna().astype("uint8")
    frame = frame.merge(
        build_calendar_features(frame["date"]),
        on="date",
        how="left",
        validate="many_to_one",
    )
    frame = add_store_metadata(frame, stores)
    frame = add_promotion_features(frame)
    frame = add_holiday_features(frame, holidays, stores)
    if frame.duplicated(GRAIN).any() or frame[GRAIN].isna().any().any():
        raise RuntimeError("dense forecast frame grain is invalid")
    return frame.sort_values(GRAIN, kind="stable").reset_index(drop=True)


def build_forecast_frame(
    train: pd.DataFrame,
    test: pd.DataFrame,
    stores: pd.DataFrame,
    holidays: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per store-family-date with safe, known-future features.

    Missing source observations are represented explicitly and retain null
    ``sales`` and ``onpromotion`` values. No target-derived, transaction, oil,
    lag, rolling, anomaly, or full-history readiness feature is added here.
    """
    source, historical_end = _prepare_source_rows(train, test)
    frame = build_dense_known_frame(source, stores, holidays)

    inferred_historical = frame["date"].le(historical_end)
    frame["is_historical"] = frame["is_historical"].fillna(inferred_historical).astype("uint8")
    frame["is_future"] = frame["is_future"].fillna(~inferred_historical).astype("uint8")
    frame["row_type"] = frame["is_future"].map({0: "historical", 1: "future"})
    frame["source_row_observed"] = frame["id"].notna().astype("uint8")
    frame["test_id"] = frame["id"].where(frame["is_future"].eq(1)).astype("UInt32")

    if frame.duplicated(GRAIN).any():
        raise RuntimeError("forecast frame grain is not unique")
    if frame[GRAIN].isna().any().any():
        raise RuntimeError("forecast frame grain contains missing values")
    observed_test_ids = frame.loc[frame["is_future"].eq(1) & frame["source_row_observed"].eq(1), "test_id"]
    if observed_test_ids.isna().any() or observed_test_ids.duplicated().any():
        raise RuntimeError("original test IDs were not preserved uniquely")
    validate_known_future_features(frame)
    return frame.sort_values(GRAIN, kind="stable").reset_index(drop=True)
