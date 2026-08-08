"""Known-future promotion, store, and holiday features."""

import pandas as pd

from src.data.clean_holidays import clean_holidays


STORE_METADATA_COLUMNS = ["store_type", "cluster", "city", "state"]
HOLIDAY_FEATURE_COLUMNS = [
    "holiday_count",
    "holiday_descriptions",
    "holiday_types",
    "holiday_locales",
    "is_holiday",
    "is_work_day",
    "is_event",
]

# Oil is intentionally not joined to the initial frame. Its publication timing,
# missing-value policy, and horizon availability must be reviewed first.
EXOGENOUS_CANDIDATES_REQUIRING_REVIEW = {
    "oil": "availability and leakage review required before model use"
}


def add_store_metadata(frame: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Attach stable store attributes without changing the frame grain."""
    metadata = stores.copy()
    if "store_type" not in metadata and "type" in metadata:
        metadata = metadata.rename(columns={"type": "store_type"})
    required = ["store_nbr", *STORE_METADATA_COLUMNS]
    missing = [column for column in required if column not in metadata]
    if missing:
        raise KeyError(f"stores: required columns not found: {', '.join(missing)}")
    if metadata["store_nbr"].duplicated().any():
        raise ValueError("stores: store_nbr must be unique")
    result = frame.merge(
        metadata[required], on="store_nbr", how="left", validate="many_to_one"
    )
    if result[STORE_METADATA_COLUMNS].isna().any().any():
        raise ValueError("stores: metadata must map to every forecast-frame store")
    return result


def add_promotion_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add an indicator from the supplied, forecast-time promotion schedule."""
    if "onpromotion" not in frame:
        raise KeyError("frame: required column not found: onpromotion")
    result = frame.copy()
    result["promotion_active"] = result["onpromotion"].gt(0).astype("uint8")
    result.loc[result["onpromotion"].isna(), "promotion_active"] = pd.NA
    result["promotion_active"] = result["promotion_active"].astype("UInt8")
    return result


def add_holiday_features(
    frame: pd.DataFrame,
    holidays: pd.DataFrame,
    stores: pd.DataFrame,
) -> pd.DataFrame:
    """Attach forecast-time event-calendar features at date-store grain."""
    events = clean_holidays(holidays, stores)
    result = frame.merge(
        events,
        on=["date", "store_nbr"],
        how="left",
        validate="many_to_one",
    )
    result["holiday_count"] = result["holiday_count"].fillna(0).astype("uint16")
    for column in ("is_holiday", "is_work_day", "is_event"):
        result[column] = result[column].fillna(0).astype("uint8")
    for column in ("holiday_descriptions", "holiday_types", "holiday_locales"):
        result[column] = result[column].fillna("")
    return result


def validate_known_future_features(frame: pd.DataFrame) -> None:
    """Raise when required scheduled features are unavailable on test rows."""
    future = frame.loc[frame["is_future"].eq(1)]
    required = ["onpromotion", "promotion_active", *HOLIDAY_FEATURE_COLUMNS]
    missing_columns = [column for column in required if column not in frame]
    if missing_columns:
        raise KeyError(
            "frame: known-future columns not found: " + ", ".join(missing_columns)
        )
    if future[required].isna().any().any():
        raise ValueError("known-future features must be available for every test row")
