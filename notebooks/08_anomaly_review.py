# %% [markdown]
# # Sales Anomaly Review
#
# An anomaly is a descriptive review flag, not an automatic data-quality error.
# Every flagged observation remains in the source and in this review output.

# %%
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED, TABLES_DIR

OUTPUT_PATH = TABLES_DIR / "sales_anomalies.csv"
NOTES_PATH = TABLES_DIR / "sales_anomalies_notes.md"
ROLLING_WINDOW_DAYS = 28
ROLLING_MIN_OBSERVATIONS = 14
ROLLING_Z_THRESHOLD = 3.0
IQR_MULTIPLIER = 1.5
MATERIAL_PROMOTION_RATE = 0.25
POTENTIAL_ISSUE_Z_THRESHOLD = 6.0


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide numeric series, returning NaN for zero denominators."""
    result = numerator.astype("float64").div(
        denominator.astype("float64").replace(0, np.nan)
    )
    return result.where(np.isfinite(result))


def load_store_daily() -> pd.DataFrame:
    """Build one store-day row and attach context without multiplying sales."""
    fact = pd.read_parquet(
        DATA_PROCESSED / "fact_daily_sales.parquet",
        columns=["date_key", "store_key", "sales", "is_promotion"],
    )
    daily = (
        fact.assign(sales=fact["sales"].astype("float64"))
        .groupby(["date_key", "store_key"], as_index=False, observed=True)
        .agg(
            actual_sales=("sales", "sum"),
            is_promotion=("is_promotion", "max"),
            promotion_rate=("is_promotion", "mean"),
        )
    )
    dates = pd.read_parquet(
        DATA_PROCESSED / "dim_date.parquet",
        columns=["date_key", "full_date", "day_of_week", "day_name"],
    )
    stores = pd.read_parquet(
        DATA_PROCESSED / "dim_store.parquet",
        columns=["store_key", "store_nbr"],
    )
    bridge = pd.read_parquet(
        DATA_PROCESSED / "bridge_store_holiday.parquet",
        columns=["date_key", "store_key", "is_holiday", "is_event"],
    )
    if bridge.duplicated(["date_key", "store_key"]).any():
        raise ValueError("Holiday bridge grain is not unique")
    result = (
        daily.merge(dates, on="date_key", validate="many_to_one")
        .merge(stores, on="store_key", validate="many_to_one")
        .merge(
            bridge,
            on=["date_key", "store_key"],
            how="left",
            validate="one_to_one",
        )
        .sort_values(["store_nbr", "full_date"])
        .reset_index(drop=True)
    )
    result[["is_holiday", "is_event"]] = result[
        ["is_holiday", "is_event"]
    ].fillna(0).astype("int8")
    if len(result) != len(daily):
        raise AssertionError("Context mapping changed store-day row count")
    if not np.isclose(result["actual_sales"].sum(), fact["sales"].sum()):
        raise AssertionError("Context mapping changed total sales")
    return result


def build_system_daily(store_daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mapped store-days to one system row per date."""
    result = (
        store_daily.groupby(
            ["date_key", "full_date", "day_of_week", "day_name"],
            as_index=False,
            observed=True,
        )
        .agg(
            actual_sales=("actual_sales", "sum"),
            is_holiday=("is_holiday", "max"),
            is_event=("is_event", "max"),
            is_promotion=("is_promotion", "max"),
            promotion_rate=("promotion_rate", "mean"),
        )
        .sort_values("full_date")
        .reset_index(drop=True)
    )
    result["store_nbr"] = pd.NA
    return result


def add_rolling_scores(frame: pd.DataFrame, level: str) -> pd.DataFrame:
    """Calculate z-scores against strictly preceding rolling observations."""
    result = frame.copy()
    if level.startswith("system"):
        shifted = result["actual_sales"].shift(1)
        result["rolling_expected_sales"] = shifted.rolling(
            ROLLING_WINDOW_DAYS, min_periods=ROLLING_MIN_OBSERVATIONS
        ).mean()
        result["rolling_sales_std"] = shifted.rolling(
            ROLLING_WINDOW_DAYS, min_periods=ROLLING_MIN_OBSERVATIONS
        ).std()
    else:
        grouped = result.groupby("store_nbr", observed=True)["actual_sales"]
        result["rolling_expected_sales"] = grouped.transform(
            lambda values: values.shift(1).rolling(
                ROLLING_WINDOW_DAYS, min_periods=ROLLING_MIN_OBSERVATIONS
            ).mean()
        )
        result["rolling_sales_std"] = grouped.transform(
            lambda values: values.shift(1).rolling(
                ROLLING_WINDOW_DAYS, min_periods=ROLLING_MIN_OBSERVATIONS
            ).std()
        )
    result["z_score"] = safe_divide(
        result["actual_sales"] - result["rolling_expected_sales"],
        result["rolling_sales_std"],
    )
    result["rolling_z_anomaly"] = (
        result["z_score"].abs().ge(ROLLING_Z_THRESHOLD)
    ).astype("int8")
    return result


def add_weekday_iqr_scores(frame: pd.DataFrame, level: str) -> pd.DataFrame:
    """Compare sales with the median and IQR of the same weekday peer group."""
    result = frame.copy()
    grouping = (
        ["day_of_week"]
        if level.startswith("system")
        else ["store_nbr", "day_of_week"]
    )
    peers = (
        result.groupby(grouping, as_index=False, observed=True)
        .agg(
            weekday_median_sales=("actual_sales", "median"),
            weekday_q1_sales=("actual_sales", lambda values: values.quantile(0.25)),
            weekday_q3_sales=("actual_sales", lambda values: values.quantile(0.75)),
            weekday_observation_count=("actual_sales", "size"),
        )
    )
    result = result.merge(peers, on=grouping, validate="many_to_one")
    result["weekday_iqr_sales"] = (
        result["weekday_q3_sales"] - result["weekday_q1_sales"]
    )
    result["weekday_iqr_score"] = safe_divide(
        result["actual_sales"] - result["weekday_median_sales"],
        result["weekday_iqr_sales"],
    )
    lower = result["weekday_q1_sales"] - IQR_MULTIPLIER * result["weekday_iqr_sales"]
    upper = result["weekday_q3_sales"] + IQR_MULTIPLIER * result["weekday_iqr_sales"]
    result["weekday_iqr_anomaly"] = (
        result["actual_sales"].lt(lower) | result["actual_sales"].gt(upper)
    ).astype("int8")
    return result


def classify_anomalies(frame: pd.DataFrame, level: str) -> pd.DataFrame:
    """Retain flags from either method and assign a review-oriented category."""
    result = add_weekday_iqr_scores(add_rolling_scores(frame, level), level)
    result["analysis_level"] = level
    result["anomaly_method"] = np.select(
        [
            result["rolling_z_anomaly"].eq(1)
            & result["weekday_iqr_anomaly"].eq(1),
            result["rolling_z_anomaly"].eq(1),
            result["weekday_iqr_anomaly"].eq(1),
        ],
        ["Rolling z-score + weekday IQR", "Rolling z-score", "Weekday IQR"],
        default="Not flagged",
    )
    anomalies = result.loc[
        result["rolling_z_anomaly"].eq(1)
        | result["weekday_iqr_anomaly"].eq(1)
    ].copy()
    anomalies["material_promotion_context"] = anomalies["promotion_rate"].ge(
        MATERIAL_PROMOTION_RATE
    ).astype("int8")
    business_context = (
        anomalies["is_holiday"].eq(1)
        | anomalies["is_event"].eq(1)
        | anomalies["material_promotion_context"].eq(1)
    )
    potential_issue = (
        ~business_context
        & (
            anomalies["actual_sales"].eq(0)
            | (
                anomalies["rolling_z_anomaly"].eq(1)
                & anomalies["weekday_iqr_anomaly"].eq(1)
                & anomalies["z_score"].abs().ge(POTENTIAL_ISSUE_Z_THRESHOLD)
            )
        )
    )
    anomalies["review_category"] = np.select(
        [business_context, potential_issue],
        ["Business event", "Potential data issue"],
        default="Unexplained anomaly",
    )
    anomalies["expected_sales"] = anomalies["rolling_expected_sales"].fillna(
        anomalies["weekday_median_sales"]
    )
    anomalies["difference"] = (
        anomalies["actual_sales"] - anomalies["expected_sales"]
    )
    anomalies["review_guidance"] = np.select(
        [
            anomalies["review_category"].eq("Business event"),
            anomalies["review_category"].eq("Potential data issue"),
        ],
        [
            "Known business context overlaps the flag; review, do not treat as error",
            "Extreme or zero value without mapped context; investigate source records",
        ],
        default="No mapped business context; investigate before drawing conclusions",
    )
    return anomalies


def write_notes(all_rows: int, anomalies: pd.DataFrame) -> None:
    """Document methods and guard against treating flags as automatic errors."""
    counts = anomalies["review_category"].value_counts()
    count_text = ", ".join(f"{name}: {count}" for name, count in counts.items())
    NOTES_PATH.write_text(
        "# Sales Anomaly Review Notes\n\n"
        f"- {all_rows:,} daily observations were reviewed; no observation was deleted "
        "or modified.\n"
        f"- Rolling z-score uses only the preceding {ROLLING_WINDOW_DAYS} observations "
        f"and requires at least {ROLLING_MIN_OBSERVATIONS}. The threshold is absolute "
        f"z-score >= {ROLLING_Z_THRESHOLD}.\n"
        f"- The second method compares each value with the median and "
        f"{IQR_MULTIPLIER}×IQR fences for its same-weekday peer group. This is a "
        "descriptive full-sample peer comparison.\n"
        "- Holiday and event context is joined at date_key + store_key before system "
        "aggregation. Promotion context is derived from the original family rows.\n"
        f"- Review categories are {count_text}. `Potential data issue` means only that "
        "an extreme/zero observation lacks mapped business context; it is not a "
        "confirmed data error. `Unexplained anomaly` also requires investigation.\n"
        "- Thresholds are review heuristics, not causal or probabilistic claims. "
        "Business events may be unmapped, and mapped context does not necessarily "
        "explain the observed sales value.\n",
        encoding="utf-8",
    )


# %% [markdown]
# ## Run anomaly review

# %%
def main() -> None:
    """Review system- and store-level daily anomalies and save them unchanged."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    store_daily = load_store_daily()
    system_daily = build_system_daily(store_daily)
    store_anomalies = classify_anomalies(store_daily, "store_day")
    system_anomalies = classify_anomalies(system_daily, "system_day")
    anomalies = pd.concat([system_anomalies, store_anomalies], ignore_index=True)
    output_columns = [
        "analysis_level",
        "full_date",
        "store_nbr",
        "actual_sales",
        "expected_sales",
        "difference",
        "z_score",
        "is_holiday",
        "is_event",
        "is_promotion",
        "promotion_rate",
        "material_promotion_context",
        "rolling_expected_sales",
        "rolling_sales_std",
        "rolling_z_anomaly",
        "weekday_median_sales",
        "weekday_q1_sales",
        "weekday_q3_sales",
        "weekday_iqr_sales",
        "weekday_iqr_score",
        "weekday_iqr_anomaly",
        "weekday_observation_count",
        "anomaly_method",
        "review_category",
        "review_guidance",
    ]
    anomalies = anomalies[output_columns].rename(columns={"full_date": "date"})
    if np.isinf(anomalies.select_dtypes("number").to_numpy()).any():
        raise AssertionError("Anomaly output contains an infinite score")
    if anomalies.duplicated(["analysis_level", "date", "store_nbr"]).any():
        raise AssertionError("Anomaly review output grain is not unique")
    anomalies.to_csv(OUTPUT_PATH, index=False)
    write_notes(len(store_daily) + len(system_daily), anomalies)
    print(f"System-day anomalies: {len(system_anomalies):,}")
    print(f"Store-day anomalies: {len(store_anomalies):,}")
    print(anomalies["review_category"].value_counts().to_string())
    print(f"Table: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Interpretation
#
# These flags identify observations for review. They are not automatically data
# errors and no anomaly is removed. Holiday, event, and promotion overlap provide
# business context, while unexplained flags still require source-level review.
