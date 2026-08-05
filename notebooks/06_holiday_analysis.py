# %% [markdown]
# # Holiday and Event Analysis
#
# Holiday mappings are joined at `date_key + store_key`, never by date alone.
# Comparisons are descriptive and do not establish that holidays cause changes
# in sales.

# %%
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED, FIGURES_DIR, TABLES_DIR

OUTPUT_PATH = TABLES_DIR / "holiday_analysis.csv"
SUMMARY_PATH = TABLES_DIR / "holiday_event_summary.csv"
NOTES_PATH = TABLES_DIR / "holiday_analysis_notes.md"
FIGURE_DIR = FIGURES_DIR / "holiday_analysis"
MIN_EVENT_OBSERVATIONS = 30
BASELINE_LAGS = (7, 14, 21, 28)

EVENT_FLAGS = {
    "National holiday": "is_national_holiday",
    "Regional holiday": "is_regional_holiday",
    "Local holiday": "is_local_holiday",
    "Holiday": "is_type_holiday",
    "Transfer": "is_transfer",
    "Additional": "is_additional",
    "Bridge": "is_bridge",
    "Event": "is_event",
    "Work Day": "is_work_day",
    "Payday + holiday": "is_payday_holiday",
}


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide numeric series, leaving zero-denominator results undefined."""
    result = numerator.astype("float64").div(
        denominator.astype("float64").replace(0, np.nan)
    )
    return result.where(np.isfinite(result))


def _contains_token(series: pd.Series, token: str) -> pd.Series:
    """Match a pipe-delimited holiday token without substring ambiguity."""
    return series.fillna("").str.split(" | ", regex=False).apply(
        lambda values: token in values
    )


def load_daily_store_analysis() -> pd.DataFrame:
    """Aggregate sales first, then attach one holiday row per date and store."""
    sales = pd.read_parquet(
        DATA_PROCESSED / "fact_daily_sales.parquet",
        columns=["date_key", "store_key", "sales"],
    )
    daily = (
        sales.assign(sales=sales["sales"].astype("float64"))
        .groupby(["date_key", "store_key"], as_index=False, observed=True)
        .agg(actual_sales=("sales", "sum"))
    )
    dates = pd.read_parquet(
        DATA_PROCESSED / "dim_date.parquet",
        columns=["date_key", "full_date", "day_of_week", "day_name", "is_payday"],
    )
    stores = pd.read_parquet(
        DATA_PROCESSED / "dim_store.parquet",
        columns=["store_key", "store_nbr", "city", "state"],
    )
    bridge = pd.read_parquet(DATA_PROCESSED / "bridge_store_holiday.parquet")
    if bridge.duplicated(["date_key", "store_key"]).any():
        raise ValueError("Holiday bridge has duplicate date_key + store_key grain")

    calendar_store = dates.merge(stores, how="cross", validate="many_to_many")
    enriched = (
        calendar_store.merge(
            daily,
            on=["date_key", "store_key"],
            how="left",
            validate="one_to_one",
            indicator="sales_observation_status",
        )
        .merge(
            bridge,
            on=["date_key", "store_key"],
            how="left",
            validate="one_to_one",
            indicator="holiday_mapping_status",
        )
    )
    if len(enriched) != len(dates) * len(stores):
        raise AssertionError("Calendar-store analysis grid is incomplete")
    if not np.isclose(enriched["actual_sales"].sum(), sales["sales"].sum()):
        raise AssertionError("Holiday join multiplied or lost sales")

    mapped = enriched["holiday_mapping_status"].eq("both")
    enriched["has_holiday_mapping"] = mapped.astype("int8")
    enriched["holiday_count"] = enriched["holiday_count"].fillna(0).astype("int16")
    for column in ["holiday_descriptions", "holiday_types", "holiday_locales"]:
        enriched[column] = enriched[column].fillna("")
    for column in ["is_holiday", "is_work_day", "is_event"]:
        enriched[column] = enriched[column].fillna(0).astype("int8")

    enriched["is_national_holiday"] = (
        _contains_token(enriched["holiday_locales"], "National")
        & enriched["is_holiday"].eq(1)
    ).astype("int8")
    enriched["is_regional_holiday"] = (
        _contains_token(enriched["holiday_locales"], "Regional")
        & enriched["is_holiday"].eq(1)
    ).astype("int8")
    enriched["is_local_holiday"] = (
        _contains_token(enriched["holiday_locales"], "Local")
        & enriched["is_holiday"].eq(1)
    ).astype("int8")
    for token, column in [
        ("Holiday", "is_type_holiday"),
        ("Transfer", "is_transfer"),
        ("Additional", "is_additional"),
        ("Bridge", "is_bridge"),
    ]:
        enriched[column] = _contains_token(enriched["holiday_types"], token).astype(
            "int8"
        )
    enriched["is_payday_holiday"] = (
        enriched["is_payday"].eq(1) & enriched["is_holiday"].eq(1)
    ).astype("int8")
    enriched["is_special_day"] = (
        enriched["has_holiday_mapping"].eq(1)
    ).astype("int8")
    return enriched.sort_values(["store_key", "full_date"]).reset_index(drop=True)


def add_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    """Add four strictly historical same-weekday sales observations and baseline."""
    result = frame.copy()
    date_gaps = result.groupby("store_key", observed=True)["full_date"].diff().dropna()
    if not date_gaps.eq(pd.Timedelta(days=1)).all():
        raise AssertionError("Calendar-store rows are not daily and continuous")
    grouped_sales = result.groupby("store_key", observed=True)["actual_sales"]
    grouped_special = result.groupby("store_key", observed=True)["is_special_day"]
    lag_sales_columns: list[str] = []
    lag_special_columns: list[str] = []
    for lag in BASELINE_LAGS:
        sales_column = f"sales_lag_{lag}d"
        special_column = f"special_day_lag_{lag}d"
        result[sales_column] = grouped_sales.shift(lag)
        result[special_column] = grouped_special.shift(lag)
        lag_sales_columns.append(sales_column)
        lag_special_columns.append(special_column)
    result["baseline_observation_count"] = result[lag_sales_columns].notna().sum(axis=1)
    result["baseline_sales"] = result[lag_sales_columns].mean(axis=1)
    result.loc[result["baseline_observation_count"].lt(4), "baseline_sales"] = np.nan
    result["baseline_special_day_count"] = result[lag_special_columns].sum(
        axis=1, min_count=1
    )
    result["baseline_status"] = np.where(
        result["baseline_observation_count"].eq(4),
        "Complete four-week baseline",
        "Incomplete baseline retained",
    )
    result["difference"] = result["actual_sales"] - result["baseline_sales"]
    result["difference_pct"] = (
        safe_divide(result["difference"], result["baseline_sales"]) * 100
    )

    # A second descriptive benchmark compares each store to its ordinary days
    # of the same weekday. Special days are not deleted from the output table.
    regular = result.loc[result["is_special_day"].eq(0)]
    same_weekday = (
        regular.groupby(["store_key", "day_of_week"], as_index=False, observed=True)
        .agg(same_weekday_regular_average_sales=("actual_sales", "mean"))
    )
    result = result.merge(
        same_weekday,
        on=["store_key", "day_of_week"],
        how="left",
        validate="many_to_one",
    )
    result["same_weekday_regular_difference"] = (
        result["actual_sales"] - result["same_weekday_regular_average_sales"]
    )
    result["same_weekday_regular_difference_pct"] = (
        safe_divide(
            result["same_weekday_regular_difference"],
            result["same_weekday_regular_average_sales"],
        )
        * 100
    )
    return result


def add_holiday_windows_and_warnings(frame: pd.DataFrame) -> pd.DataFrame:
    """Mark days before, during, and after a holiday without dropping overlaps."""
    result = frame.copy()
    by_store = result.groupby("store_key", observed=True)["is_holiday"]
    result["is_day_before_holiday"] = by_store.shift(-1).fillna(0).astype("int8")
    result["is_day_after_holiday"] = by_store.shift(1).fillna(0).astype("int8")
    result["holiday_window"] = np.select(
        [
            result["is_holiday"].eq(1),
            result["is_day_before_holiday"].eq(1),
            result["is_day_after_holiday"].eq(1),
        ],
        ["During holiday", "Before holiday", "After holiday"],
        default="Other day",
    )
    result["event_label"] = result["holiday_descriptions"].replace(
        "", "Regular day"
    )
    event_counts = result.groupby("event_label", observed=True)["actual_sales"].count()
    result["event_observation_count"] = result["event_label"].map(event_counts)
    result["small_event_warning"] = np.where(
        result["has_holiday_mapping"].eq(1)
        & result["event_observation_count"].lt(MIN_EVENT_OBSERVATIONS),
        "Event below minimum observation count",
        "",
    )
    return result


def build_category_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize overlapping event categories; category totals are not additive."""
    categories: list[pd.DataFrame] = []
    regular = frame.loc[frame["is_special_day"].eq(0)].copy()
    regular["event_category"] = "Regular day"
    categories.append(regular)
    for label, flag in EVENT_FLAGS.items():
        selected = frame.loc[frame[flag].eq(1)].copy()
        selected["event_category"] = label
        categories.append(selected)
    long = pd.concat(categories, ignore_index=True, sort=False)
    summary = (
        long.groupby("event_category", as_index=False, observed=True)
        .agg(
            observation_count=("actual_sales", "count"),
            average_actual_sales=("actual_sales", "mean"),
            median_actual_sales=("actual_sales", "median"),
            actual_sales_std=("actual_sales", "std"),
            average_baseline_sales=("baseline_sales", "mean"),
            average_difference=("difference", "mean"),
            median_difference=("difference", "median"),
            average_difference_pct=("difference_pct", "mean"),
            median_difference_pct=("difference_pct", "median"),
            q25_difference_pct=("difference_pct", lambda values: values.quantile(0.25)),
            q75_difference_pct=("difference_pct", lambda values: values.quantile(0.75)),
            payday_observation_count=("is_payday", "sum"),
        )
    )
    summary["small_sample_warning"] = np.where(
        summary["observation_count"].lt(MIN_EVENT_OBSERVATIONS),
        "Category below minimum observation count",
        "",
    )
    return summary


def create_figures(frame: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Show central tendency and full distributions without removing outliers."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plotted = summary.sort_values("median_difference_pct")
    labels = [
        f"{row.event_category} (n={row.observation_count:,})"
        for row in plotted.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(labels, plotted["median_difference_pct"], color="#2a9d8f")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(
        title="Median sales difference from four-week same-weekday baseline",
        xlabel="Median difference (%)",
        ylabel="",
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "event_category_baseline_difference.png", dpi=160)
    plt.close(fig)

    category_frames = []
    category_labels = []
    for label, flag in EVENT_FLAGS.items():
        values = frame.loc[frame[flag].eq(1), "difference_pct"].dropna()
        if not values.empty:
            category_frames.append(values.to_numpy())
            category_labels.append(label)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.boxplot(category_frames, tick_labels=category_labels, showfliers=True)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        title="Distribution of baseline differences by event category",
        ylabel="Difference from baseline (%)",
        xlabel="All observations retained; overlapping categories are shown separately",
    )
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "event_category_difference_distribution.png", dpi=160)
    plt.close(fig)

    weekday_groups = []
    weekday_labels = []
    for day_name in frame.sort_values("day_of_week")["day_name"].unique():
        for status, flag_value in [("Regular", 0), ("Holiday", 1)]:
            values = frame.loc[
                frame["day_name"].eq(day_name) & frame["is_holiday"].eq(flag_value),
                "actual_sales",
            ].dropna()
            if not values.empty:
                weekday_groups.append(np.log1p(values.to_numpy()))
                weekday_labels.append(f"{day_name}\n{status}")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.boxplot(weekday_groups, tick_labels=weekday_labels, showfliers=True)
    ax.set(
        title="Holiday and regular-day sales distributions within weekday",
        ylabel="log(1 + actual sales)",
        xlabel="Log transform changes scale only; no observations are removed",
    )
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "holiday_vs_regular_by_weekday_distribution.png", dpi=160)
    plt.close(fig)

    windows = ["Before holiday", "During holiday", "After holiday", "Other day"]
    window_data = [
        np.log1p(
            frame.loc[frame["holiday_window"].eq(window), "actual_sales"].dropna()
        )
        for window in windows
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.boxplot(window_data, tick_labels=windows, showfliers=True)
    ax.set(
        title="Sales distribution before, during, and after holidays",
        ylabel="log(1 + actual sales)",
        xlabel="All days retained",
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "holiday_window_sales_distribution.png", dpi=160)
    plt.close(fig)

    payday = frame.assign(
        payday_holiday_group=np.select(
            [
                frame["is_payday_holiday"].eq(1),
                frame["is_holiday"].eq(1),
                frame["is_payday"].eq(1),
            ],
            ["Payday + holiday", "Holiday only", "Payday only"],
            default="Neither",
        )
    )
    order = ["Neither", "Payday only", "Holiday only", "Payday + holiday"]
    payday_values = [
        payday.loc[payday["payday_holiday_group"].eq(group), "difference_pct"].dropna()
        for group in order
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.boxplot(payday_values, tick_labels=order, showfliers=True)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        title="Payday and holiday baseline-difference distributions",
        ylabel="Difference from baseline (%)",
        xlabel="All observations retained",
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "payday_holiday_distribution.png", dpi=160)
    plt.close(fig)


def write_notes(frame: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Document baseline construction, preserved special days, and limitations."""
    multi_event_days = int(frame["holiday_count"].gt(1).sum())
    incomplete = int(frame["baseline_observation_count"].lt(4).sum())
    special_without_sales = int(
        (frame["has_holiday_mapping"].eq(1) & frame["actual_sales"].isna()).sum()
    )
    small_events = int(
        frame.loc[frame["small_event_warning"].ne(""), "event_label"].nunique()
    )
    NOTES_PATH.write_text(
        "# Holiday Analysis Notes and Limitations\n\n"
        "- Grain is one row per date and store. The holiday bridge is joined on "
        "both date_key and store_key with a one-to-one validation.\n"
        f"- {multi_event_days:,} store-days contain multiple mapped events. Their "
        "descriptions/types stay aggregated in one row, so sales is not multiplied.\n"
        f"- {special_without_sales:,} mapped special store-days fall outside the "
        "observed sales fact range. They remain in the table with missing actual "
        "sales and are not used as sales observations in summaries.\n"
        "- Baseline sales is the mean of sales exactly 7, 14, 21, and 28 days "
        "earlier for the same store. It uses no future observations and therefore "
        "compares the same weekday approximately.\n"
        "- Prior special days are not removed from the baseline. Their count is "
        "recorded in `baseline_special_day_count`.\n"
        f"- {incomplete:,} calendar-store rows lack all four historical sales "
        "observations because of early dates or gaps outside the observed fact "
        "range. They remain in the output with an incomplete-baseline status and "
        "undefined baseline comparison.\n"
        f"- Events below {MIN_EVENT_OBSERVATIONS} observations are flagged; "
        f"{small_events} distinct event labels receive this warning.\n"
        "- Distribution charts retain all observations and outliers. Log scale is "
        "used only for readability in sales-distribution charts.\n"
        "- Event categories can overlap, so category-summary counts are not additive.\n"
        "- Results are descriptive associations, not causal estimates. Holidays, "
        "promotions, store operations, trends, and other contemporaneous factors may "
        "all contribute to differences.\n",
        encoding="utf-8",
    )


# %% [markdown]
# ## Run analysis

# %%
def main() -> None:
    """Run the holiday analysis and save the requested table and figures."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_daily_store_analysis()
    frame = add_baselines(frame)
    frame = add_holiday_windows_and_warnings(frame)
    summary = build_category_summary(frame)
    if frame.duplicated(["full_date", "store_nbr"]).any():
        raise AssertionError("Holiday analysis grain is not unique")
    if np.isinf(frame.select_dtypes("number").to_numpy()).any():
        raise AssertionError("Holiday analysis contains an infinite result")
    output_columns = [
        "full_date",
        "date_key",
        "store_nbr",
        "store_key",
        "city",
        "state",
        "day_of_week",
        "day_name",
        "is_payday",
        "actual_sales",
        "baseline_sales",
        "difference",
        "difference_pct",
        "baseline_observation_count",
        "baseline_special_day_count",
        "baseline_status",
        "same_weekday_regular_average_sales",
        "same_weekday_regular_difference",
        "same_weekday_regular_difference_pct",
        "has_holiday_mapping",
        "holiday_count",
        "holiday_descriptions",
        "holiday_types",
        "holiday_locales",
        "is_holiday",
        "is_national_holiday",
        "is_regional_holiday",
        "is_local_holiday",
        "is_type_holiday",
        "is_transfer",
        "is_additional",
        "is_bridge",
        "is_event",
        "is_work_day",
        "is_payday_holiday",
        "is_day_before_holiday",
        "is_day_after_holiday",
        "holiday_window",
        "event_label",
        "event_observation_count",
        "small_event_warning",
        "sales_observation_status",
        "holiday_mapping_status",
    ]
    frame[output_columns].to_csv(OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    create_figures(frame, summary)
    write_notes(frame, summary)
    print(f"Rows: {len(frame):,}")
    print(f"Mapped special store-days: {frame['has_holiday_mapping'].sum():,}")
    print(f"Multiple-event store-days: {frame['holiday_count'].gt(1).sum():,}")
    print(f"Table: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Figures: {FIGURE_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Interpretation limitations
#
# The four-week same-weekday baseline is an approximate descriptive benchmark,
# not a counterfactual. No causal conclusion is made. All special days and all
# outliers remain recorded; overlapping category summaries must not be added
# together because one store-day can belong to several event categories.
