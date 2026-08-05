# %% [markdown]
# # Transactions Analysis
#
# Sales is aggregated to one row per date and store before transactions is
# joined. Correlation is interpreted as association, not causation.

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

OUTPUT_PATH = TABLES_DIR / "transactions_analysis.csv"
STORE_SUMMARY_PATH = TABLES_DIR / "transactions_store_summary.csv"
MONTHLY_DRIVER_PATH = TABLES_DIR / "transactions_monthly_driver.csv"
NOTES_PATH = TABLES_DIR / "transactions_analysis_notes.md"
FIGURE_DIR = FIGURES_DIR / "transactions_analysis"
ROLLING_WINDOW = 28
ANOMALY_THRESHOLD = 3.5


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide numeric series and leave zero-denominator values undefined."""
    result = numerator.astype("float64").div(
        denominator.astype("float64").replace(0, np.nan)
    )
    return result.where(np.isfinite(result))


def load_store_day_data() -> pd.DataFrame:
    """Aggregate family sales before a validated one-to-one transaction join."""
    sales = pd.read_parquet(
        DATA_PROCESSED / "fact_daily_sales.parquet",
        columns=["date_key", "store_key", "sales"],
    )
    daily_sales = (
        sales.assign(sales=sales["sales"].astype("float64"))
        .groupby(["date_key", "store_key"], as_index=False, observed=True)
        .agg(total_sales=("sales", "sum"))
    )
    if daily_sales.duplicated(["date_key", "store_key"]).any():
        raise AssertionError("Daily store sales aggregation is not unique")

    transactions = pd.read_parquet(
        DATA_PROCESSED / "fact_store_transactions.parquet"
    )
    if transactions.duplicated(["date_key", "store_key"]).any():
        raise AssertionError("Transaction fact grain is not unique")
    merged = transactions.merge(
        daily_sales,
        on=["date_key", "store_key"],
        how="left",
        validate="one_to_one",
        indicator="sales_mapping_status",
    )
    if merged["total_sales"].isna().any():
        raise ValueError("At least one transaction row does not map to daily store sales")
    if len(merged) != len(transactions):
        raise AssertionError("Transaction row count changed during sales merge")
    if merged["transactions"].sum() != transactions["transactions"].sum():
        raise AssertionError("Transaction total changed during sales merge")

    dates = pd.read_parquet(
        DATA_PROCESSED / "dim_date.parquet",
        columns=[
            "date_key",
            "full_date",
            "day_of_week",
            "day_name",
            "month",
            "month_name",
            "year",
        ],
    )
    stores = pd.read_parquet(
        DATA_PROCESSED / "dim_store.parquet",
        columns=[
            "store_key",
            "store_nbr",
            "city",
            "state",
            "store_type",
            "cluster",
        ],
    )
    result = (
        merged.merge(dates, on="date_key", validate="many_to_one")
        .merge(stores, on="store_key", validate="many_to_one")
        .sort_values(["store_nbr", "full_date"])
        .reset_index(drop=True)
    )
    result["sales_volume_per_transaction"] = safe_divide(
        result["total_sales"], result["transactions"]
    )
    result["zero_transaction_flag"] = result["transactions"].eq(0).astype("int8")
    return result


def add_anomalies_and_trends(frame: pd.DataFrame) -> pd.DataFrame:
    """Add robust store anomalies and a 28-observation rolling average."""
    result = frame.copy()
    store_median = result.groupby("store_nbr", observed=True)["transactions"].transform(
        "median"
    )
    absolute_deviation = (result["transactions"] - store_median).abs()
    store_mad = absolute_deviation.groupby(result["store_nbr"], observed=True).transform(
        "median"
    )
    result["transaction_store_median"] = store_median
    result["transaction_store_mad"] = store_mad
    result["transaction_robust_z"] = safe_divide(
        0.6745 * (result["transactions"] - store_median), store_mad
    )
    result["unusual_transaction_day"] = (
        result["transaction_robust_z"].abs().ge(ANOMALY_THRESHOLD)
    ).astype("int8")
    result["anomaly_rule"] = (
        f"absolute modified z-score >= {ANOMALY_THRESHOLD}; no rows removed"
    )

    rolling = (
        result.groupby("store_nbr", observed=True)["transactions"]
        .rolling(ROLLING_WINDOW, min_periods=7)
        .mean()
        .reset_index(level=0, drop=True)
    )
    result["transactions_28_observation_rolling_average"] = rolling.sort_index()
    return result


def build_store_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank stores by transactions and sales volume per transaction."""
    summary = (
        frame.groupby(
            ["store_nbr", "city", "state", "store_type", "cluster"],
            as_index=False,
            observed=True,
        )
        .agg(
            total_sales=("total_sales", "sum"),
            total_transactions=("transactions", "sum"),
            average_daily_transactions=("transactions", "mean"),
            median_daily_transactions=("transactions", "median"),
            transaction_observation_count=("transactions", "size"),
            unusual_transaction_days=("unusual_transaction_day", "sum"),
        )
    )
    summary["sales_volume_per_transaction"] = safe_divide(
        summary["total_sales"], summary["total_transactions"]
    )
    correlations = (
        frame.groupby("store_nbr", observed=True)
        .apply(
            lambda group: group["total_sales"].corr(group["transactions"]),
            include_groups=False,
        )
        .rename("sales_transactions_correlation")
        .reset_index()
    )
    summary = summary.merge(correlations, on="store_nbr", validate="one_to_one")
    summary["rank_total_transactions"] = summary["total_transactions"].rank(
        method="min", ascending=False
    ).astype("int64")
    summary["rank_sales_volume_per_transaction"] = summary[
        "sales_volume_per_transaction"
    ].rank(method="min", ascending=False, na_option="keep").astype("Int64")
    return summary.sort_values("store_nbr").reset_index(drop=True)


def build_monthly_driver(frame: pd.DataFrame) -> pd.DataFrame:
    """Decompose monthly sales change into transaction and volume effects."""
    monthly = (
        frame.assign(year_month=frame["full_date"].dt.to_period("M").astype(str))
        .groupby("year_month", as_index=False, observed=True)
        .agg(
            total_sales=("total_sales", "sum"),
            total_transactions=("transactions", "sum"),
            store_day_count=("transactions", "size"),
        )
    )
    monthly["sales_volume_per_transaction"] = safe_divide(
        monthly["total_sales"], monthly["total_transactions"]
    )
    monthly["previous_total_sales"] = monthly["total_sales"].shift()
    monthly["previous_total_transactions"] = monthly["total_transactions"].shift()
    monthly["previous_sales_volume_per_transaction"] = monthly[
        "sales_volume_per_transaction"
    ].shift()
    monthly["sales_change"] = monthly["total_sales"].diff()
    monthly["transaction_effect"] = (
        monthly["total_transactions"] - monthly["previous_total_transactions"]
    ) * monthly["previous_sales_volume_per_transaction"]
    monthly["volume_per_transaction_effect"] = (
        monthly["sales_volume_per_transaction"]
        - monthly["previous_sales_volume_per_transaction"]
    ) * monthly["previous_total_transactions"]
    monthly["interaction_effect"] = (
        monthly["total_transactions"] - monthly["previous_total_transactions"]
    ) * (
        monthly["sales_volume_per_transaction"]
        - monthly["previous_sales_volume_per_transaction"]
    )
    monthly["sales_change_direction"] = np.select(
        [monthly["sales_change"].gt(0), monthly["sales_change"].lt(0)],
        ["Increase", "Decrease"],
        default="No change / first month",
    )
    volume_total_effect = (
        monthly["volume_per_transaction_effect"] + monthly["interaction_effect"]
    )
    monthly["dominant_change_driver"] = np.select(
        [
            monthly["sales_change"].isna(),
            monthly["transaction_effect"].abs().ge(volume_total_effect.abs()),
        ],
        ["Not available for first month", "Transactions"],
        default="Sales volume per transaction",
    )
    if not np.allclose(
        monthly["sales_change"].iloc[1:],
        (
            monthly["transaction_effect"]
            + monthly["volume_per_transaction_effect"]
            + monthly["interaction_effect"]
        ).iloc[1:],
    ):
        raise AssertionError("Monthly sales-change decomposition does not reconcile")
    return monthly


def create_figures(
    frame: pd.DataFrame,
    store_summary: pd.DataFrame,
    monthly_driver: pd.DataFrame,
) -> None:
    """Create store-day scatter, rankings, distributions, and trends."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        frame["transactions"],
        frame["total_sales"],
        s=7,
        alpha=0.12,
        color="#2a9d8f",
        edgecolors="none",
    )
    correlation = frame["total_sales"].corr(frame["transactions"])
    ax.set(
        title=(
            f"Store-day sales and transactions association "
            f"(Pearson r={correlation:.3f}, n={len(frame):,})"
        ),
        xlabel="Transactions",
        ylabel="Total sales",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "sales_vs_transactions_store_day_scatter.png", dpi=160)
    plt.close(fig)

    for metric, title, xlabel, filename in [
        (
            "total_transactions",
            "Stores with the most transactions",
            "Total transactions",
            "top_stores_total_transactions.png",
        ),
        (
            "sales_volume_per_transaction",
            "Stores with highest sales volume per transaction",
            "Sales volume per transaction",
            "top_stores_sales_volume_per_transaction.png",
        ),
    ]:
        top = store_summary.nlargest(15, metric).sort_values(metric)
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh("Store " + top["store_nbr"].astype(str), top[metric], color="#457b9d")
        ax.set(title=title, xlabel=xlabel, ylabel="")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / filename, dpi=160)
        plt.close(fig)

    anomalies = frame.loc[frame["unusual_transaction_day"].eq(1)]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(
        frame["full_date"],
        frame["transactions"],
        s=3,
        alpha=0.08,
        label="All store-days",
        color="#457b9d",
    )
    ax.scatter(
        anomalies["full_date"],
        anomalies["transactions"],
        s=14,
        alpha=0.65,
        label=f"|modified z| ≥ {ANOMALY_THRESHOLD}",
        color="#e76f51",
    )
    ax.set(
        title="Unusual transaction store-days (retained, not removed)",
        xlabel="Date",
        ylabel="Transactions",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "unusual_transaction_days.png", dpi=160)
    plt.close(fig)

    weekday_order = (
        frame[["day_of_week", "day_name"]].drop_duplicates().sort_values("day_of_week")
    )
    weekday_values = [
        frame.loc[frame["day_of_week"].eq(row.day_of_week), "transactions"].to_numpy()
        for row in weekday_order.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        weekday_values,
        tick_labels=weekday_order["day_name"].tolist(),
        showfliers=True,
    )
    ax.set(
        title="Transactions distribution by weekday",
        xlabel="Weekday",
        ylabel="Transactions per store-day",
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "transactions_by_weekday_distribution.png", dpi=160)
    plt.close(fig)

    month_order = frame[["month", "month_name"]].drop_duplicates().sort_values("month")
    month_average = frame.groupby("month", observed=True)["transactions"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        month_order["month_name"],
        month_order["month"].map(month_average),
        marker="o",
        color="#2a9d8f",
    )
    ax.set(
        title="Average store-day transactions by month",
        xlabel="Month",
        ylabel="Average transactions",
    )
    ax.tick_params(axis="x", rotation=35)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "transactions_by_month.png", dpi=160)
    plt.close(fig)

    daily = frame.groupby("full_date", as_index=False, observed=True).agg(
        total_transactions=("transactions", "sum")
    )
    daily["transactions_28d_rolling_average"] = daily[
        "total_transactions"
    ].rolling(28, min_periods=7).mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        daily["full_date"],
        daily["total_transactions"],
        linewidth=0.5,
        alpha=0.25,
        label="Daily total",
        color="#457b9d",
    )
    ax.plot(
        daily["full_date"],
        daily["transactions_28d_rolling_average"],
        linewidth=1.8,
        label="28-day rolling average",
        color="#e76f51",
    )
    ax.set(title="Transactions trend", xlabel="Date", ylabel="Transactions")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "transactions_rolling_trend.png", dpi=160)
    plt.close(fig)

    plotted = monthly_driver.dropna(subset=["sales_change"]).copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        plotted["year_month"],
        plotted["transaction_effect"],
        marker="o",
        markersize=3,
        label="Transaction effect",
        color="#457b9d",
    )
    ax.plot(
        plotted["year_month"],
        plotted["volume_per_transaction_effect"] + plotted["interaction_effect"],
        marker="o",
        markersize=3,
        label="Sales-volume-per-transaction effect incl. interaction",
        color="#e9c46a",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        title="Monthly sales-change decomposition",
        xlabel="Month",
        ylabel="Contribution to sales change",
    )
    ax.set_xticks(np.arange(0, len(plotted), 4), plotted["year_month"].iloc[::4], rotation=45)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "monthly_sales_change_drivers.png", dpi=160)
    plt.close(fig)


def write_notes(frame: pd.DataFrame, monthly_driver: pd.DataFrame) -> None:
    """Document grain, zero handling, anomalies, and interpretation limits."""
    correlation = frame["total_sales"].corr(frame["transactions"])
    zero_days = int(frame["zero_transaction_flag"].sum())
    unusual_days = int(frame["unusual_transaction_day"].sum())
    increases = monthly_driver.loc[monthly_driver["sales_change_direction"].eq("Increase")]
    driver_counts = increases["dominant_change_driver"].value_counts()
    driver_text = ", ".join(f"{name}: {count}" for name, count in driver_counts.items())
    NOTES_PATH.write_text(
        "# Transactions Analysis Notes\n\n"
        "- Grain is one point per date and store. Sales is aggregated across family "
        "before a one-to-one merge with transactions, so transactions is never "
        "repeated by family.\n"
        f"- Store-day Pearson correlation between sales and transactions is "
        f"{correlation:.6f}. Correlation is association, not causation.\n"
        f"- {zero_days:,} zero-transaction rows are retained. Their sales volume per "
        "transaction is undefined rather than infinite.\n"
        f"- {unusual_days:,} unusual store-days are flagged using an absolute "
        f"modified z-score threshold of {ANOMALY_THRESHOLD} within store. They are "
        "not removed.\n"
        f"- Across months with sales increases, dominant decomposition labels are "
        f"{driver_text}. The identity decomposes arithmetic change; it does not prove "
        "a causal mechanism.\n"
        "- Store rolling averages use 28 transaction observations with at least 7 "
        "observations. The overall trend figure uses a 28-calendar-day rolling mean.\n",
        encoding="utf-8",
    )


# %% [markdown]
# ## Run analysis

# %%
def main() -> None:
    """Run transactions analysis and save all requested artifacts."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    frame = add_anomalies_and_trends(load_store_day_data())
    store_summary = build_store_summary(frame)
    monthly_driver = build_monthly_driver(frame)
    if frame.duplicated(["full_date", "store_nbr"]).any():
        raise AssertionError("Transactions analysis grain is not unique")
    if np.isinf(frame.select_dtypes("number").to_numpy()).any():
        raise AssertionError("Transactions analysis contains an infinite result")
    frame.to_csv(OUTPUT_PATH, index=False)
    store_summary.to_csv(STORE_SUMMARY_PATH, index=False)
    monthly_driver.to_csv(MONTHLY_DRIVER_PATH, index=False)
    create_figures(frame, store_summary, monthly_driver)
    write_notes(frame, monthly_driver)
    print(f"Store-day rows: {len(frame):,}")
    print(f"Correlation: {frame['total_sales'].corr(frame['transactions']):.6f}")
    print(f"Unusual store-days: {frame['unusual_transaction_day'].sum():,}")
    print(f"Table: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Figures: {FIGURE_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Interpretation limitations
#
# Correlation does not establish causation. Changes in assortment, promotions,
# holidays, store operations, and customer mix can affect both transactions and
# sales volume per transaction. Anomaly flags are descriptive and no flagged
# observation is removed.
