# %% [markdown]
# # Business EDA: Store Performance
#
# This analysis compares stores on scale, growth, volatility, and transaction
# efficiency. Growth compares the average daily sales in each store's first and
# last 90 observed days.

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

STORE_PERFORMANCE_PATH = TABLES_DIR / "store_performance.csv"
FINDINGS_PATH = TABLES_DIR / "store_performance_findings.md"
FIGURE_DIR = FIGURES_DIR / "business_eda" / "store_performance"
GROWTH_WINDOW_DAYS = 90
TOP_N = 10
STORE_ATTRIBUTES = ["store_nbr", "city", "state", "store_type", "cluster"]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide finite numeric series, returning NaN for zero denominators."""
    denominator = denominator.replace(0, np.nan)
    result = numerator.astype("float64").div(denominator.astype("float64"))
    return result.where(np.isfinite(result))


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load only the processed columns required by the store analysis."""
    sales = pd.read_parquet(
        DATA_PROCESSED / "fact_daily_sales.parquet",
        columns=["date_key", "store_key", "sales"],
    )
    transactions = pd.read_parquet(
        DATA_PROCESSED / "fact_store_transactions.parquet",
        columns=["date_key", "store_key", "transactions"],
    )
    stores = pd.read_parquet(
        DATA_PROCESSED / "dim_store.parquet",
        columns=["store_key", *STORE_ATTRIBUTES],
    )
    dates = pd.read_parquet(
        DATA_PROCESSED / "dim_date.parquet",
        columns=["date_key", "full_date"],
    )
    return sales, transactions, stores, dates


def _calculate_growth(daily_sales: pd.DataFrame) -> pd.DataFrame:
    """Compare first and last fixed observation windows for every store."""
    ordered = daily_sales.sort_values(["store_key", "full_date"])
    first_window = (
        ordered.groupby("store_key", sort=False, observed=True)
        .head(GROWTH_WINDOW_DAYS)
        .groupby("store_key", observed=True)["daily_sales"]
        .mean()
        .rename("first_window_average_daily_sales")
    )
    last_window = (
        ordered.groupby("store_key", sort=False, observed=True)
        .tail(GROWTH_WINDOW_DAYS)
        .groupby("store_key", observed=True)["daily_sales"]
        .mean()
        .rename("last_window_average_daily_sales")
    )
    growth = pd.concat([first_window, last_window], axis=1).reset_index()
    growth["growth_rate"] = safe_divide(
        growth["last_window_average_daily_sales"]
        - growth["first_window_average_daily_sales"],
        growth["first_window_average_daily_sales"],
    )
    return growth


def build_store_performance(
    sales: pd.DataFrame,
    transactions: pd.DataFrame,
    stores: pd.DataFrame,
    dates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build one row per store without multiplying transactions by family."""
    daily_sales = (
        sales.assign(sales=sales["sales"].astype("float64"))
        .groupby(["date_key", "store_key"], as_index=False, observed=True)
        .agg(daily_sales=("sales", "sum"))
        .merge(dates, on="date_key", how="left", validate="many_to_one")
    )
    if daily_sales["full_date"].isna().any():
        raise ValueError("Some sales dates do not map to dim_date")

    sales_metrics = (
        daily_sales.groupby("store_key", as_index=False, observed=True)
        .agg(
            total_sales=("daily_sales", "sum"),
            average_daily_sales=("daily_sales", "mean"),
            sales_std=("daily_sales", "std"),
            active_days=("date_key", "nunique"),
        )
    )
    sales_metrics["coefficient_of_variation"] = safe_divide(
        sales_metrics["sales_std"], sales_metrics["average_daily_sales"]
    )

    transaction_metrics = (
        transactions.groupby("store_key", as_index=False, observed=True)
        .agg(total_transactions=("transactions", "sum"))
    )
    growth = _calculate_growth(daily_sales)
    performance = (
        stores.merge(sales_metrics, on="store_key", validate="one_to_one")
        .merge(transaction_metrics, on="store_key", how="left", validate="one_to_one")
        .merge(growth, on="store_key", validate="one_to_one")
    )
    if performance["total_transactions"].isna().any():
        missing = performance.loc[performance["total_transactions"].isna(), "store_nbr"]
        raise ValueError(f"Stores without transaction mapping: {missing.tolist()}")

    performance["sales_volume_per_transaction"] = safe_divide(
        performance["total_sales"], performance["total_transactions"]
    )
    thresholds = {
        "sales_threshold_median": float(performance["average_daily_sales"].median()),
        "volatility_threshold_median": float(
            performance["coefficient_of_variation"].median()
        ),
    }
    performance["sales_threshold_median"] = thresholds["sales_threshold_median"]
    performance["volatility_threshold_median"] = thresholds[
        "volatility_threshold_median"
    ]
    high_sales = performance["average_daily_sales"].ge(
        thresholds["sales_threshold_median"]
    )
    volatile = performance["coefficient_of_variation"].gt(
        thresholds["volatility_threshold_median"]
    )
    performance["performance_segment"] = np.select(
        [high_sales & ~volatile, high_sales & volatile, ~high_sales & ~volatile],
        [
            "High sales – stable",
            "High sales – volatile",
            "Low sales – stable",
        ],
        default="Low sales – volatile",
    )

    ranking_metrics = {
        "average_daily_sales": "rank_average_daily_sales",
        "growth_rate": "rank_growth_rate",
        "coefficient_of_variation": "rank_volatility",
        "sales_volume_per_transaction": "rank_sales_volume_per_transaction",
    }
    for metric, rank_column in ranking_metrics.items():
        performance[rank_column] = performance[metric].rank(
            method="min", ascending=False, na_option="keep"
        ).astype("Int64")

    output_columns = [
        *STORE_ATTRIBUTES,
        "total_sales",
        "average_daily_sales",
        "sales_std",
        "coefficient_of_variation",
        "active_days",
        "total_transactions",
        "sales_volume_per_transaction",
        "growth_rate",
        "first_window_average_daily_sales",
        "last_window_average_daily_sales",
        "sales_threshold_median",
        "volatility_threshold_median",
        "performance_segment",
        *ranking_metrics.values(),
    ]
    performance = performance[output_columns].sort_values("store_nbr").reset_index(drop=True)

    if performance["store_nbr"].duplicated().any():
        raise AssertionError("Store performance grain is not unique")
    if not np.isclose(performance["total_sales"].sum(), sales["sales"].sum()):
        raise AssertionError("Total sales changed during store aggregation")
    if performance["total_transactions"].sum() != transactions["transactions"].sum():
        raise AssertionError("Transactions changed during store aggregation")
    return performance, thresholds


def top_and_bottom(performance: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return distinct bottom and top stores for a ranking metric."""
    ranked = performance.dropna(subset=[metric]).sort_values(metric)
    count = min(TOP_N, len(ranked) // 2)
    bottom = ranked.head(count).assign(ranking_group="Bottom")
    top = ranked.tail(count).assign(ranking_group="Top")
    return pd.concat([bottom, top], ignore_index=True)


def _plot_ranking(
    performance: pd.DataFrame,
    metric: str,
    title: str,
    xlabel: str,
    filename: str,
    *,
    percentage: bool = False,
) -> None:
    ranked = top_and_bottom(performance, metric).copy()
    values = ranked[metric] * 100 if percentage else ranked[metric]
    labels = "Store " + ranked["store_nbr"].astype(str)
    colors = ranked["ranking_group"].map({"Bottom": "#d95f5f", "Top": "#2a9d8f"})
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels, values, color=colors)
    ax.set(title=title, xlabel=xlabel, ylabel="")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_figures(performance: pd.DataFrame, thresholds: dict[str, float]) -> None:
    """Create rankings, segmentation, and attribute-level comparisons."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _plot_ranking(
        performance,
        "average_daily_sales",
        "Top and bottom stores: average daily sales",
        "Average daily sales",
        "average_daily_sales_top_bottom.png",
    )
    _plot_ranking(
        performance,
        "growth_rate",
        f"Top and bottom stores: {GROWTH_WINDOW_DAYS}-day window growth",
        "Growth rate (%)",
        "growth_rate_top_bottom.png",
        percentage=True,
    )
    _plot_ranking(
        performance,
        "coefficient_of_variation",
        "Top and bottom stores: normalized volatility",
        "Coefficient of variation",
        "volatility_top_bottom.png",
    )
    _plot_ranking(
        performance,
        "sales_volume_per_transaction",
        "Top and bottom stores: sales volume per transaction",
        "Sales volume per transaction",
        "sales_volume_per_transaction_top_bottom.png",
    )

    colors = {
        "High sales – stable": "#2a9d8f",
        "High sales – volatile": "#e9c46a",
        "Low sales – stable": "#457b9d",
        "Low sales – volatile": "#e76f51",
    }
    fig, ax = plt.subplots(figsize=(10, 7))
    for segment, rows in performance.groupby("performance_segment", sort=True):
        ax.scatter(
            rows["average_daily_sales"],
            rows["coefficient_of_variation"],
            label=segment,
            color=colors[segment],
            alpha=0.85,
        )
    ax.axvline(thresholds["sales_threshold_median"], color="black", linestyle="--")
    ax.axhline(
        thresholds["volatility_threshold_median"], color="black", linestyle="--"
    )
    ax.set(
        title="Store segments using median thresholds",
        xlabel="Average daily sales",
        ylabel="Coefficient of variation",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "store_segments.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    for attribute in ["city", "state", "store_type", "cluster"]:
        summary = (
            performance.groupby(attribute, observed=True)["average_daily_sales"]
            .mean()
            .sort_values()
            .tail(15)
        )
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(summary.index.astype(str), summary.values, color="#457b9d")
        ax.set(
            title=f"Average store daily sales by {attribute}",
            xlabel="Mean of store average daily sales",
            ylabel=attribute,
        )
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / f"average_daily_sales_by_{attribute}.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)


def create_findings(
    performance: pd.DataFrame, thresholds: dict[str, float]
) -> list[str]:
    """Create five reproducible findings directly from computed metrics."""
    avg_leader = performance.loc[performance["average_daily_sales"].idxmax()]
    growth_leader = performance.loc[performance["growth_rate"].idxmax()]
    growth_laggard = performance.loc[performance["growth_rate"].idxmin()]
    volatility_leader = performance.loc[
        performance["coefficient_of_variation"].idxmax()
    ]
    transaction_leader = performance.loc[
        performance["sales_volume_per_transaction"].idxmax()
    ]
    undefined_growth_count = int(performance["growth_rate"].isna().sum())
    segment_counts = performance["performance_segment"].value_counts().sort_index()
    counts_text = ", ".join(f"{name}: {count}" for name, count in segment_counts.items())
    return [
        f"Store {int(avg_leader.store_nbr)} has the highest average daily sales "
        f"at {avg_leader.average_daily_sales:,.2f}.",
        f"Store {int(growth_leader.store_nbr)} has the strongest growth "
        f"({growth_leader.growth_rate:.1%}), while store "
        f"{int(growth_laggard.store_nbr)} has the weakest "
        f"({growth_laggard.growth_rate:.1%}), comparing the first and last "
        f"{GROWTH_WINDOW_DAYS} observed days; {undefined_growth_count} stores "
        f"with a zero baseline are left unranked.",
        f"Store {int(volatility_leader.store_nbr)} is the most volatile after "
        f"normalizing for scale, with a coefficient of variation of "
        f"{volatility_leader.coefficient_of_variation:.3f}.",
        f"Store {int(transaction_leader.store_nbr)} has the highest sales volume "
        f"per transaction at {transaction_leader.sales_volume_per_transaction:,.2f}.",
        f"Median thresholds are average daily sales "
        f"{thresholds['sales_threshold_median']:,.2f} and coefficient of variation "
        f"{thresholds['volatility_threshold_median']:.3f}; segment counts are "
        f"{counts_text}.",
    ]


def main() -> None:
    """Run the complete store-level business EDA and save its artifacts."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    sales, transactions, stores, dates = load_inputs()
    performance, thresholds = build_store_performance(
        sales, transactions, stores, dates
    )
    performance.to_csv(STORE_PERFORMANCE_PATH, index=False)
    create_figures(performance, thresholds)
    findings = create_findings(performance, thresholds)
    FINDINGS_PATH.write_text(
        "# Store Performance Findings\n\n"
        f"Growth window: first versus last {GROWTH_WINDOW_DAYS} observed store-days.\n\n"
        f"Segmentation thresholds: median average daily sales = "
        f"{thresholds['sales_threshold_median']:.6f}; median coefficient of "
        f"variation = {thresholds['volatility_threshold_median']:.6f}.\n\n"
        + "\n".join(f"- {finding}" for finding in findings)
        + "\n",
        encoding="utf-8",
    )
    print(performance.to_string(index=False))
    print("\nFindings")
    for finding in findings:
        print(f"- {finding}")
    print(f"\nTable: {STORE_PERFORMANCE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Figures: {FIGURE_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
