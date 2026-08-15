# %% [markdown]
# # Business EDA: Store Performance
#
# This analysis compares stores on scale, growth, volatility, and transaction
# efficiency. Recent growth uses adjacent calendar windows anchored to the
# latest observed sales date. The legacy first-versus-last metric is retained
# explicitly as a proxy.

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
FAMILY_PERFORMANCE_PATH = TABLES_DIR / "family_performance.csv"
FAMILY_READINESS_PATH = TABLES_DIR / "family_forecast_readiness.csv"
FAMILY_FINDINGS_PATH = TABLES_DIR / "family_performance_findings.md"
FAMILY_FIGURE_DIR = FIGURES_DIR / "business_eda" / "family"
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
    """Calculate legacy and recent store growth without imputing missing dates."""
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
    growth["first_vs_last_90d_growth_proxy"] = safe_divide(
        growth["last_window_average_daily_sales"]
        - growth["first_window_average_daily_sales"],
        growth["first_window_average_daily_sales"],
    )

    latest_date = ordered["full_date"].max()
    recent_start = latest_date - pd.Timedelta(days=GROWTH_WINDOW_DAYS - 1)
    previous_end = recent_start - pd.Timedelta(days=1)
    previous_start = previous_end - pd.Timedelta(days=GROWTH_WINDOW_DAYS - 1)
    yoy_start = recent_start - pd.DateOffset(years=1)
    yoy_end = latest_date - pd.DateOffset(years=1)

    def aggregate_window(start: pd.Timestamp, end: pd.Timestamp, prefix: str) -> pd.DataFrame:
        observed = ordered.loc[ordered["full_date"].between(start, end)]
        return (
            observed.groupby("store_key", as_index=False, observed=True)
            .agg(
                **{
                    f"{prefix}_sales": ("daily_sales", "sum"),
                    f"{prefix}_observed_days": ("full_date", "nunique"),
                }
            )
        )

    for window in [
        aggregate_window(recent_start, latest_date, "recent_90d"),
        aggregate_window(previous_start, previous_end, "previous_90d"),
        aggregate_window(yoy_start, yoy_end, "yoy_90d"),
    ]:
        growth = growth.merge(window, on="store_key", how="left", validate="one_to_one")

    count_columns = [
        "recent_90d_observed_days",
        "previous_90d_observed_days",
        "yoy_90d_observed_days",
    ]
    growth[count_columns] = growth[count_columns].fillna(0).astype("int64")
    growth["recent_90d_growth"] = safe_divide(
        growth["recent_90d_sales"] - growth["previous_90d_sales"],
        growth["previous_90d_sales"],
    ).where(growth["previous_90d_observed_days"].gt(0))
    growth["has_yoy_comparison"] = growth["yoy_90d_observed_days"].gt(0).astype("uint8")
    growth["recent_90d_yoy_growth"] = safe_divide(
        growth["recent_90d_sales"] - growth["yoy_90d_sales"],
        growth["yoy_90d_sales"],
    ).where(growth["has_yoy_comparison"].eq(1))

    window_dates = {
        "recent_90d_start_date": recent_start,
        "recent_90d_end_date": latest_date,
        "previous_90d_start_date": previous_start,
        "previous_90d_end_date": previous_end,
        "yoy_90d_start_date": yoy_start,
        "yoy_90d_end_date": yoy_end,
    }
    for column, value in window_dates.items():
        growth[column] = value
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
        "recent_90d_growth": "rank_recent_90d_growth",
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
        "first_vs_last_90d_growth_proxy",
        "first_window_average_daily_sales",
        "last_window_average_daily_sales",
        "recent_90d_growth",
        "recent_90d_yoy_growth",
        "has_yoy_comparison",
        "recent_90d_sales",
        "previous_90d_sales",
        "yoy_90d_sales",
        "recent_90d_observed_days",
        "previous_90d_observed_days",
        "yoy_90d_observed_days",
        "recent_90d_start_date",
        "recent_90d_end_date",
        "previous_90d_start_date",
        "previous_90d_end_date",
        "yoy_90d_start_date",
        "yoy_90d_end_date",
        "sales_threshold_median",
        "volatility_threshold_median",
        "performance_segment",
        *ranking_metrics.values(),
    ]
    performance = performance[output_columns].sort_values("store_nbr").reset_index(drop=True)

    if performance["store_nbr"].duplicated().any():
        raise AssertionError("Store performance grain is not unique")
    if not np.isclose(
        performance["total_sales"].sum(),
        sales["sales"].sum(),
        rtol=0,
        atol=1e-6,
    ):
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
        "recent_90d_growth",
        f"Top and bottom stores: recent {GROWTH_WINDOW_DAYS}-day growth",
        "Growth rate (%)",
        "recent_90d_growth_top_bottom.png",
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
    growth_leader = performance.loc[performance["recent_90d_growth"].idxmax()]
    growth_laggard = performance.loc[performance["recent_90d_growth"].idxmin()]
    volatility_leader = performance.loc[
        performance["coefficient_of_variation"].idxmax()
    ]
    transaction_leader = performance.loc[
        performance["sales_volume_per_transaction"].idxmax()
    ]
    undefined_growth_count = int(performance["recent_90d_growth"].isna().sum())
    segment_counts = performance["performance_segment"].value_counts().sort_index()
    counts_text = ", ".join(f"{name}: {count}" for name, count in segment_counts.items())
    return [
        f"Store {int(avg_leader.store_nbr)} has the highest average daily sales "
        f"at {avg_leader.average_daily_sales:,.2f}.",
        f"Store {int(growth_leader.store_nbr)} has the strongest growth "
        f"({growth_leader.recent_90d_growth:.1%}), while store "
        f"{int(growth_laggard.store_nbr)} has the weakest "
        f"({growth_laggard.recent_90d_growth:.1%}), comparing the latest "
        f"{GROWTH_WINDOW_DAYS} calendar days with the immediately preceding "
        f"window; {undefined_growth_count} stores "
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


def load_family_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load family-level inputs while retaining every zero-sales observation."""
    sales = pd.read_parquet(
        DATA_PROCESSED / "fact_daily_sales.parquet",
        columns=[
            "date_key",
            "store_key",
            "family_key",
            "sales",
            "is_promotion",
        ],
    )
    families = pd.read_parquet(
        DATA_PROCESSED / "dim_family.parquet",
        columns=["family_key", "family"],
    )
    dates = pd.read_parquet(
        DATA_PROCESSED / "dim_date.parquet",
        columns=["date_key", "year", "month"],
    )
    return sales, families, dates


def build_family_performance(
    sales: pd.DataFrame,
    families: pd.DataFrame,
    dates: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate family metrics without filtering zero-sales rows."""
    enriched = (
        sales.assign(sales=sales["sales"].astype("float64"))
        .merge(families, on="family_key", validate="many_to_one")
        .merge(dates, on="date_key", validate="many_to_one")
    )
    if len(enriched) != len(sales):
        raise AssertionError("Family enrichment changed the sales fact row count")

    performance = (
        enriched.groupby("family", as_index=False, observed=True)
        .agg(
            total_sales=("sales", "sum"),
            average_sales=("sales", "mean"),
            median_sales=("sales", "median"),
            sales_std=("sales", "std"),
            zero_sales_rate=("sales", lambda values: values.eq(0).mean()),
            promotion_rate=("is_promotion", "mean"),
        )
    )
    performance["coefficient_of_variation"] = safe_divide(
        performance["sales_std"], performance["average_sales"]
    )

    active_stores = (
        enriched.assign(has_positive_sales=enriched["sales"].gt(0))
        .groupby(["family", "store_key"], as_index=False, observed=True)
        .agg(has_positive_sales=("has_positive_sales", "max"))
        .groupby("family", as_index=False, observed=True)
        .agg(number_of_active_stores=("has_positive_sales", "sum"))
    )
    monthly = (
        enriched.groupby(["family", "year", "month"], as_index=False, observed=True)
        .agg(monthly_sales=("sales", "sum"))
        .sort_values(["family", "year", "month"])
    )
    monthly["previous_month_sales"] = monthly.groupby(
        "family", observed=True
    )["monthly_sales"].shift()
    monthly["month_over_month_growth"] = safe_divide(
        monthly["monthly_sales"] - monthly["previous_month_sales"],
        monthly["previous_month_sales"],
    )
    monthly_growth = (
        monthly.groupby("family", as_index=False, observed=True)
        .agg(monthly_growth=("month_over_month_growth", "median"))
    )
    performance = (
        performance.merge(active_stores, on="family", validate="one_to_one")
        .merge(monthly_growth, on="family", validate="one_to_one")
        .sort_values("family")
        .reset_index(drop=True)
    )
    performance["sales_contribution_rate"] = safe_divide(
        performance["total_sales"],
        pd.Series(performance["total_sales"].sum(), index=performance.index),
    )
    performance["rank_total_sales"] = performance["total_sales"].rank(
        method="min", ascending=False
    ).astype("int64")
    performance["rank_volatility"] = performance[
        "coefficient_of_variation"
    ].rank(method="min", ascending=False, na_option="keep").astype("Int64")
    performance["rank_zero_sales_rate"] = performance["zero_sales_rate"].rank(
        method="min", ascending=False
    ).astype("int64")
    performance["rank_promotion_rate"] = performance["promotion_rate"].rank(
        method="min", ascending=False
    ).astype("int64")

    performance = performance[
        [
            "family",
            "total_sales",
            "average_sales",
            "median_sales",
            "sales_std",
            "coefficient_of_variation",
            "zero_sales_rate",
            "promotion_rate",
            "number_of_active_stores",
            "monthly_growth",
            "sales_contribution_rate",
            "rank_total_sales",
            "rank_volatility",
            "rank_zero_sales_rate",
            "rank_promotion_rate",
        ]
    ]

    if performance["family"].duplicated().any():
        raise AssertionError("Family performance grain is not unique")
    if len(performance) != len(families):
        raise AssertionError("A family was lost from the performance table")
    if not np.isclose(
        performance["total_sales"].sum(),
        sales["sales"].sum(),
        rtol=0,
        atol=1e-6,
    ):
        raise AssertionError("Total sales changed during family aggregation")
    return performance


def build_family_readiness(
    performance: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Assign mutually exclusive readiness segments using explicit thresholds."""
    thresholds = {
        "volume_threshold_median": float(performance["total_sales"].median()),
        "volatility_threshold_median": float(
            performance["coefficient_of_variation"].median()
        ),
        "intermittency_threshold_q75": float(
            performance["zero_sales_rate"].quantile(0.75)
        ),
        "promotion_dependency_threshold_q75": float(
            performance["promotion_rate"].quantile(0.75)
        ),
    }
    high_volume = performance["total_sales"].ge(
        thresholds["volume_threshold_median"]
    )
    volatile = performance["coefficient_of_variation"].gt(
        thresholds["volatility_threshold_median"]
    )
    intermittent = performance["zero_sales_rate"].ge(
        thresholds["intermittency_threshold_q75"]
    )
    promotion_dependent = performance["promotion_rate"].ge(
        thresholds["promotion_dependency_threshold_q75"]
    )
    readiness = performance.copy()
    readiness["forecast_readiness"] = np.select(
        [
            promotion_dependent,
            high_volume & ~volatile,
            high_volume & volatile,
            ~high_volume & intermittent,
        ],
        [
            "Promotion dependent",
            "High volume – stable",
            "High volume – volatile",
            "Low volume – intermittent",
        ],
        default="Low volume – stable",
    )
    for name, value in thresholds.items():
        readiness[name] = value
    readiness["segmentation_rule"] = np.select(
        [
            promotion_dependent,
            high_volume & ~volatile,
            high_volume & volatile,
            ~high_volume & intermittent,
        ],
        [
            "promotion_rate >= Q75",
            "total_sales >= median and CV <= median",
            "total_sales >= median and CV > median",
            "total_sales < median and zero_sales_rate >= Q75",
        ],
        default="total_sales < median and zero_sales_rate < Q75",
    )
    return readiness, thresholds


def _plot_family_metric(
    performance: pd.DataFrame,
    metric: str,
    title: str,
    xlabel: str,
    filename: str,
    *,
    top_n: int = 15,
    percentage: bool = False,
) -> None:
    """Plot a readable horizontal ranking limited to the most relevant families."""
    ranked = performance.nlargest(top_n, metric).sort_values(metric)
    values = ranked[metric] * 100 if percentage else ranked[metric]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(ranked["family"], values, color="#2a9d8f")
    ax.set(title=title, xlabel=xlabel, ylabel="")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FAMILY_FIGURE_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_family_figures(
    performance: pd.DataFrame, readiness: pd.DataFrame
) -> None:
    """Create horizontal charts with at most 15 family labels per ranking."""
    FAMILY_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _plot_family_metric(
        performance,
        "total_sales",
        "Families contributing the most sales",
        "Total sales",
        "largest_sales_contributors.png",
    )
    _plot_family_metric(
        performance,
        "coefficient_of_variation",
        "Families with highest normalized volatility",
        "Coefficient of variation",
        "highest_volatility.png",
    )
    _plot_family_metric(
        performance,
        "zero_sales_rate",
        "Families with highest zero-sales rate",
        "Zero-sales rate (%)",
        "highest_zero_sales_rate.png",
        percentage=True,
    )
    _plot_family_metric(
        performance,
        "promotion_rate",
        "Families with highest promotion dependence signal",
        "Promotion observation rate (%)",
        "highest_promotion_rate.png",
        percentage=True,
    )
    counts = readiness["forecast_readiness"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(counts.index, counts.values, color="#457b9d")
    ax.set(
        title="Family forecast-readiness segments",
        xlabel="Number of families",
        ylabel="",
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        FAMILY_FIGURE_DIR / "forecast_readiness_segments.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def create_family_findings(
    performance: pd.DataFrame,
    readiness: pd.DataFrame,
    thresholds: dict[str, float],
) -> list[str]:
    """Generate five findings from the calculated family metrics."""
    sales_leader = performance.loc[performance["total_sales"].idxmax()]
    volatile = performance.loc[performance["coefficient_of_variation"].idxmax()]
    intermittent = performance.loc[performance["zero_sales_rate"].idxmax()]
    promotion = performance.loc[performance["promotion_rate"].idxmax()]
    segment_counts = readiness["forecast_readiness"].value_counts().sort_index()
    count_text = ", ".join(f"{name}: {count}" for name, count in segment_counts.items())
    return [
        f"{sales_leader.family} contributes the most sales at "
        f"{sales_leader.total_sales:,.2f} "
        f"({sales_leader.sales_contribution_rate:.1%} of all sales).",
        f"{volatile.family} has the highest normalized volatility, with a "
        f"coefficient of variation of {volatile.coefficient_of_variation:.3f}.",
        f"{intermittent.family} has the highest zero-sales rate at "
        f"{intermittent.zero_sales_rate:.1%}; zero-sales rows remain in all metrics.",
        f"{promotion.family} has the highest promotion observation rate at "
        f"{promotion.promotion_rate:.1%}.",
        f"Readiness segment counts are {count_text}; thresholds are median total "
        f"sales {thresholds['volume_threshold_median']:,.2f}, median CV "
        f"{thresholds['volatility_threshold_median']:.3f}, zero-sales Q75 "
        f"{thresholds['intermittency_threshold_q75']:.1%}, and promotion-rate Q75 "
        f"{thresholds['promotion_dependency_threshold_q75']:.1%}.",
    ]
def main() -> None:
    """Run store- and family-level business EDA and save all artifacts."""
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
        f"Legacy proxy: first versus last {GROWTH_WINDOW_DAYS} observed store-days; "
        "reported as `first_vs_last_90d_growth_proxy`, not recent growth.\n\n"
        f"Recent window: {performance['recent_90d_start_date'].iloc[0]:%Y-%m-%d} "
        f"through {performance['recent_90d_end_date'].iloc[0]:%Y-%m-%d}; previous "
        f"window: {performance['previous_90d_start_date'].iloc[0]:%Y-%m-%d} through "
        f"{performance['previous_90d_end_date'].iloc[0]:%Y-%m-%d}; YoY window: "
        f"{performance['yoy_90d_start_date'].iloc[0]:%Y-%m-%d} through "
        f"{performance['yoy_90d_end_date'].iloc[0]:%Y-%m-%d}. Only observed "
        "store-days are summed; missing observations are not filled with zero.\n\n"
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
    print(f"\nTable: {STORE_PERFORMANCE_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Figures: {FIGURE_DIR.relative_to(PROJECT_ROOT).as_posix()}")

    family_sales, families, family_dates = load_family_inputs()
    family_performance = build_family_performance(
        family_sales, families, family_dates
    )
    family_readiness, family_thresholds = build_family_readiness(
        family_performance
    )
    family_performance.to_csv(FAMILY_PERFORMANCE_PATH, index=False)
    family_readiness.to_csv(FAMILY_READINESS_PATH, index=False)
    create_family_figures(family_performance, family_readiness)
    family_findings = create_family_findings(
        family_performance, family_readiness, family_thresholds
    )
    FAMILY_FINDINGS_PATH.write_text(
        "# Family Performance and Forecast Readiness\n\n"
        "Monthly growth is the median of valid month-over-month changes; a "
        "transition whose prior month has zero sales is retained in the source "
        "but excluded from division.\n\n"
        "## Segmentation rules\n\n"
        "Rules are applied in priority order so every family receives exactly "
        "one segment:\n\n"
        "1. **Promotion dependent:** promotion rate is at or above Q75.\n"
        "2. **High volume – stable:** total sales is at or above the median and "
        "CV is at or below the median.\n"
        "3. **High volume – volatile:** total sales is at or above the median and "
        "CV is above the median.\n"
        "4. **Low volume – intermittent:** total sales is below the median and "
        "zero-sales rate is at or above Q75.\n"
        "5. **Low volume – stable:** remaining low-volume families.\n\n"
        f"Thresholds: total-sales median = "
        f"{family_thresholds['volume_threshold_median']:.6f}; CV median = "
        f"{family_thresholds['volatility_threshold_median']:.6f}; zero-sales "
        f"Q75 = {family_thresholds['intermittency_threshold_q75']:.6f}; "
        f"promotion-rate Q75 = "
        f"{family_thresholds['promotion_dependency_threshold_q75']:.6f}.\n\n"
        "## Findings\n\n"
        + "\n".join(f"- {finding}" for finding in family_findings)
        + "\n",
        encoding="utf-8",
    )
    print("\nFamily findings")
    for finding in family_findings:
        print(f"- {finding}")
    print(f"\nTable: {FAMILY_PERFORMANCE_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Readiness: {FAMILY_READINESS_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Figures: {FAMILY_FIGURE_DIR.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
