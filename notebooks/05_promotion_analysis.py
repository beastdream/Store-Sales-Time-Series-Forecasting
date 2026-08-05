# %% [markdown]
# # Promotion Analysis
#
# This notebook compares observations with `is_promotion = 1` and
# `is_promotion = 0`. The reported percentage is strictly a **promotion uplift
# proxy**: it is a descriptive association and is not evidence that promotion
# causes sales to increase.

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

OUTPUT_PATH = TABLES_DIR / "promotion_analysis_basic.csv"
MATCHED_OUTPUT_PATH = TABLES_DIR / "promotion_analysis_matched.csv"
INTENSITY_PATH = TABLES_DIR / "promotion_onpromotion_intensity.csv"
LIMITATIONS_PATH = TABLES_DIR / "promotion_analysis_limitations.md"
FIGURE_DIR = FIGURES_DIR / "promotion_analysis"
MIN_COHORT_OBSERVATIONS = 100
TOP_N_FAMILIES = 15
DIMENSIONS = ["family", "store_nbr", "store_type", "day_of_week", "month"]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide numeric series and return NaN for zero or non-finite results."""
    result = numerator.astype("float64").div(
        denominator.astype("float64").replace(0, np.nan)
    )
    return result.where(np.isfinite(result))


def load_analysis_data() -> pd.DataFrame:
    """Load and enrich sales facts at their original date-store-family grain."""
    sales = pd.read_parquet(
        DATA_PROCESSED / "fact_daily_sales.parquet",
        columns=[
            "date_key",
            "store_key",
            "family_key",
            "sales",
            "onpromotion",
            "is_promotion",
        ],
    )
    families = pd.read_parquet(
        DATA_PROCESSED / "dim_family.parquet",
        columns=["family_key", "family"],
    )
    stores = pd.read_parquet(
        DATA_PROCESSED / "dim_store.parquet",
        columns=["store_key", "store_nbr", "store_type"],
    )
    dates = pd.read_parquet(
        DATA_PROCESSED / "dim_date.parquet",
        columns=["date_key", "year", "day_of_week", "month"],
    )
    enriched = (
        sales.merge(families, on="family_key", validate="many_to_one")
        .merge(stores, on="store_key", validate="many_to_one")
        .merge(dates, on="date_key", validate="many_to_one")
    )
    if len(enriched) != len(sales):
        raise AssertionError("Dimension joins changed the sales fact row count")
    if enriched[DIMENSIONS].isna().any().any():
        raise ValueError("At least one analysis dimension could not be mapped")
    enriched["sales"] = enriched["sales"].astype("float64")
    enriched["sales_promo"] = enriched["sales"].where(
        enriched["is_promotion"].eq(1)
    )
    enriched["sales_nonpromo"] = enriched["sales"].where(
        enriched["is_promotion"].eq(0)
    )
    enriched["onpromotion_promo"] = enriched["onpromotion"].where(
        enriched["is_promotion"].eq(1)
    )
    return enriched


def summarize_dimension(frame: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Summarize both promotion cohorts while retaining every observed group."""
    summary = (
        frame.groupby(dimension, as_index=False, observed=True, dropna=False)
        .agg(
            average_sales=("sales", "mean"),
            median_sales=("sales", "median"),
            observation_count=("sales", "size"),
            promotion_observation_count=("sales_promo", "count"),
            nonpromotion_observation_count=("sales_nonpromo", "count"),
            promotion_rate=("is_promotion", "mean"),
            avg_sales_promo=("sales_promo", "mean"),
            avg_sales_nonpromo=("sales_nonpromo", "mean"),
            median_sales_promo=("sales_promo", "median"),
            median_sales_nonpromo=("sales_nonpromo", "median"),
            total_onpromotion=("onpromotion", "sum"),
            average_onpromotion=("onpromotion", "mean"),
            median_onpromotion=("onpromotion", "median"),
            average_onpromotion_when_promoted=("onpromotion_promo", "mean"),
        )
    )
    summary["uplift_proxy_pct"] = (
        safe_divide(
            summary["avg_sales_promo"] - summary["avg_sales_nonpromo"],
            summary["avg_sales_nonpromo"],
        )
        * 100
    )
    small_promo = summary["promotion_observation_count"].lt(
        MIN_COHORT_OBSERVATIONS
    )
    small_nonpromo = summary["nonpromotion_observation_count"].lt(
        MIN_COHORT_OBSERVATIONS
    )
    summary["small_sample_warning"] = np.select(
        [small_promo & small_nonpromo, small_promo, small_nonpromo],
        [
            "Both cohorts below minimum sample size",
            "Promotion cohort below minimum sample size",
            "Non-promotion cohort below minimum sample size",
        ],
        default="",
    )
    summary["minimum_cohort_observations"] = MIN_COHORT_OBSERVATIONS
    summary.insert(0, "group_value", summary[dimension].astype(str))
    summary.insert(0, "analysis_dimension", dimension)
    for name in DIMENSIONS:
        if name == dimension:
            summary[name] = summary[name].astype("object")
        else:
            summary[name] = pd.Series(None, index=summary.index, dtype="object")
    return summary


def build_basic_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    """Combine all five requested grouping dimensions into one stable table."""
    summaries = [summarize_dimension(frame, dimension) for dimension in DIMENSIONS]
    result = pd.concat(summaries, ignore_index=True, sort=False)
    columns = [
        "analysis_dimension",
        "group_value",
        *DIMENSIONS,
        "average_sales",
        "median_sales",
        "observation_count",
        "promotion_observation_count",
        "nonpromotion_observation_count",
        "promotion_rate",
        "avg_sales_promo",
        "avg_sales_nonpromo",
        "median_sales_promo",
        "median_sales_nonpromo",
        "uplift_proxy_pct",
        "total_onpromotion",
        "average_onpromotion",
        "median_onpromotion",
        "average_onpromotion_when_promoted",
        "small_sample_warning",
        "minimum_cohort_observations",
    ]
    return result[columns]


def build_onpromotion_intensity(frame: pd.DataFrame) -> pd.DataFrame:
    """Profile sales across promoted-item-count bands, including the zero band."""
    bins = [-1, 0, 1, 5, 10, 25, 50, np.inf]
    labels = ["0", "1", "2–5", "6–10", "11–25", "26–50", "51+"]
    intensity = frame.assign(
        onpromotion_band=pd.cut(
            frame["onpromotion"], bins=bins, labels=labels, ordered=True
        )
    )
    result = (
        intensity.groupby("onpromotion_band", as_index=False, observed=False)
        .agg(
            average_sales=("sales", "mean"),
            median_sales=("sales", "median"),
            observation_count=("sales", "size"),
            average_onpromotion=("onpromotion", "mean"),
        )
    )
    result["small_sample_warning"] = np.where(
        result["observation_count"].lt(MIN_COHORT_OBSERVATIONS),
        "Band below minimum sample size",
        "",
    )
    return result


def create_figures(
    basic: pd.DataFrame, intensity: pd.DataFrame
) -> None:
    """Create readable promotion comparisons with sample sizes in labels."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    family = basic.loc[basic["analysis_dimension"].eq("family")].copy()
    eligible = family.loc[
        family["uplift_proxy_pct"].notna()
        & family["promotion_observation_count"].ge(MIN_COHORT_OBSERVATIONS)
        & family["nonpromotion_observation_count"].ge(MIN_COHORT_OBSERVATIONS)
    ]
    top = eligible.nlargest(TOP_N_FAMILIES, "uplift_proxy_pct").sort_values(
        "uplift_proxy_pct"
    )
    labels = [
        f"{row.family} (nₚ={row.promotion_observation_count:,}, "
        f"nₙ={row.nonpromotion_observation_count:,})"
        for row in top.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(labels, top["uplift_proxy_pct"], color="#2a9d8f")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(
        title="Top families by promotion uplift proxy with cohort sample sizes",
        xlabel="Promotion uplift proxy (%)",
        ylabel="",
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "top_family_uplift_proxy_with_sample_size.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def build_matched_analysis(
    frame: pd.DataFrame, basic: pd.DataFrame
) -> pd.DataFrame:
    """Compare cohorts inside contemporaneous store-family-calendar cells."""
    grain = ["store_nbr", "family", "year", "month", "day_of_week"]
    detailed = (
        frame.groupby(grain, as_index=False, observed=True)
        .agg(
            avg_sales_promo=("sales_promo", "mean"),
            avg_sales_nonpromo=("sales_nonpromo", "mean"),
            promotion_observation_count=("sales_promo", "count"),
            nonpromotion_observation_count=("sales_nonpromo", "count"),
        )
    )
    # Matching requires both cohorts, but no eligible cell is trimmed as an outlier.
    detailed = detailed.loc[
        detailed["promotion_observation_count"].gt(0)
        & detailed["nonpromotion_observation_count"].gt(0)
    ].copy()
    store_types = frame[["store_nbr", "store_type"]].drop_duplicates()
    if store_types["store_nbr"].duplicated().any():
        raise AssertionError("store_nbr does not map to exactly one store_type")
    detailed = detailed.merge(store_types, on="store_nbr", validate="many_to_one")
    detailed["matched_uplift_proxy_pct"] = (
        safe_divide(
            detailed["avg_sales_promo"] - detailed["avg_sales_nonpromo"],
            detailed["avg_sales_nonpromo"],
        )
        * 100
    )
    detailed["matched_group_count"] = 1
    detailed["small_sample_warning"] = np.where(
        detailed[["promotion_observation_count", "nonpromotion_observation_count"]]
        .min(axis=1)
        .lt(MIN_COHORT_OBSERVATIONS),
        "At least one matched cohort below minimum sample size",
        "",
    )
    detailed["analysis_level"] = "matched_group"
    detailed["unmatched_uplift_proxy_pct"] = np.nan
    detailed["matched_vs_unmatched_difference_pct_points"] = np.nan

    def aggregate(level: str) -> pd.DataFrame:
        weighted = detailed.assign(
            promo_sales_sum=(
                detailed["avg_sales_promo"]
                * detailed["promotion_observation_count"]
            ),
            nonpromo_sales_sum=(
                detailed["avg_sales_nonpromo"]
                * detailed["nonpromotion_observation_count"]
            ),
        )
        summary = (
            weighted.groupby(level, as_index=False, observed=True)
            .agg(
                promo_sales_sum=("promo_sales_sum", "sum"),
                nonpromo_sales_sum=("nonpromo_sales_sum", "sum"),
                promotion_observation_count=("promotion_observation_count", "sum"),
                nonpromotion_observation_count=(
                    "nonpromotion_observation_count",
                    "sum",
                ),
                matched_group_count=("matched_group_count", "sum"),
            )
        )
        summary["avg_sales_promo"] = safe_divide(
            summary["promo_sales_sum"], summary["promotion_observation_count"]
        )
        summary["avg_sales_nonpromo"] = safe_divide(
            summary["nonpromo_sales_sum"],
            summary["nonpromotion_observation_count"],
        )
        summary["matched_uplift_proxy_pct"] = (
            safe_divide(
                summary["avg_sales_promo"] - summary["avg_sales_nonpromo"],
                summary["avg_sales_nonpromo"],
            )
            * 100
        )
        unmatched = basic.loc[
            basic["analysis_dimension"].eq(level),
            ["group_value", "uplift_proxy_pct"],
        ].rename(
            columns={
                "group_value": level,
                "uplift_proxy_pct": "unmatched_uplift_proxy_pct",
            }
        )
        unmatched[level] = unmatched[level].astype(str)
        summary[level] = summary[level].astype(str)
        summary = summary.merge(unmatched, on=level, validate="one_to_one")
        summary["matched_vs_unmatched_difference_pct_points"] = (
            summary["matched_uplift_proxy_pct"]
            - summary["unmatched_uplift_proxy_pct"]
        )
        summary["small_sample_warning"] = np.where(
            summary[
                ["promotion_observation_count", "nonpromotion_observation_count"]
            ]
            .min(axis=1)
            .lt(MIN_COHORT_OBSERVATIONS),
            "At least one aggregated matched cohort below minimum sample size",
            "",
        )
        summary["analysis_level"] = f"{level}_summary"
        summary = summary.drop(columns=["promo_sales_sum", "nonpromo_sales_sum"])
        return summary

    family_summary = aggregate("family")
    store_type_summary = aggregate("store_type")
    all_columns = [
        "analysis_level",
        "store_nbr",
        "family",
        "store_type",
        "year",
        "month",
        "day_of_week",
        "avg_sales_promo",
        "avg_sales_nonpromo",
        "promotion_observation_count",
        "nonpromotion_observation_count",
        "matched_group_count",
        "matched_uplift_proxy_pct",
        "unmatched_uplift_proxy_pct",
        "matched_vs_unmatched_difference_pct_points",
        "small_sample_warning",
        "minimum_cohort_observations",
        "matching_rule",
        "outlier_handling",
    ]
    frames = [detailed, family_summary, store_type_summary]
    for result in frames:
        for column in all_columns:
            if column not in result:
                result[column] = pd.NA
        result["minimum_cohort_observations"] = MIN_COHORT_OBSERVATIONS
        result["matching_rule"] = (
            "Both cohorts within store_nbr + family + year + month + day_of_week"
        )
        result["outlier_handling"] = "No automatic outlier removal"
    combined = pd.concat(
        [result[all_columns] for result in frames], ignore_index=True, sort=False
    )
    if np.isinf(combined.select_dtypes("number").to_numpy()).any():
        raise AssertionError("Matched analysis contains an infinite result")
    return combined


def create_matched_comparison_figure(matched: pd.DataFrame) -> None:
    """Compare matched and unmatched family proxies with sample sizes."""
    family = matched.loc[matched["analysis_level"].eq("family_summary")].copy()
    family["total_matched_observations"] = (
        family["promotion_observation_count"]
        + family["nonpromotion_observation_count"]
    )
    plotted = family.nlargest(TOP_N_FAMILIES, "total_matched_observations").sort_values(
        "matched_uplift_proxy_pct"
    )
    positions = np.arange(len(plotted))
    labels = [
        f"{row.family} (n={int(row.total_matched_observations):,})"
        for row in plotted.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(
        positions - 0.18,
        plotted["matched_uplift_proxy_pct"],
        height=0.34,
        label="Matched",
        color="#2a9d8f",
    )
    ax.barh(
        positions + 0.18,
        plotted["unmatched_uplift_proxy_pct"],
        height=0.34,
        label="Unmatched",
        color="#e9c46a",
    )
    ax.set_yticks(positions, labels)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(
        title="Matched versus unmatched promotion uplift proxy by family",
        xlabel="Promotion uplift proxy (%)",
        ylabel="",
    )
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "matched_vs_unmatched_family.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def create_overall_figures(
    basic: pd.DataFrame, intensity: pd.DataFrame
) -> None:
    """Create the overall cohort and onpromotion-intensity figures."""
    family = basic.loc[basic["analysis_dimension"].eq("family")].copy()
    promo_average = (
        family["avg_sales_promo"] * family["promotion_observation_count"]
    ).sum() / family["promotion_observation_count"].sum()
    nonpromo_average = (
        family["avg_sales_nonpromo"] * family["nonpromotion_observation_count"]
    ).sum() / family["nonpromotion_observation_count"].sum()
    overall = pd.DataFrame(
        {
            "cohort": ["is_promotion = 0", "is_promotion = 1"],
            "average_sales": [nonpromo_average, promo_average],
        }
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(overall["cohort"], overall["average_sales"], color=["#457b9d", "#e9c46a"])
    ax.set(title="Average sales by promotion cohort", xlabel="Average sales", ylabel="")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "average_sales_by_promotion_cohort.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    plotted = intensity.sort_values("onpromotion_band", ascending=False)
    labels = [
        f"{row.onpromotion_band} (n={row.observation_count:,})"
        for row in plotted.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(labels, plotted["average_sales"], color="#e76f51")
    ax.set(
        title="Average sales by onpromotion item-count band",
        xlabel="Average sales",
        ylabel="Promoted item count (sample size)",
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "sales_by_onpromotion_intensity.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_limitations(basic: pd.DataFrame, matched: pd.DataFrame) -> None:
    """Persist interpretation constraints and sample-size coverage."""
    warned = int(basic["small_sample_warning"].ne("").sum())
    undefined = int(basic["uplift_proxy_pct"].isna().sum())
    matched_groups = int(matched["analysis_level"].eq("matched_group").sum())
    matched_warned = int(
        matched.loc[
            matched["analysis_level"].eq("matched_group"), "small_sample_warning"
        ]
        .ne("")
        .sum()
    )
    LIMITATIONS_PATH.write_text(
        "# Promotion Analysis Limitations\n\n"
        "- The promotion uplift proxy is descriptive only. It does not establish "
        "that promotion causes sales to increase because promotion assignment is "
        "not randomized.\n"
        "- Product selection, store, seasonality, holidays, pricing, and underlying "
        "demand may differ between promotion and non-promotion observations.\n"
        "- Aggregated comparisons can hide variation within a family, store, or "
        "calendar group.\n"
        "- The matched comparison is more comparable than the unmatched overall "
        "comparison because it holds store, family, year, month, and day of week "
        "constant. It reduces observed composition and calendar-mix differences, "
        "but it is still not causal inference.\n"
        f"- Matching retains {matched_groups:,} cells containing both cohorts. It "
        "uses only observations inside each contemporaneous cell and does not use "
        "future data. Cells missing either cohort cannot contribute to matched "
        "results, so selection into the matched sample remains a limitation.\n"
        "- No outlier is automatically removed. Extreme values remain in the "
        "averages and promotion uplift proxy; robust sensitivity analysis would be "
        "needed before choosing any documented exclusion rule.\n"
        f"- Cohorts with fewer than {MIN_COHORT_OBSERVATIONS} observations are "
        f"flagged; {warned} unmatched grouped rows and {matched_warned:,} detailed "
        "matched cells have at least one small cohort. Aggregated family and store-"
        "type sample counts are also retained in the matched table.\n"
        f"- Division by a zero non-promotion average is left undefined; {undefined} "
        f"unmatched grouped {'row has' if undefined == 1 else 'rows have'} no "
        "finite promotion uplift proxy.\n"
        "- `onpromotion` is a count of promoted items, not promotion depth, discount "
        "size, or campaign exposure. Its association with sales is not causal.\n",
        encoding="utf-8",
    )


# %% [markdown]
# ## Run analysis and save artifacts

# %%
def main() -> None:
    """Run the promotion analysis and create all requested artifacts."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_analysis_data()
    basic = build_basic_analysis(frame)
    matched = build_matched_analysis(frame, basic)
    intensity = build_onpromotion_intensity(frame)
    if len(basic) != sum(frame[dimension].nunique() for dimension in DIMENSIONS):
        raise AssertionError("At least one observed dimension group was lost")
    if np.isinf(basic.select_dtypes("number").to_numpy()).any():
        raise AssertionError("Promotion analysis contains an infinite result")
    basic.to_csv(OUTPUT_PATH, index=False)
    matched.to_csv(MATCHED_OUTPUT_PATH, index=False)
    intensity.to_csv(INTENSITY_PATH, index=False)
    create_figures(basic, intensity)
    create_overall_figures(basic, intensity)
    create_matched_comparison_figure(matched)
    write_limitations(basic, matched)
    print(f"Rows: {len(basic)}")
    print(f"Small-cohort warnings: {basic['small_sample_warning'].ne('').sum()}")
    print(f"Undefined promotion uplift proxy: {basic['uplift_proxy_pct'].isna().sum()}")
    print(f"Table: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Matched table: {MATCHED_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Figures: {FIGURE_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Limitations
#
# - The promotion uplift proxy is descriptive, not causal. Promotion assignment
#   is not randomized, so this notebook does not conclude that promotion causes
#   sales to increase.
# - Family mix, store mix, seasonality, holidays, price, and baseline demand can
#   confound comparisons between promotion and non-promotion observations.
# - Groups with fewer than 100 observations in either cohort are explicitly
#   flagged and should not be interpreted as stable estimates.
# - A zero non-promotion average produces an undefined proxy rather than an
#   infinite result.
# - `onpromotion` measures the number of promoted items, not discount depth or
#   campaign exposure; its relationship with sales is also descriptive.
# - Matching within store, family, year, month, and day of week improves
#   comparability by reducing observed composition and calendar-mix differences,
#   but it is still not causal inference and cannot remove unobserved confounding.
# - Matching uses only contemporaneous observations and no future data. Cells
#   without both cohorts do not enter matched summaries.
# - No outlier is removed automatically. Any later exclusion would need a stated,
#   defensible rule and a sensitivity comparison.
