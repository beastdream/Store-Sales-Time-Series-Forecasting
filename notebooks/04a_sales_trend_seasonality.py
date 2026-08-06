# %% [markdown]
# # Sales Trend and Seasonality Analysis
#
# This notebook uses the processed sales fact and date dimension. It preserves a
# complete calendar and distinguishes a date with no sales rows from an observed
# date whose total sales is zero. Missing observations are never filled with zero.

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


DAILY_OUTPUT_PATH = TABLES_DIR / "daily_sales_summary.csv"
MONTHLY_OUTPUT_PATH = TABLES_DIR / "monthly_sales_summary.csv"
WEEKDAY_MONTH_OUTPUT_PATH = TABLES_DIR / "weekday_month_summary.csv"
FIGURE_DIR = FIGURES_DIR / "sales_trend_seasonality"


# %% [markdown]
# ## Load processed inputs
#
# `fact_daily_sales` supplies observed sales rows. `dim_date` supplies the complete
# analysis calendar and calendar attributes. No raw CSV is read or cleaned here.

# %%
def load_processed_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only processed columns required for trend and seasonality analysis."""
    sales = pd.read_parquet(
        DATA_PROCESSED / "fact_daily_sales.parquet",
        columns=["date_key", "sales"],
    )
    dates = pd.read_parquet(
        DATA_PROCESSED / "dim_date.parquet",
        columns=[
            "date_key",
            "full_date",
            "year",
            "month",
            "month_name",
            "day_of_week",
            "day_name",
            "is_weekend",
            "is_payday",
        ],
    )
    return sales, dates


# %% [markdown]
# ## Daily trend and missing-observation handling
#
# Moving averages use trailing calendar windows of 7 and 28 dates. Dates without
# sales observations remain `NaN` and are excluded from the window mean; they are
# not treated as zero. `has_sales_observation` makes this distinction explicit.

# %%
def build_daily_summary(
    sales: pd.DataFrame,
    dates: pd.DataFrame,
) -> pd.DataFrame:
    """Create a complete-calendar daily summary without zero-filling missing sales."""
    observed = (
        sales.groupby("date_key", as_index=False, observed=True)
        .agg(total_sales=("sales", "sum"), sales_row_count=("sales", "size"))
    )
    if observed["date_key"].duplicated().any():
        raise ValueError("Daily sales aggregation did not produce a unique date grain")

    daily = dates.merge(observed, on="date_key", how="left", validate="one_to_one")
    daily = daily.sort_values("full_date", kind="stable").reset_index(drop=True)
    daily["has_sales_observation"] = daily["sales_row_count"].notna().astype("uint8")
    daily["sales_ma_7"] = daily["total_sales"].rolling(7, min_periods=1).mean()
    daily["sales_ma_28"] = daily["total_sales"].rolling(28, min_periods=1).mean()
    daily = daily.rename(columns={"full_date": "date"}).drop(columns="sales_row_count")

    missing = daily["has_sales_observation"].eq(0)
    if daily.loc[missing, "total_sales"].notna().any():
        raise AssertionError("Missing sales dates must retain an undefined total_sales")
    if daily.loc[~missing, "total_sales"].isna().any():
        raise AssertionError("Observed sales dates must have a total_sales value")
    return daily


# %% [markdown]
# ## Weekly and monthly trend
#
# Weekly totals sum observed dates and retain observed/missing-day counts. Monthly
# growth is calculated only when both periods are complete observed calendar months.
# YoY growth also requires the same month one year earlier to be complete.

# %%
def build_weekly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate historical calendar dates to Monday-Sunday weekly periods."""
    last_observed = daily.loc[daily["has_sales_observation"].eq(1), "date"].max()
    historical = daily.loc[daily["date"].le(last_observed)].copy()
    historical["week_start"] = historical["date"].dt.to_period("W-SUN").dt.start_time
    weekly = (
        historical.groupby("week_start", as_index=False, observed=True)
        .agg(
            total_sales=("total_sales", lambda values: values.sum(min_count=1)),
            observed_days=("has_sales_observation", "sum"),
            calendar_days=("date", "size"),
        )
    )
    weekly["missing_sales_dates"] = weekly["calendar_days"] - weekly["observed_days"]
    return weekly


def _strict_growth(
    monthly: pd.DataFrame,
    comparison_lag: int,
) -> pd.Series:
    """Return growth only for complete, exactly aligned comparison months."""
    result = pd.Series(np.nan, index=monthly.index, dtype="float64")
    for index in range(comparison_lag, len(monthly)):
        current_period = monthly.loc[index, "year_month_period"]
        prior_period = monthly.loc[index - comparison_lag, "year_month_period"]
        if current_period - comparison_lag != prior_period:
            continue
        if not (
            monthly.loc[index, "is_complete_observation_month"]
            and monthly.loc[index - comparison_lag, "is_complete_observation_month"]
        ):
            continue
        prior_total = monthly.loc[index - comparison_lag, "total_sales"]
        if prior_total == 0 or pd.isna(prior_total):
            continue
        result.loc[index] = monthly.loc[index, "total_sales"] / prior_total - 1
    return result


def build_monthly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Create monthly totals plus strict MoM and YoY comparisons."""
    last_observed = daily.loc[daily["has_sales_observation"].eq(1), "date"].max()
    historical = daily.loc[daily["date"].le(last_observed)].copy()
    historical["year_month_period"] = historical["date"].dt.to_period("M")
    monthly = (
        historical.groupby("year_month_period", as_index=False, observed=True)
        .agg(
            year=("year", "first"),
            month=("month", "first"),
            month_name=("month_name", "first"),
            total_sales=("total_sales", lambda values: values.sum(min_count=1)),
            observed_days=("has_sales_observation", "sum"),
            calendar_days=("date", "size"),
        )
        .sort_values("year_month_period", kind="stable")
        .reset_index(drop=True)
    )
    monthly["missing_sales_dates"] = monthly["calendar_days"] - monthly["observed_days"]
    monthly["is_complete_observation_month"] = (
        monthly["missing_sales_dates"].eq(0)
        & monthly["year_month_period"].map(lambda period: period.end_time.normalize()).le(
            last_observed
        )
    )
    monthly["mom_growth"] = _strict_growth(monthly, 1)
    monthly["yoy_growth"] = _strict_growth(monthly, 12)
    monthly["year_month"] = monthly["year_month_period"].astype(str)
    return monthly.drop(columns="year_month_period").loc[
        :,
        [
            "year_month",
            "year",
            "month",
            "month_name",
            "total_sales",
            "observed_days",
            "calendar_days",
            "missing_sales_dates",
            "is_complete_observation_month",
            "mom_growth",
            "yoy_growth",
        ],
    ]


# %% [markdown]
# ## Weekday, month, weekend, and payday seasonality
#
# All averages below use observed daily totals only. Missing dates are excluded and
# are reported separately rather than being interpreted as zero-sales days.

# %%
def build_weekday_month_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Create month-by-weekday average daily sales for heatmap and tabular review."""
    observed = daily.loc[daily["has_sales_observation"].eq(1)]
    return (
        observed.groupby(
            ["month", "month_name", "day_of_week", "day_name"],
            as_index=False,
            observed=True,
        )
        .agg(
            average_sales=("total_sales", "mean"),
            observed_days=("date", "size"),
        )
        .sort_values(["month", "day_of_week"], kind="stable")
        .reset_index(drop=True)
    )


def build_seasonality_findings(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    monthly: pd.DataFrame,
) -> dict[str, object]:
    """Calculate evidence values for each required trend and seasonality finding."""
    observed = daily.loc[daily["has_sales_observation"].eq(1)].copy()
    historical_end = observed["date"].max()
    historical = daily.loc[daily["date"].le(historical_end)]
    missing_historical = historical.loc[historical["has_sales_observation"].eq(0)]
    christmas = daily.loc[(daily["date"].dt.month.eq(12)) & (daily["date"].dt.day.eq(25))]

    weekday = (
        observed.groupby(["day_of_week", "day_name"], as_index=False, observed=True)
        .agg(average_sales=("total_sales", "mean"), observed_days=("date", "size"))
        .sort_values("day_of_week")
    )
    month = (
        observed.groupby(["month", "month_name"], as_index=False, observed=True)
        .agg(average_sales=("total_sales", "mean"), observed_days=("date", "size"))
        .sort_values("month")
    )
    weekend = observed.assign(
        day_type=np.where(observed["is_weekend"].eq(1), "Weekend", "Weekday")
    ).groupby("day_type", observed=True)["total_sales"].mean()
    payday = observed.assign(
        payday_type=np.where(observed["is_payday"].eq(1), "Payday", "Non-payday")
    ).groupby("payday_type", observed=True)["total_sales"].mean()

    cutoff = pd.Timestamp("2017-08-15")
    ytd_2017 = observed.loc[
        observed["date"].between(pd.Timestamp("2017-01-01"), cutoff), "total_sales"
    ].sum()
    comparison_end = cutoff.replace(year=2016)
    ytd_2016 = observed.loc[
        observed["date"].between(pd.Timestamp("2016-01-01"), comparison_end),
        "total_sales",
    ].sum()
    ytd_growth = ytd_2017 / ytd_2016 - 1

    return {
        "observed_start": observed["date"].min(),
        "observed_end": historical_end,
        "observed_days": len(observed),
        "missing_historical_dates": missing_historical["date"].tolist(),
        "christmas_observation_flags": christmas.set_index("date")[
            "has_sales_observation"
        ].to_dict(),
        "latest_daily_sales": observed.iloc[-1]["total_sales"],
        "latest_ma_7": daily.loc[daily["date"].eq(historical_end), "sales_ma_7"].iloc[0],
        "latest_ma_28": daily.loc[daily["date"].eq(historical_end), "sales_ma_28"].iloc[0],
        "weekly_peak": weekly.loc[weekly["total_sales"].idxmax()].to_dict(),
        "latest_month": monthly.iloc[-1].to_dict(),
        "latest_valid_mom": monthly.dropna(subset=["mom_growth"]).iloc[-1].to_dict(),
        "latest_valid_yoy": monthly.dropna(subset=["yoy_growth"]).iloc[-1].to_dict(),
        "weekday": weekday,
        "month": month,
        "weekend_average": float(weekend["Weekend"]),
        "weekday_average": float(weekend["Weekday"]),
        "payday_average": float(payday["Payday"]),
        "nonpayday_average": float(payday["Non-payday"]),
        "ytd_2017": float(ytd_2017),
        "ytd_2016": float(ytd_2016),
        "ytd_growth": float(ytd_growth),
        "ytd_cutoff": cutoff,
    }


# %% [markdown]
# ## Visualizations

# %%
def _save_figure(fig: plt.Figure, filename: str) -> None:
    """Save one validated Matplotlib figure to the analysis directory."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_figures(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    weekday_month: pd.DataFrame,
    findings: dict[str, object],
) -> None:
    """Create all required trend and seasonality figures with labeled axes."""
    observed_end = findings["observed_end"]
    historical = daily.loc[daily["date"].le(observed_end)]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(historical["date"], historical["total_sales"], alpha=0.35, label="Daily total sales")
    ax.plot(historical["date"], historical["sales_ma_28"], label="28-day moving average")
    ax.set(title="Daily Total Sales with 28-Day Moving Average", xlabel="Date", ylabel="Sales volume")
    ax.legend()
    ax.grid(alpha=0.25)
    _save_figure(fig, "daily_sales_with_28d_ma.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pd.to_datetime(monthly["year_month"]), monthly["total_sales"], marker="o")
    ax.set(title="Monthly Observed Sales Trend (Partial Months Retained and Flagged)", xlabel="Month", ylabel="Observed sales volume")
    ax.grid(alpha=0.25)
    _save_figure(fig, "monthly_sales_trend.png")

    weekday = findings["weekday"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(weekday["day_name"], weekday["average_sales"])
    ax.set(title="Average Observed Daily Sales by Day of Week", xlabel="Day of week", ylabel="Average sales volume")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, "weekday_average_sales.png")

    pivot = weekday_month.pivot(index="month_name", columns="day_name", values="average_sales")
    month_order = weekday_month.sort_values("month")["month_name"].drop_duplicates().tolist()
    day_order = weekday_month.sort_values("day_of_week")["day_name"].drop_duplicates().tolist()
    pivot = pivot.reindex(index=month_order, columns=day_order)
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(pivot.to_numpy(), aspect="auto")
    ax.set_xticks(range(len(day_order)), labels=day_order, rotation=35, ha="right")
    ax.set_yticks(range(len(month_order)), labels=month_order)
    ax.set(title="Average Observed Daily Sales by Month and Weekday", xlabel="Day of week", ylabel="Month")
    fig.colorbar(image, ax=ax, label="Average sales volume")
    _save_figure(fig, "month_weekday_heatmap.png")

    observed = daily.loc[daily["has_sales_observation"].eq(1)].copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    for year in (2016, 2017):
        cutoff = pd.Timestamp(year=year, month=8, day=15)
        subset = observed.loc[
            observed["date"].between(pd.Timestamp(year=year, month=1, day=1), cutoff)
        ].copy()
        subset["day_of_year"] = subset["date"].dt.dayofyear
        subset["cumulative_sales"] = subset["total_sales"].cumsum()
        ax.plot(subset["day_of_year"], subset["cumulative_sales"], label=str(year))
    ax.set(title="YTD Sales Through August 15: 2017 versus 2016", xlabel="Day of year", ylabel="Cumulative observed sales volume")
    ax.legend(title="Year")
    ax.grid(alpha=0.25)
    _save_figure(fig, "ytd_sales_comparison.png")


# %% [markdown]
# ## Persist outputs and print evidence
#
# Every required analysis prints the values supporting its finding. These are
# descriptive summaries only; the notebook does not infer causes.

# %%
def print_findings(findings: dict[str, object]) -> None:
    """Print concise numerical evidence for every required analysis."""
    weekday = findings["weekday"]
    month = findings["month"]
    highest_weekday = weekday.loc[weekday["average_sales"].idxmax()]
    lowest_weekday = weekday.loc[weekday["average_sales"].idxmin()]
    highest_month = month.loc[month["average_sales"].idxmax()]
    lowest_month = month.loc[month["average_sales"].idxmin()]
    weekly_peak = findings["weekly_peak"]
    latest_month = findings["latest_month"]
    latest_mom = findings["latest_valid_mom"]
    latest_yoy = findings["latest_valid_yoy"]

    print(f"Daily trend: {findings['observed_days']:,} observed dates from {findings['observed_start']:%Y-%m-%d} to {findings['observed_end']:%Y-%m-%d}; latest total={findings['latest_daily_sales']:,.2f}.")
    print(f"Moving averages at {findings['observed_end']:%Y-%m-%d}: MA7={findings['latest_ma_7']:,.2f}; MA28={findings['latest_ma_28']:,.2f}. Missing dates are excluded, never filled with zero.")
    print(f"Weekly trend: peak observed weekly total={weekly_peak['total_sales']:,.2f} for week starting {weekly_peak['week_start']:%Y-%m-%d} ({int(weekly_peak['observed_days'])} observed days).")
    print(f"Monthly trend: latest period={latest_month['year_month']}, observed total={latest_month['total_sales']:,.2f}, complete_month={bool(latest_month['is_complete_observation_month'])}.")
    print(f"Latest valid MoM: {latest_mom['year_month']}={latest_mom['mom_growth']:.2%}; only two complete observed months are compared.")
    print(f"Latest valid YoY: {latest_yoy['year_month']}={latest_yoy['yoy_growth']:.2%}; only aligned complete observed months are compared.")
    print(f"Weekday average: highest={highest_weekday['day_name']} {highest_weekday['average_sales']:,.2f}; lowest={lowest_weekday['day_name']} {lowest_weekday['average_sales']:,.2f}.")
    print(f"Month-of-year average: highest={highest_month['month_name']} {highest_month['average_sales']:,.2f}; lowest={lowest_month['month_name']} {lowest_month['average_sales']:,.2f}.")
    print(f"Weekend versus weekday: {findings['weekend_average']:,.2f} versus {findings['weekday_average']:,.2f} average observed daily sales.")
    print(f"Payday versus non-payday: {findings['payday_average']:,.2f} versus {findings['nonpayday_average']:,.2f} average observed daily sales.")
    print(f"YTD through {findings['ytd_cutoff']:%Y-%m-%d}: 2017={findings['ytd_2017']:,.2f}; 2016 aligned={findings['ytd_2016']:,.2f}; growth={findings['ytd_growth']:.2%}.")
    missing = ", ".join(date.strftime("%Y-%m-%d") for date in findings["missing_historical_dates"])
    christmas = ", ".join(f"{date:%Y-%m-%d}={int(flag)}" for date, flag in findings["christmas_observation_flags"].items())
    print(f"Missing historical sales dates ({len(findings['missing_historical_dates'])}): {missing}.")
    print(f"Christmas has_sales_observation flags: {christmas}. A zero flag means no sales row was observed, not zero sales.")


def main() -> None:
    """Run analysis, validations, CSV persistence, figures, and evidence output."""
    sales, dates = load_processed_inputs()
    daily = build_daily_summary(sales, dates)
    weekly = build_weekly_summary(daily)
    monthly = build_monthly_summary(daily)
    weekday_month = build_weekday_month_summary(daily)
    findings = build_seasonality_findings(daily, weekly, monthly)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(DAILY_OUTPUT_PATH, index=False)
    monthly.to_csv(MONTHLY_OUTPUT_PATH, index=False)
    weekday_month.to_csv(WEEKDAY_MONTH_OUTPUT_PATH, index=False)
    create_figures(daily, monthly, weekday_month, findings)
    print_findings(findings)
    print(f"Daily table: {DAILY_OUTPUT_PATH}")
    print(f"Monthly table: {MONTHLY_OUTPUT_PATH}")
    print(f"Weekday-month table: {WEEKDAY_MONTH_OUTPUT_PATH}")
    print(f"Figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
