# %% [markdown]
# # Store Sales - Cleaning Review
#
# This notebook reviews outputs produced by the reusable cleaning pipeline. It
# does not repeat or redefine cleaning logic.

# %%
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR
from src.data.audit import audit_all_tables, summarize_grain
from src.data.load_raw import load_all_raw_tables
from src.data.run_cleaning import GRAIN_SPECS, OUTPUT_PATHS, run_cleaning

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", 80)

FIGURE_DIR = FIGURES_DIR / "data_cleaning"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def show(title: str, value: object) -> None:
    """Print a compact section when the notebook is run as a script."""
    print(f"\n## {title}")
    print(value)


def save_figure(fig: plt.Figure, filename: str, caption: str) -> Path:
    """Add a short caption and save one matplotlib figure."""
    fig.text(0.5, 0.02, caption, ha="center", fontsize=9)
    fig.subplots_adjust(bottom=0.20)
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# %% [markdown]
# ## Load raw and cleaned tables

# %%
if not all(path.is_file() for path in OUTPUT_PATHS.values()):
    run_cleaning()

raw_tables = load_all_raw_tables()
clean_tables = {
    name: pd.read_parquet(path)
    for name, path in OUTPUT_PATHS.items()
}

# %% [markdown]
# ## Shapes before and after cleaning

# %%
shape_summary = pd.DataFrame(
    [
        {
            "table_name": name,
            "raw_rows": len(raw_tables[name]),
            "raw_columns": raw_tables[name].shape[1],
            "clean_rows": len(clean_tables[name]),
            "clean_columns": clean_tables[name].shape[1],
        }
        for name in GRAIN_SPECS
    ]
)
show("Shapes before and after cleaning", shape_summary.to_string(index=False))

# %% [markdown]
# ## Missing values after cleaning

# %%
clean_column_audit = audit_all_tables(clean_tables)
missing_after = clean_column_audit.loc[
    clean_column_audit["missing_count"].gt(0),
    ["table_name", "column_name", "missing_count", "missing_pct"],
].reset_index(drop=True)
show("Missing values after cleaning", missing_after.to_string(index=False))

# %% [markdown]
# ## Grain after cleaning

# %%
grain_summary = pd.concat(
    [
        summarize_grain(clean_tables[name], grain_columns, name)
        for name, grain_columns in GRAIN_SPECS.items()
    ],
    ignore_index=True,
)
show("Grain checks after cleaning", grain_summary.to_string(index=False))

# %% [markdown]
# ## Cleaning examples

# %%
oil_examples = clean_tables["oil"].loc[
    clean_tables["oil"]["oil_was_imputed"].eq(1),
    ["date", "oil_price", "oil_was_imputed"],
].head(5)
show("Example imputed oil prices", oil_examples.to_string(index=False))

holidays = clean_tables["holidays"]
holiday_example_columns = [
    "date",
    "store_nbr",
    "holiday_count",
    "holiday_descriptions",
    "holiday_types",
    "holiday_locales",
]
for locale in ("National", "Regional", "Local"):
    example = holidays.loc[
        holidays["holiday_locales"].str.contains(locale, na=False),
        holiday_example_columns,
    ].head(3)
    show(f"Example {locale} holiday mapping", example.to_string(index=False))

multiple_events = holidays.loc[
    holidays["holiday_count"].gt(1),
    holiday_example_columns,
].head(5)
show("Examples with multiple same-day events", multiple_events.to_string(index=False))

# %% [markdown]
# ## Sales distribution

# %%
train = clean_tables["train"]
sales_upper = float(train["sales"].quantile(0.99))
sales_for_plot = train["sales"].clip(upper=sales_upper)

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(sales_for_plot, bins=60)
ax.set_title("Distribution of Cleaned Sales (Capped at 99th Percentile)")
ax.set_xlabel("Sales volume")
ax.set_ylabel("Number of rows")
sales_figure = save_figure(
    fig,
    "sales_distribution.png",
    "Values above the 99th percentile are capped for display only; cleaned data is unchanged.",
)

# %% [markdown]
# ## Share of zero sales

# %%
zero_sales_count = int(train["sales"].eq(0).sum())
positive_sales_count = int(train["sales"].gt(0).sum())
zero_sales_rate = zero_sales_count / len(train)
show("Sales equal to zero", f"{zero_sales_count:,} rows ({zero_sales_rate:.2%})")

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(["Sales = 0", "Sales > 0"], [zero_sales_count, positive_sales_count])
ax.set_title("Zero versus Positive Sales Rows")
ax.set_xlabel("Sales status")
ax.set_ylabel("Number of rows")
zero_sales_figure = save_figure(
    fig,
    "zero_sales_counts.png",
    "Zero-sales observations are retained by the cleaning pipeline.",
)

# %% [markdown]
# ## On-promotion distribution

# %%
promotion_upper = int(train["onpromotion"].quantile(0.99))
promotion_for_plot = train["onpromotion"].clip(upper=promotion_upper)

fig, ax = plt.subplots(figsize=(9, 5))
bins = range(0, promotion_upper + 2)
ax.hist(promotion_for_plot, bins=bins, align="left")
ax.set_title("Distribution of Items on Promotion (Capped at 99th Percentile)")
ax.set_xlabel("Items on promotion")
ax.set_ylabel("Number of rows")
promotion_figure = save_figure(
    fig,
    "onpromotion_distribution.png",
    "The upper tail is capped for display only; source promotion counts are unchanged.",
)

# %% [markdown]
# ## Share of store-days with promotion

# %%
daily_store_promotion = train.groupby(
    ["date", "store_nbr"], observed=True
)["is_promotion"].max()
promotion_day_rate = float(daily_store_promotion.mean())
show("Store-days with promotion", f"{promotion_day_rate:.2%}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(
    ["No promotion", "Has promotion"],
    [1 - promotion_day_rate, promotion_day_rate],
)
ax.set_title("Share of Store-Days with at Least One Promotion")
ax.set_xlabel("Promotion status")
ax.set_ylabel("Share of store-days")
ax.set_ylim(0, 1)
promotion_rate_figure = save_figure(
    fig,
    "promotion_day_rate.png",
    "A promoted store-day has at least one family with onpromotion greater than zero.",
)

# %% [markdown]
# ## Saved figures

# %%
figure_paths = [
    sales_figure,
    zero_sales_figure,
    promotion_figure,
    promotion_rate_figure,
]
show(
    "Saved figures",
    "\n".join(str(path.relative_to(PROJECT_ROOT)) for path in figure_paths),
)
