# %% [markdown]
# # Date Dimension Validation
#
# This notebook reads the date dimension created by the cleaning pipeline and
# validates its coverage. Calendar construction remains in `src/`.

# %%
from calendar import isleap
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.run_cleaning import DIM_DATE_PATH, run_cleaning

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)


def show(title: str, value: object) -> None:
    """Print a compact notebook section when run as a script."""
    print(f"\n## {title}")
    print(value)


# %% [markdown]
# ## Load the processed date dimension

# %%
if not DIM_DATE_PATH.is_file():
    run_cleaning()

dim_date = pd.read_parquet(DIM_DATE_PATH)
start_date = dim_date["full_date"].min()
end_date = dim_date["full_date"].max()

# %% [markdown]
# ## Coverage and cardinality

# %%
year_month = dim_date["full_date"].dt.to_period("M")
year_quarter = dim_date["full_date"].dt.to_period("Q")
overview = pd.Series(
    {
        "start_date": start_date,
        "end_date": end_date,
        "row_count": len(dim_date),
        "year_count": dim_date["year"].nunique(),
        "month_count": year_month.nunique(),
        "quarter_count": year_quarter.nunique(),
        "weekend_count": int(dim_date["is_weekend"].sum()),
        "payday_count": int(dim_date["is_payday"].sum()),
    },
    name="value",
)
show("Date range and counts", overview.to_string())

# %% [markdown]
# ## Month-end examples

# %%
month_end_examples = dim_date.loc[
    dim_date["is_month_end"].eq(1),
    ["date_key", "full_date", "day_name", "month_name", "year", "is_payday"],
].head(12)
show("Month-end examples", month_end_examples.to_string(index=False))

# %% [markdown]
# ## Leap-year validation

# %%
years = range(int(dim_date["year"].min()), int(dim_date["year"].max()) + 1)
expected_leap_years = [year for year in years if isleap(year)]
leap_days = dim_date.loc[
    dim_date["full_date"].dt.month.eq(2)
    & dim_date["full_date"].dt.day.eq(29),
    ["date_key", "full_date", "day_name"],
]
covered_leap_years = leap_days["full_date"].dt.year.tolist()
assert covered_leap_years == expected_leap_years
show("Leap days", leap_days.to_string(index=False))

# %% [markdown]
# ## Continuous-calendar validation

# %%
expected_calendar = pd.date_range(start_date, end_date, freq="D")
missing_dates = expected_calendar.difference(dim_date["full_date"])
unexpected_dates = pd.DatetimeIndex(dim_date["full_date"]).difference(expected_calendar)

assert len(dim_date) == len(expected_calendar)
assert dim_date["date_key"].is_unique
assert not dim_date.isna().any().any()
assert missing_dates.empty
assert unexpected_dates.empty

validation = pd.Series(
    {
        "expected_days": len(expected_calendar),
        "actual_rows": len(dim_date),
        "missing_dates": len(missing_dates),
        "unexpected_dates": len(unexpected_dates),
        "duplicate_date_keys": int(dim_date["date_key"].duplicated().sum()),
        "missing_cells": int(dim_date.isna().sum().sum()),
        "is_valid": True,
    },
    name="value",
)
show("Calendar validation", validation.to_string())
show("Parquet output", DIM_DATE_PATH.relative_to(PROJECT_ROOT))
