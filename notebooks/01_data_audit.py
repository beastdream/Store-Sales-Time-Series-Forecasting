# %% [markdown]
# # Store Sales - Raw Data Audit
#
# This notebook profiles raw inputs and records data-quality findings. It does
# not clean, impute, deduplicate, or otherwise modify source data.

# %%
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import REPORTS_DIR
from src.data.audit import (
    FOREIGN_KEY_COLUMNS,
    GRAIN_DEFINITIONS,
    check_foreign_key,
    check_grain,
    audit_all_tables,
    summarize_grain,
)
from src.data.load_raw import load_all_raw_tables

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)
pd.set_option("display.max_rows", 100)

DATA_QUALITY_DIR = REPORTS_DIR / "data_quality"
DATA_QUALITY_DIR.mkdir(parents=True, exist_ok=True)


def show(title: str, value: object) -> None:
    """Print a compact notebook section when run as a Python script."""
    print(f"\n## {title}")
    print(value)


# %% [markdown]
# ## Load all raw tables

# %%
tables = load_all_raw_tables()

shape_summary = pd.DataFrame(
    [
        {"table_name": name, "rows": frame.shape[0], "columns": frame.shape[1]}
        for name, frame in tables.items()
    ]
)
show("Table shapes", shape_summary.to_string(index=False))

# %% [markdown]
# ## Train and test date coverage

# %%
date_ranges = pd.DataFrame(
    [
        {
            "table_name": name,
            "min_date": tables[name]["date"].min(),
            "max_date": tables[name]["date"].max(),
        }
        for name in ("train", "test")
    ]
)
show("Train and test date ranges", date_ranges.to_string(index=False))

# %% [markdown]
# ## Column audit

# %%
column_audit = audit_all_tables(tables)
column_audit_path = DATA_QUALITY_DIR / "column_audit.csv"
column_audit.to_csv(column_audit_path, index=False)
show("Columns with missing values", column_audit.query("missing_count > 0").to_string(index=False))

# %% [markdown]
# ## Fully duplicated rows

# %%
duplicate_summary = pd.DataFrame(
    [
        {
            "table_name": name,
            "duplicate_rows": int(frame.duplicated(keep=False).sum()),
        }
        for name, frame in tables.items()
    ]
)
show("Fully duplicated rows", duplicate_summary.to_string(index=False))

# %% [markdown]
# ## Grain checks

# %%
grain_issue_frames = []
grain_summary_frames = []
for table_name, grain_columns in GRAIN_DEFINITIONS.items():
    grain_issue_frames.append(
        check_grain(tables[table_name], grain_columns, table_name)
    )
    grain_summary_frames.append(
        summarize_grain(tables[table_name], grain_columns, table_name)
    )

grain_issues = pd.concat(grain_issue_frames, ignore_index=True, sort=False)
grain_summary = pd.concat(grain_summary_frames, ignore_index=True)
grain_issues_path = DATA_QUALITY_DIR / "grain_issues.csv"
grain_issues.to_csv(grain_issues_path, index=False)
show("Grain summary", grain_summary.to_string(index=False))
show("Grain issue sample", grain_issues.to_string(index=False))

# %% [markdown]
# ## Foreign-key checks

# %%
foreign_key_relations = [
    ("train", "store_nbr", "stores", "store_nbr"),
    ("test", "store_nbr", "stores", "store_nbr"),
    ("transactions", "store_nbr", "stores", "store_nbr"),
]
foreign_key_frames = [
    check_foreign_key(
        tables[child_table],
        child_column,
        tables[parent_table],
        parent_column,
        child_table,
        parent_table,
    )
    for child_table, child_column, parent_table, parent_column in foreign_key_relations
]
foreign_key_issues = (
    pd.concat(foreign_key_frames, ignore_index=True)
    if foreign_key_frames
    else pd.DataFrame(columns=FOREIGN_KEY_COLUMNS)
)
foreign_key_issues_path = DATA_QUALITY_DIR / "foreign_key_issues.csv"
foreign_key_issues.to_csv(foreign_key_issues_path, index=False)
show("Foreign-key issues", foreign_key_issues.to_string(index=False))

# %% [markdown]
# ## Domain and missing-value checks

# %%
anomaly_counts = {
    "train_sales_negative": int(tables["train"]["sales"].lt(0).sum()),
    "train_sales_zero": int(tables["train"]["sales"].eq(0).sum()),
    "train_onpromotion_negative": int(
        tables["train"]["onpromotion"].lt(0).sum()
    ),
    "test_onpromotion_negative": int(tables["test"]["onpromotion"].lt(0).sum()),
    "missing_oil_price": int(tables["oil"]["dcoilwtico"].isna().sum()),
    "stores_with_missing_metadata": int(
        tables["stores"][["city", "state", "type", "cluster"]]
        .isna()
        .any(axis=1)
        .sum()
    ),
}
anomaly_summary = pd.DataFrame(
    [{"check": name, "affected_rows": count} for name, count in anomaly_counts.items()]
)
show("Domain and metadata checks", anomaly_summary.to_string(index=False))

important_columns = {
    "train": ["sales", "onpromotion"],
    "test": ["onpromotion"],
    "stores": ["city", "state", "type", "cluster"],
    "transactions": ["transactions"],
    "oil": ["dcoilwtico"],
}
important_missing = column_audit.loc[
    column_audit.apply(
        lambda row: row["column_name"]
        in important_columns.get(str(row["table_name"]), []),
        axis=1,
    ),
    ["table_name", "column_name", "missing_count", "missing_pct"],
].reset_index(drop=True)
show("Important missing-value rates", important_missing.to_string(index=False))

# %% [markdown]
# ## Markdown audit summary

# %%
grain_lines = [
    (
        f"- `{row.table_name}`: grain `{', '.join(row.grain_columns)}`; "
        f"duplicate combinations = {row.duplicate_combinations}; "
        f"affected rows = {row.affected_rows}; valid = {row.is_valid}."
    )
    for row in grain_summary.itertuples(index=False)
]
date_lines = [
    f"- `{row.table_name}`: {row.min_date:%Y-%m-%d} to {row.max_date:%Y-%m-%d}."
    for row in date_ranges.itertuples(index=False)
]
missing_lines = [
    (
        f"- `{row.table_name}.{row.column_name}`: {row.missing_count} missing "
        f"({row.missing_pct:.2f}%)."
    )
    for row in important_missing.itertuples(index=False)
]
duplicate_lines = [
    f"- `{row.table_name}`: {row.duplicate_rows} fully duplicated rows."
    for row in duplicate_summary.itertuples(index=False)
]
anomaly_lines = [
    f"- `{row.check}`: {row.affected_rows} affected rows."
    for row in anomaly_summary.itertuples(index=False)
]

cleaning_issues = []
if anomaly_counts["missing_oil_price"]:
    cleaning_issues.append(
        "- Decide and document an imputation policy for missing oil prices."
    )
if anomaly_counts["train_sales_zero"]:
    cleaning_issues.append(
        "- Validate zero sales against closures, holidays, and genuine no-sale days before modeling."
    )
if anomaly_counts["train_sales_negative"]:
    cleaning_issues.append("- Investigate negative sales before modeling.")
if anomaly_counts["train_onpromotion_negative"] or anomaly_counts[
    "test_onpromotion_negative"
]:
    cleaning_issues.append("- Investigate negative on-promotion counts.")
if anomaly_counts["stores_with_missing_metadata"]:
    cleaning_issues.append("- Resolve missing store metadata before dimensional joins.")
if int(duplicate_summary["duplicate_rows"].sum()):
    cleaning_issues.append("- Review fully duplicated rows before any deduplication decision.")
if not foreign_key_issues.empty:
    cleaning_issues.append("- Resolve invalid store foreign keys before joining store metadata.")
if not cleaning_issues:
    cleaning_issues.append("- No cleaning issue was identified by the checks in this audit.")

audit_summary = "\n".join(
    [
        "# Raw Data Audit Summary",
        "",
        "> This audit reports findings only; no raw data was cleaned or modified.",
        "",
        "## Grain",
        "",
        *grain_lines,
        "",
        "## Time coverage",
        "",
        *date_lines,
        "",
        "## Important missing values",
        "",
        *missing_lines,
        "",
        "## Fully duplicated rows",
        "",
        *duplicate_lines,
        "",
        "## Unusual values and metadata checks",
        "",
        *anomaly_lines,
        "",
        "## Issues to address during cleaning",
        "",
        *cleaning_issues,
        "",
    ]
)
audit_summary_path = DATA_QUALITY_DIR / "audit_summary.md"
audit_summary_path.write_text(audit_summary, encoding="utf-8")

show(
    "Saved reports",
    "\n".join(
        str(path.relative_to(PROJECT_ROOT))
        for path in (
            column_audit_path,
            grain_issues_path,
            foreign_key_issues_path,
            audit_summary_path,
        )
    ),
)
