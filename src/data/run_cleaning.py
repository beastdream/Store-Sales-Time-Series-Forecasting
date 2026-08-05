"""Run the validated raw-to-interim cleaning pipeline."""

from pathlib import Path

import pandas as pd

from src.config import DATA_INTERIM, REPORTS_DIR, ensure_project_directories
from src.data.audit import check_foreign_key, summarize_grain
from src.data.clean_core import clean_stores, clean_test, clean_train
from src.data.clean_holidays import clean_holidays
from src.data.clean_oil import clean_oil
from src.data.clean_transactions import clean_transactions
from src.data.load_raw import load_all_raw_tables


OUTPUT_PATHS = {
    "train": DATA_INTERIM / "train_clean.parquet",
    "test": DATA_INTERIM / "test_clean.parquet",
    "stores": DATA_INTERIM / "stores_clean.parquet",
    "transactions": DATA_INTERIM / "transactions_clean.parquet",
    "oil": DATA_INTERIM / "oil_clean.parquet",
    "holidays": DATA_INTERIM / "holiday_store_daily.parquet",
}

GRAIN_SPECS = {
    "train": ["date", "store_nbr", "family"],
    "test": ["date", "store_nbr", "family"],
    "stores": ["store_nbr"],
    "transactions": ["date", "store_nbr"],
    "oil": ["date"],
    "holidays": ["date", "store_nbr"],
}


def _format_markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """Build a small Markdown table without optional formatting dependencies."""
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _build_summary(
    raw_tables: dict[str, pd.DataFrame],
    clean_tables: dict[str, pd.DataFrame],
    grain_summary: pd.DataFrame,
    foreign_key_issues: pd.DataFrame,
) -> str:
    """Build the cleaning run summary after all critical validations pass."""
    table_names = ["train", "test", "stores", "transactions", "oil", "holidays"]
    row_rows = [
        [
            name,
            len(raw_tables[name]),
            len(clean_tables[name]),
            int(raw_tables[name].duplicated().sum()),
        ]
        for name in table_names
    ]
    grain_rows = [
        [
            row.table_name,
            ", ".join(row.grain_columns),
            row.duplicate_combinations,
            row.affected_rows,
            row.is_valid,
        ]
        for row in grain_summary.itertuples(index=False)
    ]

    oil_imputed = int(clean_tables["oil"]["oil_was_imputed"].sum())
    transferred_holidays = int(
        (
            raw_tables["holidays"]["type"].eq("Holiday")
            & raw_tables["holidays"]["transferred"]
        ).sum()
    )
    zero_sales = int(clean_tables["train"]["sales"].eq(0).sum())
    warnings = [
        f"- `{zero_sales}` train rows have zero sales; they were intentionally retained.",
        f"- `{oil_imputed}` daily oil prices were imputed and are flagged by `oil_was_imputed`.",
        "- Leading rows in oil change features remain missing where lag history is unavailable.",
    ]

    return "\n".join(
        [
            "# Cleaning Summary",
            "",
            "All critical grain and foreign-key validations passed before outputs were saved.",
            "",
            "## Row counts and exact duplicates removed",
            "",
            _format_markdown_table(
                ["Table", "Raw rows", "Clean rows", "Exact duplicates removed"],
                row_rows,
            ),
            "",
            "## Oil-price interpolation",
            "",
            f"- Calendar range: `{clean_tables['oil']['date'].min():%Y-%m-%d}` to "
            f"`{clean_tables['oil']['date'].max():%Y-%m-%d}`.",
            f"- Imputed daily prices: `{oil_imputed}`.",
            f"- Remaining missing `oil_price`: "
            f"`{int(clean_tables['oil']['oil_price'].isna().sum())}`.",
            "",
            "## Holiday store mapping",
            "",
            f"- Raw holiday rows: `{len(raw_tables['holidays'])}`.",
            f"- Transferred Holiday rows excluded: `{transferred_holidays}`.",
            f"- Daily store holiday rows created: `{len(clean_tables['holidays'])}`.",
            "",
            "## Grain checks",
            "",
            _format_markdown_table(
                ["Table", "Grain", "Duplicate combinations", "Affected rows", "Valid"],
                grain_rows,
            ),
            "",
            "## Foreign-key checks",
            "",
            f"- Invalid key values: `{len(foreign_key_issues)}`.",
            f"- Affected child rows: "
            f"`{int(foreign_key_issues['affected_rows'].sum()) if not foreign_key_issues.empty else 0}`.",
            "",
            "## Remaining warnings",
            "",
            *warnings,
            "",
        ]
    )


def run_cleaning() -> None:
    """Load, clean, validate, and persist all core interim data tables."""
    ensure_project_directories()
    raw_tables = load_all_raw_tables()

    train_clean = clean_train(raw_tables["train"])
    test_clean = clean_test(raw_tables["test"])
    stores_clean = clean_stores(raw_tables["stores"])
    transactions_clean = clean_transactions(raw_tables["transactions"])

    start_date = train_clean["date"].min()
    end_date = test_clean["date"].max()
    if pd.isna(start_date) or pd.isna(end_date):
        raise RuntimeError("Cleaning stopped: train/test date range is unavailable")

    oil_clean = clean_oil(raw_tables["oil"], start_date, end_date)
    holiday_store_daily = clean_holidays(raw_tables["holidays"], stores_clean)
    clean_tables = {
        "train": train_clean,
        "test": test_clean,
        "stores": stores_clean,
        "transactions": transactions_clean,
        "oil": oil_clean,
        "holidays": holiday_store_daily,
    }

    grain_summary = pd.concat(
        [
            summarize_grain(clean_tables[name], grain, name)
            for name, grain in GRAIN_SPECS.items()
        ],
        ignore_index=True,
    )
    invalid_grains = grain_summary.loc[~grain_summary["is_valid"]]
    if not invalid_grains.empty:
        names = ", ".join(invalid_grains["table_name"].astype(str))
        raise RuntimeError(f"Cleaning stopped: invalid grain in {names}")

    foreign_key_issues = pd.concat(
        [
            check_foreign_key(
                clean_tables[child],
                "store_nbr",
                stores_clean,
                "store_nbr",
                child,
                "stores",
            )
            for child in ("train", "test", "transactions")
        ],
        ignore_index=True,
    )
    if not foreign_key_issues.empty:
        raise RuntimeError(
            "Cleaning stopped: invalid store foreign keys affect "
            f"{int(foreign_key_issues['affected_rows'].sum())} rows"
        )
    if oil_clean["oil_price"].isna().any():
        raise RuntimeError("Cleaning stopped: oil_price still contains missing values")

    # Persistence starts only after every critical validation above has passed.
    for name, output_path in OUTPUT_PATHS.items():
        clean_tables[name].to_parquet(output_path, index=False)

    summary = _build_summary(
        raw_tables,
        clean_tables,
        grain_summary,
        foreign_key_issues,
    )
    summary_path: Path = REPORTS_DIR / "data_quality" / "cleaning_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary, encoding="utf-8")

    print("Cleaning pipeline completed successfully.")
    for path in (*OUTPUT_PATHS.values(), summary_path):
        print(path)


if __name__ == "__main__":
    run_cleaning()
