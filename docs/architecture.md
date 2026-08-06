# Project Architecture

## End-to-end flow

```text
data/raw/*.csv
    │
    ▼
Raw loading + audit
    │
    ▼
Cleaning and business-key validation
    │
    ▼
data/interim/*_clean.parquet
    │
    ▼
Dimension, fact, bridge, and store-date construction
    │
    ▼
data/processed/*.parquet
    │
    ├──► PostgreSQL DDL/load/marts/quality checks (optional; runtime unvalidated)
    │
    ▼
Business EDA and readiness notebook scripts
    │
    ▼
reports/tables + reports/figures + reports/*.md
    │
    ▼
Power BI semantic model (designed, not implemented)
    │
    ▼
Future DS forecasting pipeline
```

The pipeline does not treat a missing store-day observation as zero. The complete
store-date grid and its observation flags preserve that distinction for analysis
and Power BI.

## Stage ownership

| Stage | Responsible code | Inputs | Principal outputs |
|---|---|---|---|
| Raw loading | `src/data/load_raw.py` | `data/raw/*.csv` | Typed pandas DataFrames; clear missing-file errors |
| Raw audit | `notebooks/01_data_audit.py`, `src/data/audit.py` | Raw DataFrames | `reports/data_quality/column_audit.csv`, grain/FK issue files, `audit_summary.md`, cleaning figures |
| Cleaning | `src/data/run_cleaning.py`; cleaners in `src/data/clean_*.py`; notebook 02 as wrapper | Raw DataFrames | Six validated interim Parquet files and `cleaning_summary.md` |
| Date dimension | `src/data/build_date_dimension.py`, `notebooks/03_date_dimension.py` | Analysis date bounds | Validated calendar dimension contract |
| Warehouse build | `src/data/run_warehouse_build.py`; `build_dimensions.py`, `build_store_date_dimension.py`, `build_facts.py`, `build_bridges.py` | Interim Parquet | Eight processed Parquet tables and `warehouse_reconciliation.md` |
| Store/family EDA | `notebooks/04_business_eda.py` | Processed facts/dimensions | Store/family performance CSVs, findings, ranking/segment figures |
| Trend/seasonality | `notebooks/04a_sales_trend_seasonality.py` | Processed sales/date tables | Daily/monthly/weekday summaries and seasonal figures |
| Promotion analysis | `notebooks/05_promotion_analysis.py` | Processed daily sales | Promotion association tables, limitations, figures |
| Holiday analysis | `notebooks/06_holiday_analysis.py` | Sales plus store-date/holiday context | Holiday association tables, notes, figures |
| Transaction analysis | `notebooks/07_transactions_analysis.py` | Store-day sales and transactions | Transaction summaries, notes, figures |
| Anomaly review | `notebooks/08_anomaly_review.py` | Processed facts/context | `sales_anomalies.csv`, notes, review flags |
| Forecast readiness | `notebooks/09_forecast_readiness.py` | Sales plus EDA/anomaly artifacts | `forecast_readiness.csv` and `forecast_readiness.md` |
| Consolidated interpretation | Evidence from generated reports | Report tables and Markdown | `reports/business_insights.md`, `reports/da_project_validation.md` |
| PostgreSQL load | `src/load_to_postgres.py`, `sql/ddl/*.sql` | Processed Parquet | `analytics` schema tables, only when a database is configured |
| SQL marts/quality | `sql/marts/*.sql`, `sql/data_quality/*.sql`, `src/run_sql_quality_checks.py` | Loaded PostgreSQL tables | SQL marts and quality report; dry-run parsing supported |
| Power BI | Contract in `docs/powerbi_model.md` | Processed model tables or validated database tables | Future semantic model/report; no current `.pbix`/`.pbit` |
| Future forecasting | Not yet implemented | Store-family history and future-known features | Future baselines, backtests, forecasts, intervals, model artifacts |

## Cleaning layer

The cleaning entry point reads all raw tables, normalizes dates and types, retains
observed zero sales, rejects invalid negative measures, aggregates applicable
holiday records to store-day grain, and checks source grains/FKs before writing.
Interim files are implementation artifacts and are excluded from Git.

## Warehouse-build layer

The Python warehouse build is the authoritative validated local pipeline. It:

1. Creates stable store and family surrogate keys.
2. Creates a continuous date dimension over train and test bounds.
3. Builds the complete date × store dimension and observation flags.
4. Builds sales at date–store–family grain.
5. Builds transactions at date–store grain without family multiplication.
6. Builds a daily oil fact and a geography-aware store-holiday bridge.
7. Rejects duplicate grains, missing keys, unmapped FKs, or reconciliation drift
   before writing processed Parquet.

The current reconciliation covers 3,000,888 sales rows, 83,488 transaction rows,
1,704 dates, 54 stores, 33 families, 92,016 store-dates, and 7,938 holiday-bridge
rows. See [Warehouse reconciliation](../reports/data_quality/warehouse_reconciliation.md).

## PostgreSQL branch

DDL mirrors the processed model, with explicit PK/FK/unique/check constraints.
Load and quality runners are present, and the SQL quality suite supports a
connection-free parse mode. However, the current validation environment had no
PostgreSQL connection variables and no `.env`; therefore runtime DDL execution,
data load, mart execution, and SQL reconciliation remain unverified. A successful
Parquet build or SQL dry run must not be presented as a successful database run.

## Analysis and reporting layer

Notebook files are executable Python scripts, allowing deterministic artifact
regeneration. Report CSVs are evidence tables; Markdown files interpret those
tables and state caveats. Large reproducible CSVs and raw data are ignored by Git,
while durable Markdown documentation remains trackable.

Promotion and holiday comparisons are associative. Nothing in this layer provides
randomization or controls sufficient for causal inference. Sales metrics represent
volume rather than revenue, and the absence of cost/profit/inventory prevents
financial or replenishment conclusions.

## Power BI boundary

The intended semantic model consumes conformed dimensions and facts, with
`dim_store_date` owning store-day/holiday filtering. This repository currently
documents the model only; it contains no finished report or fabricated screenshot.
Relationship details are in [Power BI model](powerbi_model.md).

## Future forecasting boundary

Readiness flags route series to modeling strategies but do not constitute model
evaluation. The future DS stage must add temporal cross-validation, baselines,
future-feature availability tests, interval calibration, experiment tracking, and
versioned artifacts without leaking test/future information into training.

