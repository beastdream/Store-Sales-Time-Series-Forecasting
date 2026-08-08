# Project Architecture

## Current implemented flow

```text
Raw Data
    ↓
Cleaning / Validation
    ↓
Interim Parquet
    ↓
Analytical Warehouse / Dimensional Model
    ↓
Processed Parquet
    ↓
EDA / Analytical Reports
    ↓
Power BI Dashboard
    ↓
Forecast Readiness Assessment
```

All components above are implemented. The local Power BI report is
[`powerbi/store_sales_analytics.pbix`](../powerbi/store_sales_analytics.pbix) and
contains eight analytical pages. PostgreSQL is a separate optional deployment
branch; its runtime has not been validated.

## Planned Data Science flow

```text
[Future DS Phase]
Feature Engineering
    ↓
Temporal Backtesting
    ↓
Baseline and Forecast Models
    ↓
Evaluation and Error Analysis
    ↓
Final 16-Day Forecast
```

Nothing in this planned block is represented as completed. `data/features/` and
`models/` currently contain placeholders only. The plan is documented in the
[Data Science Roadmap](data_science_roadmap.md).

## Stage ownership

| Stage | Responsible code/artifact | Inputs | Principal outputs | Status |
|---|---|---|---|---|
| Raw loading | `src/data/load_raw.py` | `data/raw/*.csv` | Typed DataFrames | Implemented |
| Raw audit | `notebooks/01_data_audit.py`, `src/data/audit.py` | Raw tables | Audit CSV/Markdown and cleaning figures | Implemented |
| Cleaning | `src/data/run_cleaning.py`, `src/data/clean_*.py` | Raw tables | Six interim Parquet files, cleaning summary | Implemented |
| Warehouse build | `src/data/run_warehouse_build.py`, dimension/fact/bridge builders | Interim Parquet | Eight processed Parquet tables, reconciliation report | Implemented |
| Business EDA | Notebooks 04/04a–08 | Processed warehouse | Store/family/trend/promotion/holiday/transaction/anomaly tables and figures | Implemented |
| Forecast readiness | `notebooks/09_forecast_readiness.py` | Historical sales plus DA outputs | Readiness CSV and Markdown | Implemented |
| Consolidated reporting | `reports/` | Reproducible analytical artifacts | Business insights and validation evidence | Implemented |
| Power BI | `powerbi/store_sales_analytics.pbix` | Dimensional/report tables | Eight-page interactive local dashboard | Complete |
| PostgreSQL deployment | `src/load_to_postgres.py`, `sql/` | Processed Parquet | Optional database tables/marts/checks | Code present; runtime NOT RUN |
| Feature engineering | Future DS work | Historical data and forecast-origin-safe covariates | Planned feature Parquet | Not started |
| Backtesting/modeling | Future DS work | Planned features | Planned scores/models/forecasts | Not started |

## Analytical dimensional model

### `DimDate`

One row per calendar date. It supplies year, quarter, month, weekday, weekend,
month-boundary, and payday attributes. The dimension extends to 2017-08-31 for the
test horizon; actual historical Sales Volume ends on 2017-08-15.

### `DimStore`

One row per store, with the business store number, city, state, type, and cluster.

### `DimFamily`

One row per product family. Family describes a category rather than an SKU.

### `DimStoreDate`

One row per date × store across the complete calendar-store grid. It is the
store-date bridge/context dimension and carries:

- holiday/event/work-day flags and aggregated descriptions;
- `has_sales_observation` and `has_transaction_observation`;
- the unique `date_store_key` used by store-day facts.

This table allows date and store filters to reach sales and transactions through
one conformed path, supports local/regional holiday applicability, retains regular
days, and preserves missing observation separately from observed zero.

### `FactDailySales`

One observed row per date × store × family. Measures are Sales Volume,
`onpromotion`, and the binary promotion-active flag. `sales` is not revenue.

### `FactStoreTransactions`

One observed row per date × store. It has no family grain and must not be expanded
across 33 families. Family filters therefore must not be assumed to filter this
fact directly.

### `FactOilPrice`

One row per calendar date, with cleaned oil price, lagged changes, and an imputation
flag. Its future availability requires a separate leakage/production-realism
decision in the DS phase.

The holiday-only `BridgeStoreHoliday` remains an audit/detail artifact, not the
primary filter path, because regular days are absent from it.

## Preferred Power BI relationships

```text
DimDate ───────┐
               ▼
          DimStoreDate ─────► FactDailySales ◄───── DimFamily
               │
DimStore ──────┘
               └────────────► FactStoreTransactions

DimDate ─────────────────────► FactOilPrice
```

Active relationships use single-direction filtering from the one side to the many
side:

| One side | Many side | Key |
|---|---|---|
| `DimDate` | `DimStoreDate` | `date_key` |
| `DimStore` | `DimStoreDate` | `store_key` |
| `DimStoreDate` | `FactDailySales` | `date_store_key` |
| `DimStoreDate` | `FactStoreTransactions` | `date_store_key` |
| `DimFamily` | `FactDailySales` | `family_key` |
| `DimDate` | `FactOilPrice` | `date_key` |

Do not add duplicate active relationships from `DimDate` or `DimStore` directly to
the two store-day facts. That would introduce multiple filter paths. Do not use
many-to-many or bidirectional filtering as a workaround.

## Data quality and reconciliation boundary

The local Python warehouse rejects duplicate grain, missing keys, unmapped foreign
keys, or reconciliation drift before persistence. Current file-based validation
covers 3,000,888 sales rows, 83,488 transaction rows, 1,704 dates, 54 stores, 33
families, 92,016 store-dates, and 7,938 holiday-bridge rows.

SQL quality files parse in dry-run mode. PostgreSQL runtime DDL execution, load,
marts, and SQL reconciliation remain `NOT RUN`; this does not reduce the evidence
for the completed local Parquet/Power BI work, but it remains a deployment
limitation.

## Analysis and dashboard boundary

EDA and dashboard measures describe Sales Volume and observed associations. They do
not provide causal evidence for promotion or holiday impact. The completed PBIX
contains the following pages:

1. Executive Overview
2. Sales Trend & Seasonality
3. Store Performance
4. Product Family Performance
5. Promotion Analysis
6. Holiday & Event Analysis
7. Transactions & Oil Drivers
8. Forecast Readiness & Anomalies

Page 8 switches readiness and anomaly views using bookmarks. Full model details are
in [Power BI Model](powerbi_model.md).

## DS transition boundary

Forecast readiness is historical diagnostic metadata, not a trained model or a
guarantee of accuracy. The future DS phase must use 16-day walk-forward validation,
forecast-origin-safe features, baselines, leakage checks, error analysis, and
versioned model artifacts. No feature generation, training, or prediction is part
of the completed DA architecture.
