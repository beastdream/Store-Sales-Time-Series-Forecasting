# Store Sales — Time Series Forecasting

## Project Overview

This repository contains the completed Data Engineering, Data Analysis, Power BI,
and forecast-readiness phases for the Corporación Favorita store-sales dataset.
It builds validated analytical data, a dimensional model, reproducible EDA outputs,
and an eight-page interactive Power BI dashboard. Forecast model development has
not started and belongs to the next phase.

## Business / Analytical Problem

The completed analytical phase examines how **Sales Volume** varies by time, store,
product family, promotion activity, holidays/events, transactions, and oil-price
context. It also assesses whether each store–family series has enough history and
stability for forecasting.

`sales` represents Sales Volume, not revenue or profit. The dataset contains no
unit prices, cost, profit, inventory, stockout, margin, or lead-time fields, so the
project cannot measure profitability or translate a forecast directly into an
inventory order.

Promotion and holiday views report descriptive associations, comparisons, and
proxy differences. They do not establish causal uplift.

## Dataset

The source includes competition CSVs for train/test sales, stores, transactions,
oil prices, holidays/events, and sample submission. Raw files are expected locally
under `data/raw/` and are excluded by `.gitignore`.

- Historical actual-sales period: **2013-01-01 through 2017-08-15**.
- The 2017 actual-sales period is partial and must not be compared naively with
  complete calendar years.
- Competition/test period: **2017-08-16 through 2017-08-31**.
- The date dimension extends through 2017-08-31 to support the test horizon, but
  dashboard labels and historical-actual metrics use 2017-08-15 as the final
  observed sales date.
- Stores: **54**.
- Product families: **33**.
- Potential store–family series: **1,782**.

Validated warehouse totals and row counts are recorded in the current
[warehouse reconciliation](reports/data_quality/warehouse_reconciliation.md).

## Project Architecture

```text
Raw CSV
  → Cleaning / Validation
  → Interim Parquet
  → Analytical Warehouse / Dimensional Model
  → Processed Parquet
  → EDA / Analytical Reports
  → Power BI Dashboard
  → Forecast Readiness Assessment
  → [Next phase] Feature Engineering / Backtesting / Forecast Models
```

The implemented and planned boundaries are detailed in
[Architecture](docs/architecture.md). Physical table contracts are in the
[Data Dictionary](docs/data_dictionary.md).

## Data Pipeline

The Python pipeline validates required columns, types, numeric domains, source and
target grains, foreign-key mappings, missing surrogate keys, row counts, date
ranges, and measure reconciliation before writing artifacts.

Key grains are:

- `FactDailySales`: date × store × family.
- `FactStoreTransactions`: date × store; transactions are never expanded across
  product families.
- `DimStoreDate`: complete date × store context grid, including observation flags.
- `FactOilPrice`: calendar date.
- `ForecastReadiness`: store × family.

`DimStoreDate` preserves the difference between a missing source observation and
an observed zero. The local Parquet warehouse is validated. PostgreSQL DDL, loaders,
marts, and quality SQL exist, but PostgreSQL runtime execution is still **NOT RUN**
because no configured database was available; a SQL parse dry-run is not a runtime
database validation.

## Data Analysis

The DA notebooks cover audit and cleaning, store/family performance,
trend/seasonality, descriptive promotion comparison, holiday/event comparison,
transactions and oil drivers, anomaly review, and forecast readiness. Tables,
figures, and notes are under `reports/`; the consolidated interpretation is in
[Business Insights](reports/business_insights.md).

Selected report-verified findings include:

- GROCERY I contributes 32.0% of Sales Volume and BEVERAGES contributes 20.2%;
  together they account for about 52.2%.
- Store 44 has the highest average daily Sales Volume at 36,869.09.
- Store-day Sales Volume and transactions have a Pearson correlation of 0.837;
  this is an association, not proof of causation.
- Recent store growth compares 2017-05-18–2017-08-15 with the immediately
  preceding 90-day window. The first-versus-last 90-day metric remains explicitly
  labeled as a long-history proxy rather than recent growth.

## Power BI Dashboard

The completed local report is
[`powerbi/store_sales_analytics.pbix`](powerbi/store_sales_analytics.pbix). Its
metadata confirms eight analytical pages:

1. Executive Overview
2. Sales Trend & Seasonality
3. Store Performance
4. Product Family Performance
5. Promotion Analysis
6. Holiday & Event Analysis
7. Transactions & Oil Drivers
8. Forecast Readiness & Anomalies

The report provides interactive filtering and drillable analytical views. Page 8
uses bookmark navigation between **Forecast Readiness** and **Sales Anomalies**.
The first bookmark shows `FR_Group` and hides `AN_Group`; the second reverses that
visibility. Visual comparisons of promotion and holidays remain descriptive and
must not be interpreted as causal effects.

The semantic model, relationships, measures, slicers, hidden-key guidance, pages,
and bookmarks are documented in [Power BI Model](docs/powerbi_model.md). The file
is completed locally; publication, gateway configuration, scheduled refresh, and
Power BI Service operation are not evidenced by this repository.

## Forecast Readiness

The assessment covers all 54 × 33 = 1,782 store–family series. Primary classes are
Ready 364, Ready with caution 345, Intermittent demand 417, Insufficient history
144, High volatility 102, and Promotion dependent 410. Independent flags overlap:
438 series carry at least two risk flags.

These are historical data-readiness rules, not forecast accuracy results or model
features by default. Thresholds and priority rules are documented in
[Forecast Readiness](reports/forecast_readiness.md).

## Data Science / Forecasting — Next Phase

The intended future problem is:

- Forecast target: `sales` (Sales Volume).
- Forecast grain: **Store × Product Family × Day**.
- Historical actual-sales end: **2017-08-15**.
- Competition/test horizon: **2017-08-16 through 2017-08-31**.
- Forecast horizon: **16 days**.
- Expected predictions: **28,512 rows** (`54 × 33 × 16`).

Temporal validation, baseline evaluation, feature engineering, machine-learning
forecasting, model evaluation, prediction intervals, and final prediction
generation are planned work. No forecasting accuracy is claimed. See the
[Data Science Roadmap](docs/data_science_roadmap.md).

## Validation / Testing

The current repository test suite was rerun on 2026-08-08: **135 tests passed**.
The reproducible validator checks cleaning, warehouse build, Parquet readability,
grain, reconciliations, artifacts, readiness flags, Git hygiene, and SQL quality
dry-run:

```bash
python -m src.validate_da_project
```

The latest file-based evidence is in
[DA Project Validation](reports/da_project_validation.md). PostgreSQL runtime is
explicitly `NOT RUN`, not PASS or FAIL.

## Known Limitations

- Sales is volume, not revenue; price and financial measures are unavailable.
- Cost, profit, inventory, stockouts, margins, and lead times are unavailable.
- Promotion and holiday analyses are descriptive associations/proxy differences,
  not causal inference.
- Anomaly flags identify observations for review; they do not prove data errors.
- 2017 actual sales end on August 15 and form only a partial year.
- PostgreSQL runtime DDL/load/mart/quality validation has not run.
- Power BI Service publication, gateway, scheduled refresh, and production access
  are not verified in the repository.
- No feature dataset, trained forecast model, backtest score, interval, or final
  submission has been produced in the Data Science phase.

## Repository Structure

```text
data/
  raw/          Local source CSVs (ignored)
  interim/      Cleaned Parquet artifacts (generated, ignored)
  processed/    Validated dimensional warehouse Parquet (generated, ignored)
  features/     Reserved for the future DS phase; currently placeholder only
docs/           Data dictionary, architecture, Power BI model, DS roadmap
models/         Reserved for future trained artifacts; currently placeholder only
notebooks/      Executable DA notebook scripts
powerbi/        Completed local Power BI report
reports/        Analytical tables, figures, insights, and validation evidence
sql/            PostgreSQL DDL, marts, and quality SQL
src/            Cleaning, warehouse, database, and validation modules
tests/          Automated contracts and regression tests
```

## How to Run

Use Python 3.11 or a compatible environment. Supply the competition CSVs under
`data/raw/` before running the pipeline.

```bash
python -m pip install -r requirements.txt
python notebooks/01_data_audit.py
python -m src.data.run_cleaning
python -m src.data.run_warehouse_build
python notebooks/03_date_dimension.py
python notebooks/04a_sales_trend_seasonality.py
python notebooks/04_business_eda.py
python notebooks/05_promotion_analysis.py
python notebooks/06_holiday_analysis.py
python notebooks/07_transactions_analysis.py
python notebooks/08_anomaly_review.py
python notebooks/09_forecast_readiness.py
python -m pytest -q
python -m src.run_sql_quality_checks --dry-run
```

Notebook 02 is an interactive wrapper around the canonical cleaning module.
PostgreSQL loading remains optional and must not be represented as validated until
a configured database run and reconciliation pass.

## Current Project Status

| Phase | Status |
|---|---|
| Phase 1 — Data Engineering | **Complete** |
| Phase 2 — Data Analysis / EDA | **Complete** |
| Phase 3 — Power BI Dashboard | **Complete** |
| Phase 4 — Forecast Readiness Assessment | **Complete** |
| Phase 5 — Forecasting / Data Science | **Not started — next phase** |

This task boundary is documentation readiness for the DS handoff. No forecasting
implementation or trained model is included.
