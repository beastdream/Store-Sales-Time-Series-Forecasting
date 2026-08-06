# Store Sales — Time Series Forecasting

## Project overview

This repository builds a reproducible analytics foundation for the Corporación
Favorita store-sales dataset: validated raw-data cleaning, a dimensional warehouse,
business EDA, forecast-readiness assessment, and a documented Power BI model. The
current scope stops before model training and before a finished Power BI report.

## Business problem

The project identifies how sales volume varies across dates, stores, product
families, promotions, holidays, and store traffic, then assesses whether each
store–family series is suitable for forecasting. In this dataset, `sales` is
**sales volume, not revenue**. There is no cost, profit, inventory, stockout,
lead-time, or margin data, so the analysis cannot measure profitability or convert
a forecast directly into an ordering decision.

## Dataset

The source consists of the competition CSVs for train/test sales, stores,
transactions, oil prices, holidays/events, and sample submission. Raw CSVs belong
under `data/raw/` and are intentionally excluded from Git. The observed sales fact
runs from 2013-01-01 through 2017-08-15; the full analysis calendar extends through
the test horizon on 2017-08-31.

Validated warehouse totals are 1,073,644,952.2030685 units of sales volume,
7,810,622 promoted items, and 141,478,945 transactions across 54 stores and 33
families. These values come from the current
[warehouse reconciliation](reports/data_quality/warehouse_reconciliation.md).

## Data grain

The central grains are:

- Sales: one observed row per date × store × family.
- Transactions: one observed row per date × store; transactions are never expanded
  across the 33 families.
- Store-date context: one row per date × store over the complete analysis calendar.
- Oil: one row per calendar date.
- Forecast readiness: one row per store × family, or 1,782 series.

See the complete [data dictionary](docs/data_dictionary.md).

## Project architecture

```text
Raw CSV
  → Cleaning
  → Interim Parquet
  → Warehouse Build
  → Processed Parquet
  → Business EDA
  → Reports
  → Power BI
  → Future DS Forecasting
```

Responsibility and output ownership are documented in
[Architecture](docs/architecture.md).

## Data-quality controls

The Python pipeline validates required columns, numeric domains, source and target
grain, foreign-key mappings, missing surrogate keys, row counts, date ranges, and
measure reconciliation before persisting outputs. Current processed tables have
zero duplicate grains and zero missing keys. Sales, promotions, transactions,
dimension counts, store-date coverage, and observation flags reconcile to their
clean sources.

SQL DDL and data-quality queries also exist, but the PostgreSQL runtime remains
**unvalidated** because no database connection/configuration was available during
project validation. SQL syntax has been dry-run parsed; this is not evidence that
DDL, loads, marts, or checks ran successfully in PostgreSQL. See
[DA project validation](reports/da_project_validation.md).

## Data warehouse model

The warehouse contains conformed date, store, family, and store-date dimensions;
sales, transaction, and oil facts; plus a holiday bridge. `dim_store_date` is the
complete date–store grid and preserves the crucial distinction between “no source
observation” and an observed zero. It also supplies the intended Power BI holiday
filter path without duplicating transactions at family grain.

The physical contracts are in the [data dictionary](docs/data_dictionary.md), and
the flow is described in [Architecture](docs/architecture.md).

## Business analysis

The notebook scripts cover audit/cleaning, warehouse construction, store and family
performance, trend/seasonality, promotion association, holiday association,
transactions, anomaly review, and forecast readiness. Generated tables and figures
are stored under `reports/`; the consolidated interpretation is
[Business insights](reports/business_insights.md).

Promotion comparisons are observational associations and **not causal effects**.
Promotion assignment is not randomized, and price, campaign selection, demand,
inventory, and other confounders are unavailable.

## Key findings

Only metrics already published in project reports are summarized here:

- GROCERY I contributes 32.0% of sales volume and BEVERAGES contributes 20.2%; the
  two families together account for about 52.2%.
- Store 44 has the highest average daily sales volume at 36,869.09.
- The recent store-growth window is 2017-05-18–2017-08-15 versus the immediately
  preceding 2017-02-17–2017-05-17 window. The older first-versus-last 90-day metric
  remains explicitly labeled as a proxy, not recent growth.
- Total sales and store-day transactions have a reported Pearson correlation of
  0.837; this is association, not proof that traffic causes sales changes.
- Forecast readiness contains 709 series with zero serious risk flags, 635 with
  one, 380 with two, and 58 with three or more.

## Limitations

- `sales` is volume, not revenue; unit prices are unavailable.
- Cost, profit, inventory, stockouts, lead times, and margins are unavailable.
- Promotion and holiday analyses are descriptive/associational, not causal.
- Anomaly flags identify observations for review; they do not prove data errors.
- 2017 is incomplete after August, so naive full-year comparisons are invalid.
- PostgreSQL execution and reconciliation have not been runtime-validated.
- No forecast model, backtest, prediction interval, or approved model artifact
  exists yet.

## Forecast readiness

The readiness artifact covers all 54 × 33 store–family series. The primary classes
are Ready 364, Ready with caution 345, Intermittent demand 417, Insufficient history
144, High volatility 102, and Promotion dependent 410. Independent risk flags may
overlap; 438 series carry at least two risks. Thresholds, priority rules, and
family/store summaries are documented in
[Forecast readiness](reports/forecast_readiness.md).

## How to reproduce

Use Python 3.11 or a compatible environment. Raw source CSVs must be supplied
locally under `data/raw/`; they are not committed.

```bash
python -m pip install -r requirements.txt
python notebooks/01_data_audit.py
python -m src.data.run_cleaning
python -m src.data.run_warehouse_build
python notebooks/03_date_dimension.py
python notebooks/04_business_eda.py
python notebooks/04a_sales_trend_seasonality.py
python notebooks/05_promotion_analysis.py
python notebooks/06_holiday_analysis.py
python notebooks/07_transactions_analysis.py
python notebooks/08_anomaly_review.py
python notebooks/09_forecast_readiness.py
python -m pytest -v
```

Notebook 02 wraps the cleaning workflow and may be used interactively, but the
module command above is the canonical pipeline entry point. PostgreSQL loading is
optional and requires a real `.env` created from `.env.example`; consult the
validation report before treating its outputs as verified.

## Power BI status

The semantic model and expected relationships are documented, but Power BI is
**not completed**. No screenshot, `.pbix`, or `.pbit` is provided or claimed. The
intended model uses `dim_store_date` as the conformed store-day/holiday path with
single-direction one-to-many relationships. See
[Power BI model](docs/powerbi_model.md).

## Data Science next step

Define the forecast horizon and temporal splits, establish seasonal-naive and
intermittent-demand baselines, then evaluate models by readiness group using both
point-error and interval metrics. Promotion availability must be tested as a future
feature scenario, and every model should retain fallbacks for insufficient-history
and intermittent series.
