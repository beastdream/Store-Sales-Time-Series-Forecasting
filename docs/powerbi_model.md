# Power BI Model

## Completed report status

The completed local report is
[`powerbi/store_sales_analytics.pbix`](../powerbi/store_sales_analytics.pbix).
Repository inspection confirms eight 1280 × 720 report pages and two Page 8
bookmarks. This document describes the completed local model/report; it does not
claim Power BI Service publication, gateway configuration, scheduled refresh, or
production access.

PostgreSQL runtime remains `NOT RUN`. The dashboard completion is based on the
local PBIX and validated file-based analytical model, not a claimed database
deployment.

## Tables loaded into the analytical model

| Table | Grain | Purpose |
|---|---|---|
| `DimDate` | date | Calendar filtering and labels |
| `DimStore` | store | Store, geography, type, and cluster filtering |
| `DimFamily` | product family | Family filtering |
| `DimStoreDate` | date × store | Conformed store-day context, holiday/event and observation flags |
| `FactDailySales` | date × store × family | Sales Volume and promotion observations |
| `FactStoreTransactions` | date × store | Store-day transaction measures |
| `FactOilPrice` | date | Oil price and change context |
| `ForecastReadiness` | store × family | Historical readiness classes, flags, and series metrics |
| `SalesAnomalies` | anomaly observation | Historical anomaly-review detail |
| `_Measures` | measure table | Central DAX measure organization |

The processed `BridgeStoreHoliday` remains useful for audit/detail outside the
primary semantic path. `DimStoreDate` is used for dashboard holiday filtering
because it contains both holiday and regular store-days.

## Important relationships

All active relationships should use one-to-many cardinality and single-direction
filtering from dimensions to facts:

| One side | Many side | Key | Filter direction |
|---|---|---|---|
| `DimDate` | `DimStoreDate` | `date_key` | `DimDate → DimStoreDate` |
| `DimStore` | `DimStoreDate` | `store_key` | `DimStore → DimStoreDate` |
| `DimStoreDate` | `FactDailySales` | `date_store_key` | `DimStoreDate → FactDailySales` |
| `DimStoreDate` | `FactStoreTransactions` | `date_store_key` | `DimStoreDate → FactStoreTransactions` |
| `DimFamily` | `FactDailySales` | `family_key` | `DimFamily → FactDailySales` |
| `DimDate` | `FactOilPrice` | `date_key` | `DimDate → FactOilPrice` |

```text
DimDate ───────┐
               ▼
          DimStoreDate ─────► FactDailySales ◄───── DimFamily
               │
DimStore ──────┘
               └────────────► FactStoreTransactions

DimDate ─────────────────────► FactOilPrice
```

The retained `date_key` and `store_key` columns in facts are audit keys, not a
reason to create duplicate active paths around `DimStoreDate`. Family filtering
must not be assumed to filter `FactStoreTransactions`, because transactions do not
exist at family grain.

## Grain and observation semantics

- `FactDailySales`: one observed date–store–family row.
- `FactStoreTransactions`: one observed date–store row; never repeated by family.
- `FactOilPrice`: one calendar-date row.
- `DimStoreDate`: complete date–store grid.
- `ForecastReadiness`: one store–family row.
- `SalesAnomalies`: one review record at its documented analysis level.

`has_sales_observation = 0` and `has_transaction_observation = 0` mean a source
row is absent; they do not mean an observed measure of zero. Dashboard labels use
2017-08-15 as the final actual-sales date even though `DimDate` extends to the test
horizon on 2017-08-31.

## Important measures

The report organizes reusable measures in `_Measures`. Important measure groups
visible in the PBIX metadata include:

- Core volume: `Total Sales Volume`, `Average Daily Sales`, `Sales 7D Moving
  Average`, `Sales 28D Moving Average`, `YoY Growth %`.
- Coverage: `Active Stores`, `Active Families`, `Zero-Sales Observation Rate`.
- Transactions: `Total Transactions`, `Average Daily Transactions`, `Sales Volume
  per Transaction`.
- Store momentum: `Recent 90D Sales`, `Previous 90D Sales`, `Recent 90D Growth %`.
- Promotion comparison: `Promotion-Active Observation Rate`, `Promotion-Active
  Sales Volume`, `Nonpromotion Sales Volume`, and `Promotion Difference Proxy %`.
- Holiday comparison: holiday/regular Sales Volume and average-store-day measures,
  plus `Holiday Difference Proxy %`.
- Oil context: latest/average price, 1-day and 7-day changes, and imputed-day count.
- Readiness: `Total Series`, `Ready Series`, `Ready Series %`, intermittent,
  promotion-dependent and high-volatility series counts, and FR series metrics.
- Anomalies: count, actual Sales Volume, expected Sales Volume, and baseline bias.

Names containing “Difference Proxy” represent descriptive comparisons, not causal
uplift. Sales measures represent volume, not revenue or profit.

## Technical keys and summarization

Technical and surrogate keys should not be summarized. They should normally be
hidden from report users after relationships are configured, including:

- `date_key`, `store_key`, `family_key`, `date_store_key`, and `sales_id`;
- composite/report join helpers and technical anomaly identifiers where not needed
  for drill-through;
- raw numeric keys retained only for model relationships or audit.

Business identifiers such as store number may remain visible as categorical labels,
but their default summarization must be **Do not summarize**.

## Slicers and interactive filtering

The report includes interactive slicers appropriate to each view. Ownership should
remain consistent:

| Filter | Source |
|---|---|
| Date/year/month/weekday | `DimDate` |
| Store/state/city/type/cluster | `DimStore` |
| Product family | `DimFamily` |
| Holiday/event/work-day | `DimStoreDate` |
| Observation availability | `DimStoreDate` |
| Readiness class | `ForecastReadiness` |
| Anomaly level/method/review category | `SalesAnomalies` |

Date, family, state, store type, and readiness/anomaly controls support interactive
filtering and drillable analytical views. Fact keys should not be exposed as user
slicers.

## Report page structure

1. **Executive Overview** — core Sales Volume, transactions, activity, trend, and
   store/family overview.
2. **Sales Trend & Seasonality** — moving averages, month/weekday patterns, and
   partial-year-aware trend views.
3. **Store Performance** — store rankings, recent 90-day momentum, transaction
   context, and store attributes.
4. **Product Family Performance** — contribution, variability, intermittency, and
   promotion-active comparisons by family.
5. **Promotion Analysis** — descriptive promotion-active versus nonpromotion
   comparisons and proxy differences.
6. **Holiday & Event Analysis** — holiday/regular store-day comparisons, event
   types/locales, and time patterns.
7. **Transactions & Oil Drivers** — transactions, Sales Volume per transaction,
   oil context, and descriptive driver views.
8. **Forecast Readiness & Anomalies** — two bookmark-controlled analytical views.

## Page 8 bookmark navigation

The Page 8 bookmark navigator switches between the two groups verified in the PBIX:

```text
Forecast Readiness bookmark
    → shows FR_Group
    → hides AN_Group

Sales Anomalies bookmark
    → hides FR_Group
    → shows AN_Group
```

Both bookmarks target the `Forecast Readiness & Anomalies` page. Bookmark state
inspection confirms that the two visual groups invert their `isHidden` state.

## Model cautions

- Keys must use **Do not summarize** and normally remain hidden.
- Family filters must not be presented as if transactions exist by family.
- Missing observations must not be silently converted to observed zero.
- Promotion and holiday visuals show descriptive association/proxy difference,
  not causal effect.
- Historical actual Sales Volume ends on 2017-08-15; 2017 is a partial year.
- Forecast-readiness classes and anomaly outputs are historical diagnostics, not
  forecast accuracy or automatically valid model features.
- Power BI Service refresh, credentials, gateway, and access controls remain an
  operational follow-up outside the local PBIX evidence.
