# Power BI Model

## Status

This document is an implementation contract, not evidence of a completed Power BI
deliverable. No `.pbix`, `.pbit`, screenshot, published workspace, refresh, or
gateway has been created or validated in this repository. PostgreSQL runtime is
also unvalidated, so Power BI should initially consume the validated processed
Parquet tables or a database load that has subsequently passed all checks.

## Modeling principle

`DimStoreDate` is the conformed store-day and holiday-filtering dimension. It
contains every date × store combination, including non-holidays and store-days
without fact observations. This avoids using the holiday-only bridge as a slicer
and preserves “missing observation” separately from an observed zero.

All active relationships are one-to-many with **single-direction** filtering from
dimensions toward facts. Do not introduce bidirectional or many-to-many paths as a
shortcut.

## Expected tables

| Power BI table | Physical source | Grain | Role |
|---|---|---|---|
| `DimDate` | `dim_date.parquet` | date | Calendar slicers |
| `DimStore` | `dim_store.parquet` | store | Store/geography slicers |
| `DimFamily` | `dim_family.parquet` | family | Product-family slicers |
| `DimStoreDate` | `dim_store_date.parquet` | date × store | Conformed store-day, holiday/event and observation flags |
| `FactDailySales` | `fact_daily_sales.parquet` | date × store × family | Sales-volume and promotion measures |
| `FactStoreTransactions` | `fact_store_transactions.parquet` | date × store | Transaction measures without family duplication |
| `FactOilPrice` | `fact_oil_price.parquet` | date | Oil-price context |
| `BridgeStoreHoliday` | `bridge_store_holiday.parquet` | applicable event date × store | Audit/detail only; optional in report view |

`forecast_readiness.csv` may be loaded as a store–family assessment table for a
dedicated readiness page, but it is not part of the core star and should not create
ambiguous active paths. If used, create a unique composite store-family key or a
dedicated bridge after validating cardinality.

## Active relationships

| One side | Many side | Key | Cardinality and filter |
|---|---|---|---|
| `DimDate` | `DimStoreDate` | `date_key` | `1 → *`, single direction |
| `DimStore` | `DimStoreDate` | `store_key` | `1 → *`, single direction |
| `DimStoreDate` | `FactDailySales` | `date_store_key` | `1 → *`, single direction |
| `DimStoreDate` | `FactStoreTransactions` | `date_store_key` | `1 → *`, single direction |
| `DimFamily` | `FactDailySales` | `family_key` | `1 → *`, single direction |
| `DimDate` | `FactOilPrice` | `date_key` | `1 → *`, single direction |

Although both store-day facts retain `date_key` and `store_key` for audit/SQL, do
not create additional active links from `DimDate` or `DimStore` directly to those
facts. The second path would conflict with the active path through `DimStoreDate`.

`BridgeStoreHoliday` should remain disconnected from the facts in the active
semantic model, or be hidden/audit-only. Its non-event population is absent, so it
cannot support a correct holiday-versus-non-holiday slicer.

## Intended filter flow

```text
DimDate ───────┐
               ▼
          DimStoreDate ─────► FactDailySales ◄───── DimFamily
               │
DimStore ──────┘
               └────────────► FactStoreTransactions

DimDate ─────────────────────► FactOilPrice
```

Examples:

- A date filter flows from `DimDate` through `DimStoreDate` to both store-day facts.
- A city/store filter flows from `DimStore` through `DimStoreDate` to both facts.
- A family filter affects sales only; it must not multiply or filter transactions
  as though transactions existed at family grain.
- Holiday and observation slicers come from `DimStoreDate`.

## Slicer ownership

| Slicer | Use | Avoid |
|---|---|---|
| Date/year/quarter/month/weekday/payday | `DimDate` | Fact date columns |
| Store/city/state/type/cluster | `DimStore` | Fact store keys |
| Product family | `DimFamily` | Fact family key |
| Holiday/non-holiday | `DimStoreDate[is_holiday]` | Holiday-only bridge |
| Event/work-day | `DimStoreDate[is_event]`, `[is_work_day]` | Inferred fact flags |
| Observation availability | `DimStoreDate[has_sales_observation]`, `[has_transaction_observation]` | Treating missing rows as zero |

## Measure caveats

- Name sales measures as sales volume; do not label them revenue.
- Do not create profit or inventory measures because the required source fields do
  not exist.
- Sum transactions only from `FactStoreTransactions` at store-day grain.
- A missing fact row is not zero. If a presentation measure deliberately replaces
  blank with zero, name and document it separately from the base measure.
- Promotion comparisons are associations, not causal uplift estimates.

## Implementation checklist

1. Load the seven core tables; optionally load `BridgeStoreHoliday` for audit.
2. Mark `DimDate[full_date]` as the date table.
3. Verify uniqueness on every relationship’s one side.
4. Create only the six active relationships listed above.
5. Set every cross-filter direction to Single.
6. Hide technical keys while retaining them for relationship/audit use.
7. Confirm holiday slicers expose both 0 and 1 and do not drop non-event dates.
8. Reconcile total sales volume and transactions to the warehouse report.
9. Validate blanks versus observed zeros using the two observation flags.
10. Record refresh source, credentials handling, model version, and validation
    evidence when an actual Power BI file is created.

The underlying validated counts and a more detailed rationale are available in the
existing [Power BI model design](../reports/data_quality/powerbi_model_design.md).
