# Power BI Model Design

> Historical design-stage evidence. The design was subsequently implemented in
> `powerbi/store_sales_analytics.pbix`. Current pages, measures, bookmarks, status,
> and cautions are documented in `docs/powerbi_model.md`.

## Modeling decision

Power BI uses `DimStoreDate` as the primary store-day and holiday filtering
dimension. It is the complete Cartesian set of every analysis date and store, so
holiday slicers retain both holiday and non-holiday store-days. The
holiday-only `BridgeStoreHoliday` remains available for audit and detailed event
inspection, but it is not the primary slicer table and does not create an active
relationship path to the facts.

All active relationships use **single-direction filtering** from the one side to
the many side. The model uses no many-to-many relationships and no bidirectional
filtering. Bidirectional filtering must not be enabled as a workaround for holiday
filtering.

## Table contracts

| Power BI table | Physical source | Grain | Primary key | Foreign keys and retained audit keys | Model role |
| --- | --- | --- | --- | --- | --- |
| `DimDate` | `dim_date.parquet` | One row per calendar date | `date_key` | None | Date attributes and date slicers |
| `DimStore` | `dim_store.parquet` | One row per store | `store_key` | None | Store attributes and store slicers |
| `DimFamily` | `dim_family.parquet` | One row per product family | `family_key` | None | Family slicers |
| `DimStoreDate` | `dim_store_date.parquet` | One row per `date_key + store_key`, including non-holidays and dates without fact observations | `date_store_key`; alternate unique key `date_key + store_key` | `date_key → DimDate`; `store_key → DimStore` | Primary holiday/event slicer and conformed store-day filter |
| `FactDailySales` | `fact_daily_sales.parquet` | One observed row per `date_key + store_key + family_key` | `sales_id`; business grain is also unique | `date_store_key → DimStoreDate`; `family_key → DimFamily`; `date_key` and `store_key` remain for audit/SQL | Sales measures |
| `FactStoreTransactions` | `fact_store_transactions.parquet` | One observed row per `date_key + store_key` | Composite `date_key + store_key` | Local composite `(date_key, store_key) → DimStoreDate`; Power BI derives `date_store_key` after import as a semantic helper | Transaction measures |
| `FactOilPrice` | `fact_oil_price.parquet` | One row per calendar date | `date_key` | `date_key → DimDate` | Daily oil measures |
| `BridgeStoreHoliday` | `bridge_store_holiday.parquet` | One row per holiday/event `date_key + store_key`; non-event store-days are absent | Composite `date_key + store_key` | `date_key → DimDate`; `store_key → DimStore` in warehouse SQL | Audit/detail only; not the main slicer and not an active fact-filter path |

## Active relationships

| One side | Many side | Join key | Cardinality | Filter direction |
| --- | --- | --- | --- | --- |
| `DimDate` | `DimStoreDate` | `date_key` | `1 → *` | Single: `DimDate → DimStoreDate` |
| `DimStore` | `DimStoreDate` | `store_key` | `1 → *` | Single: `DimStore → DimStoreDate` |
| `DimStoreDate` | `FactDailySales` | `date_store_key` | `1 → *` | Single: `DimStoreDate → FactDailySales` |
| `DimStoreDate` | `FactStoreTransactions` | semantic helper `date_store_key` | `1 → *` | Single: `DimStoreDate → FactStoreTransactions` |
| `DimFamily` | `FactDailySales` | `family_key` | `1 → *` | Single: `DimFamily → FactDailySales` |
| `DimDate` | `FactOilPrice` | `date_key` | `1 → *` | Single: `DimDate → FactOilPrice` |

The local transaction fact contains only `date_key`, `store_key`, and
`transactions`. Power BI derives `date_store_key = date_key * 100 + store_key`
after import; this is a semantic-model helper, not a persisted warehouse column.
Although the facts retain `date_key` and `store_key` for audit and SQL, Power BI
must not add active direct relationships from `DimDate` or `DimStore` to the two
store-day facts. Those relationships would duplicate the active path through
`DimStoreDate` and introduce ambiguity. Retained keys do not imply extra active
relationships.

## Slicer ownership

| Slicer purpose | Use fields from | Do not use as the primary source |
| --- | --- | --- |
| Date, year, quarter, month, weekday, payday | `DimDate` | Fact date columns |
| Store, city, state, type, cluster | `DimStore` | Fact store columns |
| Family | `DimFamily` | Fact family keys |
| Holiday versus non-holiday | `DimStoreDate[is_holiday]` | `BridgeStoreHoliday` |
| Event and work-day flags | `DimStoreDate[is_event]`, `DimStoreDate[is_work_day]` | Fact rows |
| Holiday count/type/locale/description | `DimStoreDate` | The holiday-only bridge for general slicing |
| Observation availability | `DimStoreDate[has_sales_observation]`, `DimStoreDate[has_transaction_observation]` | Inferring availability from zero-valued measures |

`BridgeStoreHoliday` may be hidden from report view or loaded only into an audit
page. If exposed for detail, keep it outside the active fact-filtering model so it
does not create a second path or tempt a many-to-many relationship.

## Why `DimStoreDate` is required

Holiday applicability depends on both date and store geography. `DimDate` alone
cannot represent a local or regional holiday correctly, while a holiday-only bridge
omits every non-holiday store-day. `DimStoreDate` provides one unique key for the
complete date-store grid and carries holiday, event, work-day, and observation
flags on the same conformed grain. A holiday slicer can therefore select either
state without dropping the non-holiday comparison population by construction.

The current processed data validates this design:

- `DimStoreDate`: 92,016 unique store-days.
- `is_holiday = 1`: 4,816 store-days.
- `is_holiday = 0`: 87,200 store-days.
- Holiday/event mappings with `holiday_count > 0`: 7,938 store-days; this count is
  larger than the holiday flag count because events and work days are retained too.
- Orphan persisted sales `date_store_key` values: 0; transaction rows map to
  `DimStoreDate` by the local composite `(date_key, store_key)` with 0 orphans.

## Missing observation is not zero sales

`has_sales_observation = 0` means no sales fact row was observed for that store-day.
It does not mean an observed sales value of zero. Likewise,
`has_transaction_observation = 0` means no transaction row exists, not that an
observed transaction count equals zero. Actual zero measures remain fact rows and
therefore have an observation flag of 1.

Measures should preserve this distinction. A report may deliberately display zero
with a separately named presentation measure, but it must not overwrite the base
semantic meaning or use `COALESCE` to recode missing observations without an
explicit business rule. Availability slicers and quality diagnostics must use the
two observation flags from `DimStoreDate`.

## Original Power BI implementation checklist

1. Load the seven core model tables and optionally retain `BridgeStoreHoliday` for
   audit/detail.
2. Mark `DimDate` as the date table using its date column.
3. Create only the six active relationships listed above.
4. Set every relationship cross-filter direction to Single.
5. Hide surrogate and retained audit keys from report consumers while keeping them
   in the model.
6. Build holiday slicers from `DimStoreDate`, and validate that both 0 and 1 are
   selectable for `is_holiday`.
7. Do not add many-to-many or bidirectional relationships to compensate for a
   measure or slicer issue; correct the measure or model path instead.

This historical design task itself did not create a PBIX. The later implementation
is now complete at `powerbi/store_sales_analytics.pbix`; this statement must not be
read as the current project status.
