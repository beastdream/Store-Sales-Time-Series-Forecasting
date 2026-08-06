# Data Dictionary

Types below describe the processed Parquet artifacts; PostgreSQL-compatible types
are noted where useful. “Nullable” reflects the validated artifact contract, not
only the pandas dtype. All surrogate keys must map to their dimensions.

## `dim_date`

- **Source:** `data/processed/dim_date.parquet`
- **Grain:** one row per calendar date.
- **Primary key:** `date_key`.
- **Foreign keys:** none.
- **Important caveats:** covers the full train/test analysis calendar, including
  dates without sales observations. Weekday numbering is Monday = 0. Payday is the
  15th or calendar month-end.

| Column | Type | Meaning | Nullable |
|---|---|---|---|
| `date_key` | int32 / INTEGER | Date surrogate in `YYYYMMDD` form | No |
| `full_date` | datetime / DATE | Calendar date | No |
| `day` | uint8 / SMALLINT | Day of month | No |
| `day_of_week` | uint8 / SMALLINT | Monday-based weekday number, 0–6 | No |
| `day_name` | string | English weekday name | No |
| `week_of_year` | uint8 / SMALLINT | ISO week number | No |
| `month` | uint8 / SMALLINT | Calendar month, 1–12 | No |
| `month_name` | string | English month name | No |
| `quarter` | uint8 / SMALLINT | Calendar quarter, 1–4 | No |
| `year` | int16 / SMALLINT | Calendar year | No |
| `is_weekend` | uint8 / BOOLEAN | 1 for Saturday/Sunday | No |
| `is_month_start` | uint8 / BOOLEAN | 1 on the first calendar day of month | No |
| `is_month_end` | uint8 / BOOLEAN | 1 on calendar month-end | No |
| `is_payday` | uint8 / BOOLEAN | 1 on the 15th or month-end | No |

## `dim_store`

- **Source:** `data/processed/dim_store.parquet`
- **Grain:** one row per source store number.
- **Primary key:** `store_key`.
- **Foreign keys:** none.
- **Important caveats:** `store_nbr` is the stable business key; `store_type` is
  source metadata and does not encode a documented causal/business hierarchy.

| Column | Type | Meaning | Nullable |
|---|---|---|---|
| `store_key` | int32 / INTEGER | Warehouse store surrogate key | No |
| `store_nbr` | uint8 / SMALLINT | Source store business identifier | No |
| `city` | string | Store city | No |
| `state` | string | Store state/region | No |
| `store_type` | string | Source store type category | No |
| `cluster` | uint8 / SMALLINT | Source grouping of similar stores | No |

## `dim_family`

- **Source:** `data/processed/dim_family.parquet`
- **Grain:** one row per product family.
- **Primary key:** `family_key`.
- **Foreign keys:** none.
- **Important caveats:** family is a product grouping, not an individual SKU;
  item-level assortment, price, cost, and margin are unavailable.

| Column | Type | Meaning | Nullable |
|---|---|---|---|
| `family_key` | int32 / INTEGER | Warehouse family surrogate key | No |
| `family` | category / VARCHAR | Source product-family label | No |

## `dim_store_date`

- **Source:** `data/processed/dim_store_date.parquet`
- **Grain:** one row per `date_key + store_key` across the complete calendar-store
  grid.
- **Primary key:** `date_store_key`; `date_key + store_key` is also unique.
- **Foreign keys:** `date_key → dim_date.date_key`; `store_key → dim_store.store_key`.
- **Important caveats:** this is the intended Power BI store-day filter dimension.
  Observation flags distinguish missing source rows from observed zero measures.
  Holiday text fields may aggregate multiple applicable events.

| Column | Type | Meaning | Nullable |
|---|---|---|---|
| `date_store_key` | int64 / BIGINT | Unique store-date key | No |
| `date_key` | int32 / INTEGER | Calendar-date FK | No |
| `store_key` | int32 / INTEGER | Store FK | No |
| `holiday_count` | int64 / INTEGER | Applicable source holiday/event records after aggregation | No |
| `holiday_descriptions` | string / TEXT | Deterministically aggregated descriptions; empty when none | No |
| `holiday_types` | string / TEXT | Aggregated holiday/event types | No |
| `holiday_locales` | string / TEXT | Aggregated National/Regional/Local scopes | No |
| `is_holiday` | uint8 / SMALLINT | 1 for an actual holiday after transfer rules | No |
| `is_work_day` | uint8 / SMALLINT | 1 for a source work-day exception | No |
| `is_event` | uint8 / SMALLINT | 1 for an event record | No |
| `has_sales_observation` | uint8 / SMALLINT | 1 if a sales source row exists at store-day grain | No |
| `has_transaction_observation` | uint8 / SMALLINT | 1 if a transaction source row exists | No |

## `fact_daily_sales`

- **Source:** `data/processed/fact_daily_sales.parquet`
- **Grain:** one observed row per `date_key + store_key + family_key`.
- **Primary key:** `sales_id`; business grain is also unique.
- **Foreign keys:** `date_key → dim_date`; `store_key → dim_store`;
  `date_store_key → dim_store_date`; `family_key → dim_family`.
- **Important caveats:** `sales` is **sales volume, not revenue** and may be
  fractional. An observed zero is retained. `onpromotion` is an item count, while
  `is_promotion` is only a binary convenience flag.

| Column | Type | Meaning | Nullable |
|---|---|---|---|
| `sales_id` | uint32 / INTEGER | Stable source sales-row identifier | No |
| `date_key` | int32 / INTEGER | Date FK | No |
| `store_key` | int32 / INTEGER | Store FK | No |
| `date_store_key` | int64 / BIGINT | Conformed store-date FK | No |
| `family_key` | int32 / INTEGER | Product-family FK | No |
| `sales` | float64 / NUMERIC(20,7) | Observed sales volume | No |
| `onpromotion` | uint16 / INTEGER | Number of promoted items | No |
| `is_promotion` | uint8 / SMALLINT | 1 when `onpromotion > 0` | No |

## `fact_store_transactions`

- **Source:** `data/processed/fact_store_transactions.parquet`
- **Grain:** one observed row per `date_key + store_key`.
- **Primary key:** composite `date_key + store_key`; `date_store_key` is unique at
  this grain.
- **Foreign keys:** `date_key → dim_date`; `store_key → dim_store`;
  `date_store_key → dim_store_date`.
- **Important caveats:** transactions are store-day measures and must not be joined
  directly to family-grain sales in a way that repeats them for every family.

| Column | Type | Meaning | Nullable |
|---|---|---|---|
| `date_key` | int32 / INTEGER | Date FK | No |
| `store_key` | int32 / INTEGER | Store FK | No |
| `date_store_key` | int64 / BIGINT | Conformed store-date FK | No |
| `transactions` | uint32 / INTEGER | Observed store transaction count | No |

## `fact_oil_price`

- **Source:** `data/processed/fact_oil_price.parquet`
- **Grain:** one row per analysis calendar date.
- **Primary key:** `date_key`.
- **Foreign keys:** `date_key → dim_date.date_key`.
- **Important caveats:** missing source prices are time-series imputed and marked;
  initial lag/change fields can legitimately be null. Oil association does not
  establish an economic causal effect on sales.

| Column | Type | Meaning | Nullable |
|---|---|---|---|
| `date_key` | int32 / INTEGER | Date PK/FK | No |
| `oil_price` | float32 / NUMERIC(10,4) | Cleaned daily oil price | No |
| `oil_change_1d` | float32 / NUMERIC | One-day absolute change | Yes |
| `oil_change_7d` | float32 / NUMERIC | Seven-day absolute change | Yes |
| `oil_pct_change_7d` | float32 / NUMERIC | Seven-day proportional change | Yes |
| `oil_was_imputed` | uint8 / SMALLINT | 1 when source price was missing | No |

## `bridge_store_holiday`

- **Source:** `data/processed/bridge_store_holiday.parquet`
- **Grain:** one applicable holiday/event record per `date_key + store_key`, after
  same-day events are aggregated.
- **Primary key:** composite `date_key + store_key`.
- **Foreign keys:** `date_key → dim_date`; `store_key → dim_store`.
- **Important caveats:** non-event store-days are absent. National events expand to
  all stores; regional/local events map by state/city. Use `dim_store_date` rather
  than this holiday-only bridge as the primary Power BI holiday slicer.

| Column | Type | Meaning | Nullable |
|---|---|---|---|
| `date_key` | int32 / INTEGER | Date FK | No |
| `store_key` | int32 / INTEGER | Store FK | No |
| `holiday_count` | int64 / SMALLINT | Number of aggregated applicable records | No |
| `holiday_descriptions` | string / TEXT | Aggregated descriptions | No |
| `holiday_types` | string / TEXT | Aggregated event types | No |
| `holiday_locales` | string / TEXT | Aggregated locale scopes | No |
| `is_holiday` | uint8 / SMALLINT | Actual-holiday flag | No |
| `is_work_day` | uint8 / SMALLINT | Work-day exception flag | No |
| `is_event` | uint8 / SMALLINT | Event flag | No |

## `forecast_readiness`

- **Source:** `reports/tables/forecast_readiness.csv`
- **Grain:** one row per `store_nbr + family` (54 × 33 = 1,782 rows).
- **Primary key:** composite `store_nbr + family` in the report artifact.
- **Foreign keys:** logical business-key references to `dim_store.store_nbr` and
  `dim_family.family`; this CSV is not currently a PostgreSQL warehouse table.
- **Important caveats:** readiness is rule-based data assessment, not model
  performance. Risk flags are independent and may overlap; `readiness_class` is a
  single priority label. Threshold columns are repeated to make classification
  reproducible. Series without positive sales can have null history/CV values.

| Column | Type | Meaning | Nullable |
|---|---|---|---|
| `store_nbr` | int64 | Store business key | No |
| `family` | string | Product-family business key | No |
| `city`, `state`, `store_type` | string | Denormalized store context | No |
| `cluster` | int64 | Store cluster | No |
| `history_start`, `history_end` | date string | First/last positive-sales date | Yes |
| `history_length` | int64 | Inclusive history span in days | No |
| `active_days` | int64 | Days with positive sales | No |
| `zero_sales_rate` | float64 | Share of observed periods with zero sales | No |
| `average_sales` | float64 | Mean sales volume over the assessed window | No |
| `sales_std` | float64 | Sales-volume standard deviation | No |
| `coefficient_of_variation` | float64 | `sales_std / average_sales`; undefined at zero mean | Yes |
| `promotion_rate` | float64 | Share of assessed rows with promotion | No |
| `observed_period_count` | int64 | Store-family dates observed in the source | No |
| `missing_period_count` | int64 | Calendar dates missing inside the assessed window | No |
| `has_positive_sales` | binary int | Whether the series ever has positive sales | No |
| `is_insufficient_history` | binary int | Independent insufficient-history risk flag | No |
| `is_intermittent` | binary int | Independent high-zero-rate risk flag | No |
| `is_promotion_dependent` | binary int | Independent high-promotion-rate risk flag | No |
| `is_high_volatility` | binary int | Independent high-CV risk flag | No |
| `is_ready` | binary int | Meets Ready rule and has no serious risk flag | No |
| `risk_flag_count` | int64 | Sum of four risk flags; excludes `is_ready` | No |
| `readiness_class` | string | Primary class selected by documented priority | No |
| `classification_rule` | string | Rule that produced the primary class | No |
| `zero_sales_rate_median` | float64 | Eligible-series median zero-sales threshold | No |
| `zero_sales_rate_q75` | float64 | Eligible-series Q75 zero-sales threshold | No |
| `cv_median` | float64 | Eligible-series median CV threshold | No |
| `cv_q75` | float64 | Eligible-series Q75 CV threshold | No |
| `promotion_rate_q75` | float64 | Eligible-series Q75 promotion threshold | No |
| `missing_period_count_q75` | float64 | Eligible-series Q75 missing-period threshold | No |

For the exact threshold values and primary-label priority, see
[Forecast readiness](../reports/forecast_readiness.md).
