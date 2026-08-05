# SQL Quality Report

- PASS: `46`
- WARNING: `0`
- FAIL: `0`

| SQL file | Check | Status | Actual | Expected | Details |
| --- | --- | --- | --- | --- | --- |
| 01_row_counts.sql | dim_date_row_count | PASS | 1704 | > 0 | analytics.dim_date row count |
| 01_row_counts.sql | dim_store_row_count | PASS | 54 | > 0 | analytics.dim_store row count |
| 01_row_counts.sql | dim_family_row_count | PASS | 33 | > 0 | analytics.dim_family row count |
| 01_row_counts.sql | fact_daily_sales_row_count | PASS | 100000 | > 0 | analytics.fact_daily_sales row count |
| 01_row_counts.sql | fact_store_transactions_row_count | PASS | 2538 | > 0 | analytics.fact_store_transactions row count |
| 01_row_counts.sql | fact_oil_price_row_count | PASS | 1704 | > 0 | analytics.fact_oil_price row count |
| 01_row_counts.sql | bridge_store_holiday_row_count | PASS | 270 | > 0 | analytics.bridge_store_holiday row count |
| 02_duplicate_grain.sql | dim_date_duplicate_grain | PASS | 0 | 0 | Duplicate date_key combinations |
| 02_duplicate_grain.sql | dim_store_duplicate_grain | PASS | 0 | 0 | Duplicate store_key combinations |
| 02_duplicate_grain.sql | dim_family_duplicate_grain | PASS | 0 | 0 | Duplicate family_key combinations |
| 02_duplicate_grain.sql | fact_daily_sales_duplicate_grain | PASS | 0 | 0 | Duplicate date_key + store_key + family_key combinations |
| 02_duplicate_grain.sql | fact_store_transactions_duplicate_grain | PASS | 0 | 0 | Duplicate date_key + store_key combinations |
| 02_duplicate_grain.sql | fact_oil_price_duplicate_grain | PASS | 0 | 0 | Duplicate date_key combinations |
| 02_duplicate_grain.sql | bridge_store_holiday_duplicate_grain | PASS | 0 | 0 | Duplicate date_key + store_key combinations |
| 03_foreign_keys.sql | warehouse_missing_date_key | PASS | 0 | 0 | Missing date_key values across facts and bridge |
| 03_foreign_keys.sql | bridge_store_holiday_orphan_store | PASS | 0 | 0 | Holiday bridge rows without a matching store dimension row |
| 03_foreign_keys.sql | bridge_store_holiday_orphan_date | PASS | 0 | 0 | Holiday bridge rows without a matching date dimension row |
| 03_foreign_keys.sql | fact_store_transactions_orphan_store | PASS | 0 | 0 | Transaction rows without a matching store dimension row |
| 03_foreign_keys.sql | fact_oil_price_orphan_date | PASS | 0 | 0 | Oil rows without a matching date dimension row |
| 03_foreign_keys.sql | fact_store_transactions_orphan_date | PASS | 0 | 0 | Transaction rows without a matching date dimension row |
| 03_foreign_keys.sql | fact_daily_sales_orphan_family | PASS | 0 | 0 | Sales rows without a matching family dimension row |
| 03_foreign_keys.sql | fact_daily_sales_orphan_store | PASS | 0 | 0 | Sales rows without a matching store dimension row |
| 03_foreign_keys.sql | fact_daily_sales_orphan_date | PASS | 0 | 0 | Sales rows without a matching date dimension row |
| 04_measure_validation.sql | family_count | PASS | 33 | > 0 | Number of families in dim_family |
| 04_measure_validation.sql | store_count | PASS | 54 | > 0 | Number of stores in dim_store |
| 04_measure_validation.sql | date_range | PASS | 2013-01-01 to 2017-08-31 | non-empty ordered range | Minimum and maximum date in dim_date |
| 04_measure_validation.sql | total_transactions | PASS | 4262484 | >= 0 | Warehouse total transactions |
| 04_measure_validation.sql | negative_transactions | PASS | 0 | 0 | Rows where transactions is negative |
| 04_measure_validation.sql | total_sales | PASS | 19194218.7818 | >= 0 | Warehouse total sales |
| 04_measure_validation.sql | negative_onpromotion | PASS | 0 | 0 | Rows where onpromotion is negative |
| 04_measure_validation.sql | negative_sales | PASS | 0 | 0 | Rows where sales is negative |
| 05_mart_reconciliation.sql | daily_store_sales_reconciliation | PASS | 19194218.7818 | 19194218.7818 | Total sales in daily store mart must equal the sales fact total |
| 05_mart_reconciliation.sql | daily_store_transactions_reconciliation | PASS | 4262484 | 4262484 | Total transactions in daily store mart must equal the transaction fact total |
| 05_mart_reconciliation.sql | family_sales_reconciliation | PASS | 19194218.7818 | 19194218.7818 | Total sales in family mart must equal the sales fact total |
| 05_mart_reconciliation.sql | store_count_reconciliation | PASS | 54 | 54 | Store mart row count must equal the store dimension count |
| 05_mart_reconciliation.sql | family_count_reconciliation | PASS | 33 | 33 | Distinct families in family mart must equal the family dimension count |
| 05_mart_reconciliation.sql | holiday_store_mapping | PASS | 0 | 0 | Holiday bridge keys and holiday mart store numbers must map to dim_store |
| 05_mart_reconciliation.sql | daily_store_mart_duplicate_grain | PASS | 0 | 0 | Duplicate full_date and store_nbr combinations in daily store mart |
| 05_mart_reconciliation.sql | family_mart_duplicate_grain | PASS | 0 | 0 | Duplicate year, month and family combinations in family mart |
| 05_mart_reconciliation.sql | store_mart_duplicate_grain | PASS | 0 | 0 | Duplicate store_nbr values in store mart |
| 05_mart_reconciliation.sql | holiday_mart_duplicate_grain | PASS | 0 | 0 | Duplicate full_date and store_nbr combinations in holiday mart |
| 05_mart_reconciliation.sql | seasonality_mart_duplicate_grain | PASS | 0 | 0 | Duplicate seasonality grain combinations in seasonality mart |
| 05_mart_reconciliation.sql | nonnegative_sales | PASS | 0 | 0 | Negative sales rows across facts and marts |
| 05_mart_reconciliation.sql | nonnegative_transactions | PASS | 0 | 0 | Negative transaction rows across facts and transaction-bearing marts |
| 05_mart_reconciliation.sql | finite_division_results | PASS | 0 | 0 | Infinite division results across all marts with calculated ratios |
| 05_mart_reconciliation.sql | mart_date_range_within_dimension | PASS | 0 | 0 | Date-bearing mart rows must map to a date or year-month in dim_date |
