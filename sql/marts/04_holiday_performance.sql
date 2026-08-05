CREATE OR REPLACE VIEW mart.holiday_performance AS
WITH sales_by_store_day AS (
    SELECT
        date_key,
        store_key,
        SUM(sales) AS total_sales
    FROM analytics.fact_daily_sales
    GROUP BY date_key, store_key
),
transactions_by_store_day AS (
    SELECT
        date_key,
        store_key,
        SUM(transactions) AS transactions
    FROM analytics.fact_store_transactions
    GROUP BY date_key, store_key
)
SELECT
    date_dim.full_date,
    store_dim.store_nbr,
    holiday.is_holiday,
    holiday.is_work_day,
    holiday.is_event,
    holiday.holiday_count,
    holiday.holiday_types,
    sales.total_sales,
    transactions.transactions
FROM analytics.bridge_store_holiday AS holiday
JOIN analytics.dim_date AS date_dim
    ON date_dim.date_key = holiday.date_key
JOIN analytics.dim_store AS store_dim
    ON store_dim.store_key = holiday.store_key
LEFT JOIN sales_by_store_day AS sales
    ON sales.date_key = holiday.date_key
   AND sales.store_key = holiday.store_key
LEFT JOIN transactions_by_store_day AS transactions
    ON transactions.date_key = holiday.date_key
   AND transactions.store_key = holiday.store_key;

COMMENT ON VIEW mart.holiday_performance IS
    'Grain: one row per full_date and store_nbr present in the store-holiday bridge.';

WITH expected AS (
    SELECT
        COUNT(*) AS expected_row_count,
        SUM(sales.total_sales) AS expected_total_sales,
        SUM(transactions.transactions) AS expected_total_transactions
    FROM analytics.bridge_store_holiday AS holiday
    LEFT JOIN (
        SELECT date_key, store_key, SUM(sales) AS total_sales
        FROM analytics.fact_daily_sales
        GROUP BY date_key, store_key
    ) AS sales
        ON sales.date_key = holiday.date_key
       AND sales.store_key = holiday.store_key
    LEFT JOIN analytics.fact_store_transactions AS transactions
        ON transactions.date_key = holiday.date_key
       AND transactions.store_key = holiday.store_key
),
actual AS (
    SELECT
        COUNT(*) AS mart_row_count,
        SUM(total_sales) AS mart_total_sales,
        SUM(transactions) AS mart_total_transactions
    FROM mart.holiday_performance
),
duplicate_grain AS (
    SELECT COUNT(*) AS duplicate_combinations
    FROM (
        SELECT full_date, store_nbr
        FROM mart.holiday_performance
        GROUP BY full_date, store_nbr
        HAVING COUNT(*) > 1
    ) AS duplicates
)
SELECT
    expected.expected_row_count,
    actual.mart_row_count,
    expected.expected_row_count = actual.mart_row_count AS row_count_reconciled,
    duplicate_grain.duplicate_combinations,
    duplicate_grain.duplicate_combinations = 0 AS grain_is_valid,
    expected.expected_total_sales,
    actual.mart_total_sales,
    expected.expected_total_sales IS NOT DISTINCT FROM actual.mart_total_sales
        AS sales_reconciled,
    expected.expected_total_transactions,
    actual.mart_total_transactions,
    expected.expected_total_transactions IS NOT DISTINCT FROM
        actual.mart_total_transactions AS transactions_reconciled
FROM expected
CROSS JOIN actual
CROSS JOIN duplicate_grain;
