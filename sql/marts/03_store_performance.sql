CREATE OR REPLACE VIEW mart.store_performance AS
WITH sales_by_store_day AS (
    SELECT
        date_key,
        store_key,
        SUM(sales) AS daily_sales
    FROM analytics.fact_daily_sales
    GROUP BY date_key, store_key
),
sales_by_store AS (
    SELECT
        store_key,
        SUM(daily_sales) AS total_sales,
        AVG(daily_sales) AS average_daily_sales,
        STDDEV_SAMP(daily_sales) AS sales_std,
        COUNT(*) AS active_days
    FROM sales_by_store_day
    GROUP BY store_key
),
transactions_by_store AS (
    SELECT
        store_key,
        SUM(transactions) AS total_transactions
    FROM analytics.fact_store_transactions
    GROUP BY store_key
)
SELECT
    store_dim.store_nbr,
    store_dim.city,
    store_dim.state,
    store_dim.store_type,
    store_dim.cluster,
    sales.total_sales,
    sales.average_daily_sales,
    sales.sales_std,
    sales.active_days,
    transactions.total_transactions,
    sales.total_sales / NULLIF(transactions.total_transactions, 0)
        AS sales_volume_per_transaction
FROM sales_by_store AS sales
JOIN analytics.dim_store AS store_dim
    ON store_dim.store_key = sales.store_key
LEFT JOIN transactions_by_store AS transactions
    ON transactions.store_key = sales.store_key;

COMMENT ON VIEW mart.store_performance IS
    'Grain: one row per store_nbr; sales is aggregated to store-day before store statistics.';

WITH mart_checks AS (
    SELECT
        COUNT(*) AS row_count,
        COUNT(*) - COUNT(DISTINCT store_nbr) AS duplicate_grain_count,
        SUM(total_sales) AS mart_total_sales,
        SUM(total_transactions) AS mart_total_transactions
    FROM mart.store_performance
),
fact_checks AS (
    SELECT
        (SELECT COUNT(*) FROM analytics.dim_store) AS expected_store_count,
        (SELECT SUM(sales) FROM analytics.fact_daily_sales) AS fact_total_sales,
        (SELECT SUM(transactions) FROM analytics.fact_store_transactions)
            AS fact_total_transactions
)
SELECT
    mart.row_count,
    fact.expected_store_count,
    mart.row_count = fact.expected_store_count AS row_count_reconciled,
    mart.duplicate_grain_count,
    mart.duplicate_grain_count = 0 AS grain_is_valid,
    fact.fact_total_sales,
    mart.mart_total_sales,
    fact.fact_total_sales = mart.mart_total_sales AS sales_reconciled,
    fact.fact_total_transactions,
    mart.mart_total_transactions,
    fact.fact_total_transactions = mart.mart_total_transactions
        AS transactions_reconciled
FROM mart_checks AS mart
CROSS JOIN fact_checks AS fact;
