CREATE OR REPLACE VIEW mart.daily_store_performance AS
WITH sales_by_store_day AS (
    SELECT
        date_key,
        store_key,
        SUM(sales) AS total_sales,
        SUM(CASE WHEN is_promotion = 1 THEN sales ELSE 0::NUMERIC END)
            AS promoted_sales,
        SUM(CASE WHEN is_promotion = 0 THEN sales ELSE 0::NUMERIC END)
            AS non_promoted_sales,
        SUM(onpromotion) AS promoted_item_count
    FROM analytics.fact_daily_sales
    GROUP BY date_key, store_key
)
SELECT
    date_dim.full_date,
    sales.date_key,
    store_dim.store_nbr,
    store_dim.city,
    store_dim.state,
    store_dim.store_type,
    store_dim.cluster,
    sales.total_sales,
    sales.promoted_sales,
    sales.non_promoted_sales,
    sales.promoted_item_count,
    sales.promoted_sales / NULLIF(sales.total_sales, 0)
        AS promotion_sales_share,
    transactions.transactions,
    sales.total_sales / NULLIF(transactions.transactions, 0)
        AS sales_volume_per_transaction
FROM sales_by_store_day AS sales
JOIN analytics.dim_date AS date_dim
    ON date_dim.date_key = sales.date_key
JOIN analytics.dim_store AS store_dim
    ON store_dim.store_key = sales.store_key
LEFT JOIN analytics.fact_store_transactions AS transactions
    ON transactions.date_key = sales.date_key
   AND transactions.store_key = sales.store_key;

COMMENT ON VIEW mart.daily_store_performance IS
    'Grain: one row per full_date and store_nbr. Sales is aggregated before transactions are joined.';

WITH fact_totals AS (
    SELECT
        SUM(sales) AS total_sales
    FROM analytics.fact_daily_sales
),
transaction_totals AS (
    SELECT
        SUM(transactions) AS total_transactions
    FROM analytics.fact_store_transactions
),
mart_totals AS (
    SELECT
        SUM(total_sales) AS total_sales,
        SUM(transactions) AS total_transactions
    FROM mart.daily_store_performance
),
duplicate_grain AS (
    SELECT COUNT(*) AS duplicate_combinations
    FROM (
        SELECT full_date, store_nbr
        FROM mart.daily_store_performance
        GROUP BY full_date, store_nbr
        HAVING COUNT(*) > 1
    ) AS duplicates
)
SELECT
    fact_totals.total_sales AS fact_total_sales,
    mart_totals.total_sales AS mart_total_sales,
    fact_totals.total_sales = mart_totals.total_sales AS sales_reconciled,
    transaction_totals.total_transactions AS fact_total_transactions,
    mart_totals.total_transactions AS mart_total_transactions,
    transaction_totals.total_transactions = mart_totals.total_transactions
        AS transactions_reconciled,
    duplicate_grain.duplicate_combinations,
    duplicate_grain.duplicate_combinations = 0 AS grain_is_valid
FROM fact_totals
CROSS JOIN transaction_totals
CROSS JOIN mart_totals
CROSS JOIN duplicate_grain;
