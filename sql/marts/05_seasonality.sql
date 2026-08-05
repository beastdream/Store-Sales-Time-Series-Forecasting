CREATE OR REPLACE VIEW mart.seasonality AS
WITH sales_by_date AS (
    SELECT
        date_key,
        SUM(sales) AS daily_sales
    FROM analytics.fact_daily_sales
    GROUP BY date_key
),
transactions_by_date AS (
    SELECT
        date_key,
        SUM(transactions) AS daily_transactions
    FROM analytics.fact_store_transactions
    GROUP BY date_key
)
SELECT
    date_dim.year,
    date_dim.month,
    date_dim.day_of_week,
    date_dim.day_name,
    date_dim.is_weekend,
    date_dim.is_payday,
    SUM(sales.daily_sales) AS total_sales,
    AVG(sales.daily_sales) AS average_daily_sales,
    STDDEV_SAMP(sales.daily_sales) AS sales_std,
    COUNT(*) AS active_days,
    SUM(transactions.daily_transactions) AS total_transactions,
    SUM(sales.daily_sales)
        / NULLIF(SUM(transactions.daily_transactions), 0)
        AS sales_volume_per_transaction
FROM sales_by_date AS sales
JOIN analytics.dim_date AS date_dim
    ON date_dim.date_key = sales.date_key
LEFT JOIN transactions_by_date AS transactions
    ON transactions.date_key = sales.date_key
GROUP BY
    date_dim.year,
    date_dim.month,
    date_dim.day_of_week,
    date_dim.day_name,
    date_dim.is_weekend,
    date_dim.is_payday;

COMMENT ON VIEW mart.seasonality IS
    'Grain: one row per year, month, day_of_week, day_name, is_weekend, and is_payday combination.';

WITH expected_grain AS (
    SELECT COUNT(*) AS expected_row_count
    FROM (
        SELECT
            date_dim.year,
            date_dim.month,
            date_dim.day_of_week,
            date_dim.day_name,
            date_dim.is_weekend,
            date_dim.is_payday
        FROM analytics.fact_daily_sales AS fact
        JOIN analytics.dim_date AS date_dim
            ON date_dim.date_key = fact.date_key
        GROUP BY
            date_dim.year,
            date_dim.month,
            date_dim.day_of_week,
            date_dim.day_name,
            date_dim.is_weekend,
            date_dim.is_payday
    ) AS combinations
),
mart_checks AS (
    SELECT
        COUNT(*) AS mart_row_count,
        SUM(total_sales) AS mart_total_sales,
        SUM(total_transactions) AS mart_total_transactions
    FROM mart.seasonality
),
fact_checks AS (
    SELECT
        (SELECT SUM(sales) FROM analytics.fact_daily_sales) AS fact_total_sales,
        (SELECT SUM(transactions) FROM analytics.fact_store_transactions)
            AS fact_total_transactions
)
SELECT
    expected.expected_row_count,
    mart.mart_row_count,
    expected.expected_row_count = mart.mart_row_count AS row_count_reconciled,
    fact.fact_total_sales,
    mart.mart_total_sales,
    fact.fact_total_sales = mart.mart_total_sales AS sales_reconciled,
    fact.fact_total_transactions,
    mart.mart_total_transactions,
    fact.fact_total_transactions = mart.mart_total_transactions
        AS transactions_reconciled
FROM expected_grain AS expected
CROSS JOIN mart_checks AS mart
CROSS JOIN fact_checks AS fact;
