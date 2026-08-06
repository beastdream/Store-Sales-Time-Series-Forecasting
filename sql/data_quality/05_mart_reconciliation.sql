WITH metrics AS (
    SELECT
        (SELECT COALESCE(SUM(sales), 0) FROM analytics.fact_daily_sales)
            AS fact_sales,
        (SELECT COALESCE(SUM(total_sales), 0)
         FROM mart.daily_store_performance) AS daily_store_sales,
        (SELECT COALESCE(SUM(transactions), 0)
         FROM analytics.fact_store_transactions) AS fact_transactions,
        (SELECT COALESCE(SUM(transactions), 0)
         FROM mart.daily_store_performance) AS daily_store_transactions,
        (SELECT COALESCE(SUM(total_sales), 0)
         FROM mart.family_performance) AS family_sales,
        (SELECT COUNT(*) FROM analytics.dim_store) AS dimension_store_count,
        (SELECT COUNT(*) FROM mart.store_performance) AS mart_store_count,
        (SELECT COUNT(*) FROM analytics.dim_family) AS dimension_family_count,
        (SELECT COUNT(DISTINCT family) FROM mart.family_performance)
            AS mart_family_count
),
violations AS (
    SELECT
        (
            SELECT COUNT(*)
            FROM analytics.bridge_store_holiday AS holiday
            LEFT JOIN analytics.dim_store AS store_dim
                ON store_dim.store_key = holiday.store_key
            WHERE store_dim.store_key IS NULL
        ) + (
            SELECT COUNT(*)
            FROM mart.holiday_performance AS holiday
            LEFT JOIN analytics.dim_store AS store_dim
                ON store_dim.store_nbr = holiday.store_nbr
            WHERE store_dim.store_key IS NULL
        ) AS invalid_holiday_store_count,
        (SELECT COUNT(*) FROM (
            SELECT full_date, store_nbr
            FROM mart.daily_store_performance
            GROUP BY full_date, store_nbr HAVING COUNT(*) > 1
        ) AS duplicate_grain) AS daily_store_duplicate_grain_count,
        (SELECT COUNT(*) FROM (
            SELECT year, month, family
            FROM mart.family_performance
            GROUP BY year, month, family HAVING COUNT(*) > 1
        ) AS duplicate_grain) AS family_duplicate_grain_count,
        (SELECT COUNT(*) FROM (
            SELECT store_nbr
            FROM mart.store_performance
            GROUP BY store_nbr HAVING COUNT(*) > 1
        ) AS duplicate_grain) AS store_duplicate_grain_count,
        (SELECT COUNT(*) FROM (
            SELECT full_date, store_nbr
            FROM mart.holiday_performance
            GROUP BY full_date, store_nbr HAVING COUNT(*) > 1
        ) AS duplicate_grain) AS holiday_duplicate_grain_count,
        (SELECT COUNT(*) FROM (
            SELECT year, month, day_of_week, day_name, is_weekend, is_payday
            FROM mart.seasonality
            GROUP BY year, month, day_of_week, day_name, is_weekend, is_payday
            HAVING COUNT(*) > 1
        ) AS duplicate_grain) AS seasonality_duplicate_grain_count,
        (SELECT COUNT(*) FROM analytics.fact_daily_sales WHERE sales < 0)
            + (SELECT COUNT(*) FROM mart.daily_store_performance
               WHERE total_sales < 0)
            + (SELECT COUNT(*) FROM mart.family_performance
               WHERE total_sales < 0)
            + (SELECT COUNT(*) FROM mart.store_performance
               WHERE total_sales < 0)
            + (SELECT COUNT(*) FROM mart.holiday_performance
               WHERE total_sales < 0)
            + (SELECT COUNT(*) FROM mart.seasonality
               WHERE total_sales < 0) AS negative_sales_count,
        (SELECT COUNT(*) FROM analytics.fact_store_transactions
         WHERE transactions < 0)
            + (SELECT COUNT(*) FROM mart.daily_store_performance
               WHERE transactions < 0)
            + (SELECT COUNT(*) FROM mart.store_performance
               WHERE total_transactions < 0)
            + (SELECT COUNT(*) FROM mart.holiday_performance
               WHERE transactions < 0)
            + (SELECT COUNT(*) FROM mart.seasonality
               WHERE total_transactions < 0) AS negative_transactions_count,
        (SELECT COUNT(*) FROM mart.daily_store_performance
         WHERE promotion_active_sales_share_proxy::TEXT IN ('Infinity', '-Infinity')
            OR sales_volume_per_transaction::TEXT IN ('Infinity', '-Infinity'))
            + (SELECT COUNT(*) FROM mart.family_performance
               WHERE promotion_uplift_proxy_pct::TEXT IN ('Infinity', '-Infinity'))
            + (SELECT COUNT(*) FROM mart.store_performance
               WHERE sales_volume_per_transaction::TEXT IN ('Infinity', '-Infinity'))
            + (SELECT COUNT(*) FROM mart.seasonality
               WHERE sales_volume_per_transaction::TEXT IN ('Infinity', '-Infinity'))
            AS infinite_division_result_count,
        (SELECT COUNT(*) FROM mart.daily_store_performance AS mart_row
         WHERE NOT EXISTS (
             SELECT 1 FROM analytics.dim_date AS date_dim
             WHERE date_dim.full_date = mart_row.full_date
         ))
            + (SELECT COUNT(*) FROM mart.holiday_performance AS mart_row
               WHERE NOT EXISTS (
                   SELECT 1 FROM analytics.dim_date AS date_dim
                   WHERE date_dim.full_date = mart_row.full_date
               ))
            + (SELECT COUNT(*) FROM mart.family_performance AS mart_row
               WHERE NOT EXISTS (
                   SELECT 1 FROM analytics.dim_date AS date_dim
                   WHERE date_dim.year = mart_row.year
                     AND date_dim.month = mart_row.month
               ))
            + (SELECT COUNT(*) FROM mart.seasonality AS mart_row
               WHERE NOT EXISTS (
                   SELECT 1 FROM analytics.dim_date AS date_dim
                   WHERE date_dim.year = mart_row.year
                     AND date_dim.month = mart_row.month
               )) AS mart_date_outside_dimension_count
    FROM metrics
)
SELECT 'daily_store_sales_reconciliation' AS check_name, 'FAIL' AS severity,
       daily_store_sales::TEXT AS actual_value, fact_sales::TEXT AS expected_value,
       daily_store_sales = fact_sales AS passed,
       'Total sales in daily store mart must equal the sales fact total' AS details
FROM metrics
UNION ALL
SELECT 'daily_store_transactions_reconciliation', 'FAIL',
       daily_store_transactions::TEXT, fact_transactions::TEXT,
       daily_store_transactions = fact_transactions,
       'Total transactions in daily store mart must equal the transaction fact total'
FROM metrics
UNION ALL
SELECT 'family_sales_reconciliation', 'FAIL', family_sales::TEXT,
       fact_sales::TEXT, family_sales = fact_sales,
       'Total sales in family mart must equal the sales fact total'
FROM metrics
UNION ALL
SELECT 'store_count_reconciliation', 'FAIL', mart_store_count::TEXT,
       dimension_store_count::TEXT, mart_store_count = dimension_store_count,
       'Store mart row count must equal the store dimension count'
FROM metrics
UNION ALL
SELECT 'family_count_reconciliation', 'FAIL', mart_family_count::TEXT,
       dimension_family_count::TEXT, mart_family_count = dimension_family_count,
       'Distinct families in family mart must equal the family dimension count'
FROM metrics
UNION ALL
SELECT 'holiday_store_mapping', 'FAIL', invalid_holiday_store_count::TEXT, '0',
       invalid_holiday_store_count = 0,
       'Holiday bridge keys and holiday mart store numbers must map to dim_store'
FROM violations
UNION ALL
SELECT 'daily_store_mart_duplicate_grain', 'FAIL',
       daily_store_duplicate_grain_count::TEXT, '0',
       daily_store_duplicate_grain_count = 0,
       'Duplicate full_date and store_nbr combinations in daily store mart'
FROM violations
UNION ALL
SELECT 'family_mart_duplicate_grain', 'FAIL',
       family_duplicate_grain_count::TEXT, '0', family_duplicate_grain_count = 0,
       'Duplicate year, month and family combinations in family mart'
FROM violations
UNION ALL
SELECT 'store_mart_duplicate_grain', 'FAIL', store_duplicate_grain_count::TEXT,
       '0', store_duplicate_grain_count = 0,
       'Duplicate store_nbr values in store mart'
FROM violations
UNION ALL
SELECT 'holiday_mart_duplicate_grain', 'FAIL',
       holiday_duplicate_grain_count::TEXT, '0', holiday_duplicate_grain_count = 0,
       'Duplicate full_date and store_nbr combinations in holiday mart'
FROM violations
UNION ALL
SELECT 'seasonality_mart_duplicate_grain', 'FAIL',
       seasonality_duplicate_grain_count::TEXT, '0',
       seasonality_duplicate_grain_count = 0,
       'Duplicate seasonality grain combinations in seasonality mart'
FROM violations
UNION ALL
SELECT 'nonnegative_sales', 'FAIL', negative_sales_count::TEXT, '0',
       negative_sales_count = 0, 'Negative sales rows across facts and marts'
FROM violations
UNION ALL
SELECT 'nonnegative_transactions', 'FAIL', negative_transactions_count::TEXT, '0',
       negative_transactions_count = 0,
       'Negative transaction rows across facts and transaction-bearing marts'
FROM violations
UNION ALL
SELECT 'finite_division_results', 'FAIL', infinite_division_result_count::TEXT,
       '0', infinite_division_result_count = 0,
       'Infinite division results across all marts with calculated ratios'
FROM violations
UNION ALL
SELECT 'mart_date_range_within_dimension', 'FAIL',
       mart_date_outside_dimension_count::TEXT, '0',
       mart_date_outside_dimension_count = 0,
       'Date-bearing mart rows must map to a date or year-month in dim_date'
FROM violations;
