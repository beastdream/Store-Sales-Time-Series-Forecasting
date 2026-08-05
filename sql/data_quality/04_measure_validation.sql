SELECT
    'negative_sales' AS check_name,
    'FAIL' AS severity,
    COUNT(*)::TEXT AS actual_value,
    '0' AS expected_value,
    COUNT(*) = 0 AS passed,
    'Rows where sales is negative' AS details
FROM analytics.fact_daily_sales
WHERE sales < 0
UNION ALL
SELECT 'negative_onpromotion', 'FAIL', COUNT(*)::TEXT, '0', COUNT(*) = 0,
       'Rows where onpromotion is negative'
FROM analytics.fact_daily_sales
WHERE onpromotion < 0
UNION ALL
SELECT 'negative_transactions', 'FAIL', COUNT(*)::TEXT, '0', COUNT(*) = 0,
       'Rows where transactions is negative'
FROM analytics.fact_store_transactions
WHERE transactions < 0
UNION ALL
SELECT 'total_sales', 'WARNING', COALESCE(SUM(sales), 0)::TEXT, '>= 0',
       COALESCE(SUM(sales), 0) >= 0, 'Warehouse total sales'
FROM analytics.fact_daily_sales
UNION ALL
SELECT 'total_transactions', 'WARNING', COALESCE(SUM(transactions), 0)::TEXT, '>= 0',
       COALESCE(SUM(transactions), 0) >= 0, 'Warehouse total transactions'
FROM analytics.fact_store_transactions
UNION ALL
SELECT 'store_count', 'FAIL', COUNT(*)::TEXT, '> 0', COUNT(*) > 0,
       'Number of stores in dim_store'
FROM analytics.dim_store
UNION ALL
SELECT 'family_count', 'FAIL', COUNT(*)::TEXT, '> 0', COUNT(*) > 0,
       'Number of families in dim_family'
FROM analytics.dim_family
UNION ALL
SELECT
    'date_range',
    'FAIL',
    COALESCE(MIN(full_date)::TEXT, 'NULL') || ' to ' ||
        COALESCE(MAX(full_date)::TEXT, 'NULL'),
    'non-empty ordered range',
    MIN(full_date) IS NOT NULL AND MAX(full_date) >= MIN(full_date),
    'Minimum and maximum date in dim_date'
FROM analytics.dim_date;
