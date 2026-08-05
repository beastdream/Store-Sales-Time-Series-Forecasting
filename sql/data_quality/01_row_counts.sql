SELECT
    'dim_date_row_count' AS check_name,
    'FAIL' AS severity,
    COUNT(*)::TEXT AS actual_value,
    '> 0' AS expected_value,
    COUNT(*) > 0 AS passed,
    'analytics.dim_date row count' AS details
FROM analytics.dim_date
UNION ALL
SELECT 'dim_store_row_count', 'FAIL', COUNT(*)::TEXT, '> 0', COUNT(*) > 0,
       'analytics.dim_store row count'
FROM analytics.dim_store
UNION ALL
SELECT 'dim_family_row_count', 'FAIL', COUNT(*)::TEXT, '> 0', COUNT(*) > 0,
       'analytics.dim_family row count'
FROM analytics.dim_family
UNION ALL
SELECT 'fact_daily_sales_row_count', 'FAIL', COUNT(*)::TEXT, '> 0', COUNT(*) > 0,
       'analytics.fact_daily_sales row count'
FROM analytics.fact_daily_sales
UNION ALL
SELECT 'fact_store_transactions_row_count', 'FAIL', COUNT(*)::TEXT, '> 0', COUNT(*) > 0,
       'analytics.fact_store_transactions row count'
FROM analytics.fact_store_transactions
UNION ALL
SELECT 'fact_oil_price_row_count', 'FAIL', COUNT(*)::TEXT, '> 0', COUNT(*) > 0,
       'analytics.fact_oil_price row count'
FROM analytics.fact_oil_price
UNION ALL
SELECT 'bridge_store_holiday_row_count', 'WARNING', COUNT(*)::TEXT, '> 0', COUNT(*) > 0,
       'analytics.bridge_store_holiday row count'
FROM analytics.bridge_store_holiday;
