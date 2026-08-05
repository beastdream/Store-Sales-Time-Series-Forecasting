SELECT
    'dim_date_duplicate_grain' AS check_name,
    'FAIL' AS severity,
    COUNT(*)::TEXT AS actual_value,
    '0' AS expected_value,
    COUNT(*) = 0 AS passed,
    'Duplicate date_key combinations' AS details
FROM (
    SELECT date_key FROM analytics.dim_date GROUP BY date_key HAVING COUNT(*) > 1
) AS duplicates
UNION ALL
SELECT 'dim_store_duplicate_grain', 'FAIL', COUNT(*)::TEXT, '0', COUNT(*) = 0,
       'Duplicate store_key combinations'
FROM (
    SELECT store_key FROM analytics.dim_store GROUP BY store_key HAVING COUNT(*) > 1
) AS duplicates
UNION ALL
SELECT 'dim_family_duplicate_grain', 'FAIL', COUNT(*)::TEXT, '0', COUNT(*) = 0,
       'Duplicate family_key combinations'
FROM (
    SELECT family_key FROM analytics.dim_family GROUP BY family_key HAVING COUNT(*) > 1
) AS duplicates
UNION ALL
SELECT 'fact_daily_sales_duplicate_grain', 'FAIL', COUNT(*)::TEXT, '0', COUNT(*) = 0,
       'Duplicate date_key + store_key + family_key combinations'
FROM (
    SELECT date_key, store_key, family_key
    FROM analytics.fact_daily_sales
    GROUP BY date_key, store_key, family_key
    HAVING COUNT(*) > 1
) AS duplicates
UNION ALL
SELECT 'fact_store_transactions_duplicate_grain', 'FAIL', COUNT(*)::TEXT, '0', COUNT(*) = 0,
       'Duplicate date_key + store_key combinations'
FROM (
    SELECT date_key, store_key
    FROM analytics.fact_store_transactions
    GROUP BY date_key, store_key
    HAVING COUNT(*) > 1
) AS duplicates
UNION ALL
SELECT 'fact_oil_price_duplicate_grain', 'FAIL', COUNT(*)::TEXT, '0', COUNT(*) = 0,
       'Duplicate date_key combinations'
FROM (
    SELECT date_key FROM analytics.fact_oil_price GROUP BY date_key HAVING COUNT(*) > 1
) AS duplicates
UNION ALL
SELECT 'bridge_store_holiday_duplicate_grain', 'FAIL', COUNT(*)::TEXT, '0', COUNT(*) = 0,
       'Duplicate date_key + store_key combinations'
FROM (
    SELECT date_key, store_key
    FROM analytics.bridge_store_holiday
    GROUP BY date_key, store_key
    HAVING COUNT(*) > 1
) AS duplicates;
