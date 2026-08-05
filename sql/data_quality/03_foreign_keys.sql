SELECT
    'fact_daily_sales_orphan_date' AS check_name,
    'FAIL' AS severity,
    COUNT(*)::TEXT AS actual_value,
    '0' AS expected_value,
    COUNT(*) = 0 AS passed,
    'Sales rows without a matching date dimension row' AS details
FROM analytics.fact_daily_sales AS fact
WHERE NOT EXISTS (
    SELECT 1 FROM analytics.dim_date AS dim WHERE dim.date_key = fact.date_key
)
UNION ALL
SELECT 'fact_daily_sales_orphan_store', 'FAIL', COUNT(*)::TEXT, '0', COUNT(*) = 0,
       'Sales rows without a matching store dimension row'
FROM analytics.fact_daily_sales AS fact
WHERE NOT EXISTS (
    SELECT 1 FROM analytics.dim_store AS dim WHERE dim.store_key = fact.store_key
)
UNION ALL
SELECT 'fact_daily_sales_orphan_family', 'FAIL', COUNT(*)::TEXT, '0', COUNT(*) = 0,
       'Sales rows without a matching family dimension row'
FROM analytics.fact_daily_sales AS fact
WHERE NOT EXISTS (
    SELECT 1 FROM analytics.dim_family AS dim WHERE dim.family_key = fact.family_key
)
UNION ALL
SELECT 'fact_store_transactions_orphan_date', 'FAIL', COUNT(*)::TEXT, '0', COUNT(*) = 0,
       'Transaction rows without a matching date dimension row'
FROM analytics.fact_store_transactions AS fact
WHERE NOT EXISTS (
    SELECT 1 FROM analytics.dim_date AS dim WHERE dim.date_key = fact.date_key
)
UNION ALL
SELECT 'fact_store_transactions_orphan_store', 'FAIL', COUNT(*)::TEXT, '0', COUNT(*) = 0,
       'Transaction rows without a matching store dimension row'
FROM analytics.fact_store_transactions AS fact
WHERE NOT EXISTS (
    SELECT 1 FROM analytics.dim_store AS dim WHERE dim.store_key = fact.store_key
)
UNION ALL
SELECT 'fact_oil_price_orphan_date', 'FAIL', COUNT(*)::TEXT, '0', COUNT(*) = 0,
       'Oil rows without a matching date dimension row'
FROM analytics.fact_oil_price AS fact
WHERE NOT EXISTS (
    SELECT 1 FROM analytics.dim_date AS dim WHERE dim.date_key = fact.date_key
)
UNION ALL
SELECT 'bridge_store_holiday_orphan_date', 'FAIL', COUNT(*)::TEXT, '0', COUNT(*) = 0,
       'Holiday bridge rows without a matching date dimension row'
FROM analytics.bridge_store_holiday AS bridge
WHERE NOT EXISTS (
    SELECT 1 FROM analytics.dim_date AS dim WHERE dim.date_key = bridge.date_key
)
UNION ALL
SELECT 'bridge_store_holiday_orphan_store', 'FAIL', COUNT(*)::TEXT, '0', COUNT(*) = 0,
       'Holiday bridge rows without a matching store dimension row'
FROM analytics.bridge_store_holiday AS bridge
WHERE NOT EXISTS (
    SELECT 1 FROM analytics.dim_store AS dim WHERE dim.store_key = bridge.store_key
)
UNION ALL
SELECT
    'warehouse_missing_date_key',
    'FAIL',
    COUNT(*)::TEXT,
    '0',
    COUNT(*) = 0,
    'Missing date_key values across facts and bridge'
FROM (
    SELECT date_key FROM analytics.fact_daily_sales
    UNION ALL
    SELECT date_key FROM analytics.fact_store_transactions
    UNION ALL
    SELECT date_key FROM analytics.fact_oil_price
    UNION ALL
    SELECT date_key FROM analytics.bridge_store_holiday
) AS warehouse_dates
WHERE date_key IS NULL;
