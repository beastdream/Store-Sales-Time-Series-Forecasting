CREATE OR REPLACE VIEW mart.family_performance AS
WITH family_month_metrics AS (
    SELECT
        date_dim.year,
        date_dim.month,
        family_dim.family,
        SUM(fact.sales) AS total_sales,
        AVG(fact.sales) AS average_sales,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY fact.sales) AS median_sales,
        STDDEV_SAMP(fact.sales) AS sales_std,
        COUNT(*) FILTER (WHERE fact.sales = 0)::NUMERIC
            / NULLIF(COUNT(*), 0) AS zero_sales_rate,
        COUNT(*) FILTER (WHERE fact.is_promotion = 1)::NUMERIC
            / NULLIF(COUNT(*), 0) AS promotion_rate,
        AVG(fact.sales) FILTER (WHERE fact.is_promotion = 1) AS avg_sales_promo,
        AVG(fact.sales) FILTER (WHERE fact.is_promotion = 0) AS avg_sales_nonpromo,
        COUNT(*) FILTER (WHERE fact.is_promotion = 1) AS promo_observation_count,
        COUNT(*) FILTER (WHERE fact.is_promotion = 0) AS nonpromo_observation_count
    FROM analytics.fact_daily_sales AS fact
    JOIN analytics.dim_date AS date_dim
        ON date_dim.date_key = fact.date_key
    JOIN analytics.dim_family AS family_dim
        ON family_dim.family_key = fact.family_key
    GROUP BY date_dim.year, date_dim.month, family_dim.family
)
SELECT
    year,
    month,
    family,
    total_sales,
    average_sales,
    median_sales,
    sales_std,
    zero_sales_rate,
    promotion_rate,
    avg_sales_promo,
    avg_sales_nonpromo,
    promo_observation_count,
    nonpromo_observation_count,
    (avg_sales_promo - avg_sales_nonpromo)
        / NULLIF(avg_sales_nonpromo, 0) * 100 AS promotion_uplift_proxy_pct
FROM family_month_metrics;

COMMENT ON VIEW mart.family_performance IS
    'Grain: one row per year, month, and family; zero-sales observations are retained.';

COMMENT ON COLUMN mart.family_performance.promotion_uplift_proxy_pct IS
    'Descriptive promotion association proxy based on average sales; observation counts indicate reliability.';

WITH fact_month_family AS (
    SELECT
        date_dim.year,
        date_dim.month,
        family_dim.family,
        SUM(fact.sales) AS fact_total_sales
    FROM analytics.fact_daily_sales AS fact
    JOIN analytics.dim_date AS date_dim
        ON date_dim.date_key = fact.date_key
    JOIN analytics.dim_family AS family_dim
        ON family_dim.family_key = fact.family_key
    GROUP BY date_dim.year, date_dim.month, family_dim.family
),
mart_month_family AS (
    SELECT
        year,
        month,
        family,
        total_sales AS mart_total_sales
    FROM mart.family_performance
)
SELECT
    COALESCE(fact.year, mart.year) AS year,
    COALESCE(fact.month, mart.month) AS month,
    COALESCE(fact.family, mart.family) AS family,
    fact.fact_total_sales,
    mart.mart_total_sales,
    fact.family IS NOT NULL AND mart.family IS NOT NULL AS family_preserved,
    fact.fact_total_sales = mart.mart_total_sales AS total_sales_reconciled
FROM fact_month_family AS fact
FULL OUTER JOIN mart_month_family AS mart
    ON mart.year = fact.year
   AND mart.month = fact.month
   AND mart.family = fact.family
WHERE fact.family IS NULL
   OR mart.family IS NULL
   OR fact.fact_total_sales IS DISTINCT FROM mart.mart_total_sales
ORDER BY year, month, family;
