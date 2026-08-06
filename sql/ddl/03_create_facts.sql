CREATE TABLE IF NOT EXISTS analytics.fact_daily_sales (
    sales_id INTEGER PRIMARY KEY,
    date_key INTEGER NOT NULL,
    store_key INTEGER NOT NULL,
    date_store_key BIGINT NOT NULL,
    family_key INTEGER NOT NULL,
    sales NUMERIC(20, 7) NOT NULL CHECK (sales >= 0),
    onpromotion INTEGER NOT NULL CHECK (onpromotion >= 0),
    is_promotion SMALLINT NOT NULL CHECK (is_promotion IN (0, 1)),
    CONSTRAINT uq_fact_daily_sales_grain
        UNIQUE (date_key, store_key, family_key),
    CONSTRAINT fk_fact_daily_sales_date
        FOREIGN KEY (date_key) REFERENCES analytics.dim_date (date_key),
    CONSTRAINT fk_fact_daily_sales_store
        FOREIGN KEY (store_key) REFERENCES analytics.dim_store (store_key),
    CONSTRAINT fk_fact_daily_sales_store_date
        FOREIGN KEY (date_store_key)
        REFERENCES analytics.dim_store_date (date_store_key),
    CONSTRAINT fk_fact_daily_sales_family
        FOREIGN KEY (family_key) REFERENCES analytics.dim_family (family_key)
);

COMMENT ON TABLE analytics.fact_daily_sales IS
    'Grain: one row per date, store, and product family.';

CREATE TABLE IF NOT EXISTS analytics.fact_store_transactions (
    date_key INTEGER NOT NULL,
    store_key INTEGER NOT NULL,
    date_store_key BIGINT NOT NULL,
    transactions INTEGER NOT NULL CHECK (transactions >= 0),
    CONSTRAINT pk_fact_store_transactions
        PRIMARY KEY (date_key, store_key),
    CONSTRAINT fk_fact_store_transactions_date
        FOREIGN KEY (date_key) REFERENCES analytics.dim_date (date_key),
    CONSTRAINT fk_fact_store_transactions_store
        FOREIGN KEY (store_key) REFERENCES analytics.dim_store (store_key),
    CONSTRAINT fk_fact_store_transactions_store_date
        FOREIGN KEY (date_store_key)
        REFERENCES analytics.dim_store_date (date_store_key)
);

COMMENT ON TABLE analytics.fact_store_transactions IS
    'Grain: one row per date and store; transactions are not repeated by family.';

CREATE TABLE IF NOT EXISTS analytics.fact_oil_price (
    date_key INTEGER PRIMARY KEY,
    oil_price NUMERIC(10, 4) NOT NULL,
    oil_change_1d NUMERIC(10, 4),
    oil_change_7d NUMERIC(10, 4),
    oil_pct_change_7d NUMERIC(18, 8),
    oil_was_imputed SMALLINT NOT NULL CHECK (oil_was_imputed IN (0, 1)),
    CONSTRAINT fk_fact_oil_price_date
        FOREIGN KEY (date_key) REFERENCES analytics.dim_date (date_key)
);

COMMENT ON TABLE analytics.fact_oil_price IS
    'Grain: one row per calendar date in the analysis range.';

CREATE TABLE IF NOT EXISTS analytics.bridge_store_holiday (
    date_key INTEGER NOT NULL,
    store_key INTEGER NOT NULL,
    holiday_count SMALLINT NOT NULL CHECK (holiday_count >= 1),
    holiday_descriptions TEXT NOT NULL,
    holiday_types TEXT NOT NULL,
    holiday_locales TEXT NOT NULL,
    is_holiday SMALLINT NOT NULL CHECK (is_holiday IN (0, 1)),
    is_work_day SMALLINT NOT NULL CHECK (is_work_day IN (0, 1)),
    is_event SMALLINT NOT NULL CHECK (is_event IN (0, 1)),
    CONSTRAINT pk_bridge_store_holiday
        PRIMARY KEY (date_key, store_key),
    CONSTRAINT fk_bridge_store_holiday_date
        FOREIGN KEY (date_key) REFERENCES analytics.dim_date (date_key),
    CONSTRAINT fk_bridge_store_holiday_store
        FOREIGN KEY (store_key) REFERENCES analytics.dim_store (store_key)
);

COMMENT ON TABLE analytics.bridge_store_holiday IS
    'Grain: one aggregated holiday record per date and store.';

CREATE INDEX IF NOT EXISTS idx_fact_daily_sales_date_key
    ON analytics.fact_daily_sales (date_key);

CREATE INDEX IF NOT EXISTS idx_fact_daily_sales_store_key
    ON analytics.fact_daily_sales (store_key);

CREATE INDEX IF NOT EXISTS idx_fact_daily_sales_family_key
    ON analytics.fact_daily_sales (family_key);

CREATE INDEX IF NOT EXISTS idx_fact_daily_sales_date_store_key
    ON analytics.fact_daily_sales (date_store_key);

CREATE INDEX IF NOT EXISTS idx_fact_store_transactions_date_key
    ON analytics.fact_store_transactions (date_key);

CREATE INDEX IF NOT EXISTS idx_fact_store_transactions_store_key
    ON analytics.fact_store_transactions (store_key);

CREATE INDEX IF NOT EXISTS idx_fact_store_transactions_date_store_key
    ON analytics.fact_store_transactions (date_store_key);

CREATE INDEX IF NOT EXISTS idx_fact_oil_price_date_key
    ON analytics.fact_oil_price (date_key);

CREATE INDEX IF NOT EXISTS idx_bridge_store_holiday_date_key
    ON analytics.bridge_store_holiday (date_key);

CREATE INDEX IF NOT EXISTS idx_bridge_store_holiday_store_key
    ON analytics.bridge_store_holiday (store_key);
