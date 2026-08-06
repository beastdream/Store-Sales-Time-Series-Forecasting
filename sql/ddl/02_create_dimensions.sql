CREATE TABLE IF NOT EXISTS analytics.dim_date (
    date_key INTEGER PRIMARY KEY,
    full_date DATE NOT NULL,
    day SMALLINT NOT NULL CHECK (day BETWEEN 1 AND 31),
    day_of_week SMALLINT NOT NULL CHECK (day_of_week BETWEEN 0 AND 6),
    day_name VARCHAR(9) NOT NULL,
    week_of_year SMALLINT NOT NULL CHECK (week_of_year BETWEEN 1 AND 53),
    month SMALLINT NOT NULL CHECK (month BETWEEN 1 AND 12),
    month_name VARCHAR(9) NOT NULL,
    quarter SMALLINT NOT NULL CHECK (quarter BETWEEN 1 AND 4),
    year SMALLINT NOT NULL,
    is_weekend BOOLEAN NOT NULL,
    is_month_start BOOLEAN NOT NULL,
    is_month_end BOOLEAN NOT NULL,
    is_payday BOOLEAN NOT NULL,
    CONSTRAINT uq_dim_date_full_date UNIQUE (full_date)
);

COMMENT ON TABLE analytics.dim_date IS
    'Grain: one row per calendar date.';

CREATE TABLE IF NOT EXISTS analytics.dim_store (
    store_key INTEGER PRIMARY KEY,
    store_nbr SMALLINT NOT NULL,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(100) NOT NULL,
    store_type VARCHAR(20) NOT NULL,
    cluster SMALLINT NOT NULL,
    CONSTRAINT uq_dim_store_store_nbr UNIQUE (store_nbr)
);

COMMENT ON TABLE analytics.dim_store IS
    'Grain: one row per source store number.';

CREATE TABLE IF NOT EXISTS analytics.dim_family (
    family_key INTEGER PRIMARY KEY,
    family VARCHAR(100) NOT NULL,
    CONSTRAINT uq_dim_family_family UNIQUE (family)
);

COMMENT ON TABLE analytics.dim_family IS
    'Grain: one row per product family.';

CREATE TABLE IF NOT EXISTS analytics.dim_store_date (
    date_store_key BIGINT PRIMARY KEY,
    date_key INTEGER NOT NULL,
    store_key INTEGER NOT NULL,
    holiday_count INTEGER NOT NULL CHECK (holiday_count >= 0),
    holiday_descriptions TEXT NOT NULL,
    holiday_types TEXT NOT NULL,
    holiday_locales TEXT NOT NULL,
    is_holiday SMALLINT NOT NULL CHECK (is_holiday IN (0, 1)),
    is_work_day SMALLINT NOT NULL CHECK (is_work_day IN (0, 1)),
    is_event SMALLINT NOT NULL CHECK (is_event IN (0, 1)),
    has_sales_observation SMALLINT NOT NULL
        CHECK (has_sales_observation IN (0, 1)),
    has_transaction_observation SMALLINT NOT NULL
        CHECK (has_transaction_observation IN (0, 1)),
    CONSTRAINT uq_dim_store_date_grain UNIQUE (date_key, store_key),
    CONSTRAINT fk_dim_store_date_date
        FOREIGN KEY (date_key) REFERENCES analytics.dim_date (date_key),
    CONSTRAINT fk_dim_store_date_store
        FOREIGN KEY (store_key) REFERENCES analytics.dim_store (store_key)
);

COMMENT ON TABLE analytics.dim_store_date IS
    'Grain: one row per date and store across the complete analysis calendar.';

CREATE INDEX IF NOT EXISTS idx_dim_date_full_date
    ON analytics.dim_date (full_date);

CREATE INDEX IF NOT EXISTS idx_dim_store_store_nbr
    ON analytics.dim_store (store_nbr);

CREATE INDEX IF NOT EXISTS idx_dim_family_family
    ON analytics.dim_family (family);

CREATE INDEX IF NOT EXISTS idx_dim_store_date_date_key
    ON analytics.dim_store_date (date_key);

CREATE INDEX IF NOT EXISTS idx_dim_store_date_store_key
    ON analytics.dim_store_date (store_key);
