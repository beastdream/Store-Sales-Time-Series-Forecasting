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

CREATE INDEX IF NOT EXISTS idx_dim_date_full_date
    ON analytics.dim_date (full_date);

CREATE INDEX IF NOT EXISTS idx_dim_store_store_nbr
    ON analytics.dim_store (store_nbr);

CREATE INDEX IF NOT EXISTS idx_dim_family_family
    ON analytics.dim_family (family);
