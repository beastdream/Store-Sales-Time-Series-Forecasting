# Raw Data Audit Summary

> This audit reports findings only; no raw data was cleaned or modified.

## Grain

- `train`: grain `date, store_nbr, family`; duplicate combinations = 0; affected rows = 0; valid = True.
- `test`: grain `date, store_nbr, family`; duplicate combinations = 0; affected rows = 0; valid = True.
- `stores`: grain `store_nbr`; duplicate combinations = 0; affected rows = 0; valid = True.
- `transactions`: grain `date, store_nbr`; duplicate combinations = 0; affected rows = 0; valid = True.
- `oil`: grain `date`; duplicate combinations = 0; affected rows = 0; valid = True.
- `sample_submission`: grain `id`; duplicate combinations = 0; affected rows = 0; valid = True.

## Time coverage

- `train`: 2013-01-01 to 2017-08-15.
- `test`: 2017-08-16 to 2017-08-31.

## Important missing values

- `train.sales`: 0 missing (0.00%).
- `train.onpromotion`: 0 missing (0.00%).
- `test.onpromotion`: 0 missing (0.00%).
- `stores.city`: 0 missing (0.00%).
- `stores.state`: 0 missing (0.00%).
- `stores.type`: 0 missing (0.00%).
- `stores.cluster`: 0 missing (0.00%).
- `transactions.transactions`: 0 missing (0.00%).
- `oil.dcoilwtico`: 43 missing (3.53%).

## Fully duplicated rows

- `train`: 0 fully duplicated rows.
- `test`: 0 fully duplicated rows.
- `stores`: 0 fully duplicated rows.
- `transactions`: 0 fully duplicated rows.
- `oil`: 0 fully duplicated rows.
- `holidays`: 0 fully duplicated rows.
- `sample_submission`: 0 fully duplicated rows.

## Unusual values and metadata checks

- `train_sales_negative`: 0 affected rows.
- `train_sales_zero`: 939130 affected rows.
- `train_onpromotion_negative`: 0 affected rows.
- `test_onpromotion_negative`: 0 affected rows.
- `missing_oil_price`: 43 affected rows.
- `stores_with_missing_metadata`: 0 affected rows.

## Issues to address during cleaning

- Decide and document an imputation policy for missing oil prices.
- Validate zero sales against closures, holidays, and genuine no-sale days before modeling.
