# Warehouse Reconciliation

All validations and reconciliations passed before warehouse tables were saved.

## Table validation

| Table name | Row count | Expected grain | Duplicate grain count | Missing surrogate key count | Minimum date | Maximum date |
| --- | --- | --- | --- | --- | --- | --- |
| dim_date | 1704 | date_key | 0 | 0 | 2013-01-01 | 2017-08-31 |
| dim_store | 54 | store_key | 0 | 0 | N/A | N/A |
| dim_family | 33 | family_key | 0 | 0 | N/A | N/A |
| dim_store_date | 92016 | date_key + store_key | 0 | 0 | 2013-01-01 | 2017-08-31 |
| fact_daily_sales | 3000888 | date_key + store_key + family_key | 0 | 0 | 2013-01-01 | 2017-08-15 |
| fact_store_transactions | 83488 | date_key + store_key | 0 | 0 | 2013-01-01 | 2017-08-15 |
| fact_oil_price | 1704 | date_key | 0 | 0 | 2013-01-01 | 2017-08-31 |
| bridge_store_holiday | 7938 | date_key + store_key | 0 | 0 | 2013-01-01 | 2017-08-24 |

## Required reconciliations

| Check | Clean value | Warehouse value | Status |
| --- | --- | --- | --- |
| Total sales | 1073644952.2030685 | 1073644952.2030685 | PASS |
| Total onpromotion | 7810622 | 7810622 | PASS |
| Total transactions | 141478945 | 141478945 | PASS |
| Store count | 54 | 54 | PASS |
| Family count | 33 | 33 | PASS |
| Store-date row count | 92016 | 92016 | PASS |
| Store-date holiday mappings | 7938 | 7938 | PASS |
| Store-date sales observations | 90936 | 90936 | PASS |
| Store-date transaction observations | 83488 | 83488 | PASS |
| Sales fact unmapped date-store keys | 0 | 0 | PASS |
| Transaction fact unmapped date-store keys | 0 | 0 | PASS |
