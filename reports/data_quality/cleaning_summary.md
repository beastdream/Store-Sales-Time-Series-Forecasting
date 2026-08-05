# Cleaning Summary

All critical grain and foreign-key validations passed before outputs were saved.

## Row counts and exact duplicates removed

| Table | Raw rows | Clean rows | Exact duplicates removed |
| --- | --- | --- | --- |
| train | 3000888 | 3000888 | 0 |
| test | 28512 | 28512 | 0 |
| stores | 54 | 54 | 0 |
| transactions | 83488 | 83488 | 0 |
| oil | 1218 | 1704 | 0 |
| holidays | 350 | 9139 | 0 |

## Oil-price interpolation

- Calendar range: `2013-01-01` to `2017-08-31`.
- Imputed daily prices: `529`.
- Remaining missing `oil_price`: `0`.

## Holiday store mapping

- Raw holiday rows: `350`.
- Transferred Holiday rows excluded: `12`.
- Daily store holiday rows created: `9139`.

## Grain checks

| Table | Grain | Duplicate combinations | Affected rows | Valid |
| --- | --- | --- | --- | --- |
| train | date, store_nbr, family | 0 | 0 | True |
| test | date, store_nbr, family | 0 | 0 | True |
| stores | store_nbr | 0 | 0 | True |
| transactions | date, store_nbr | 0 | 0 | True |
| oil | date | 0 | 0 | True |
| holidays | date, store_nbr | 0 | 0 | True |

## Foreign-key checks

- Invalid key values: `0`.
- Affected child rows: `0`.

## Remaining warnings

- `939130` train rows have zero sales; they were intentionally retained.
- `529` daily oil prices were imputed and are flagged by `oil_was_imputed`.
- Leading rows in oil change features remain missing where lag history is unavailable.
