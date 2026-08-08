# DA Project Validation

## Execution timestamp

2026-08-08T21:10:20.5871829+07:00

## Environment

- Python: `3.11.9`
- Platform: `Windows-10-10.0.26200-SP0`
- Working directory: repository root (relative paths only in report)
- PostgreSQL configured: `no`

## PASS

- **Pytest suite:** `python -m pytest -q` returned exit code 0 on 2026-08-08. Last output: 135 passed in 6.83s
- **Cleaning pipeline:** `python -m src.data.run_cleaning` returned exit code 0. Last output: .\reports\data_quality\cleaning_summary.md
- **Warehouse pipeline:** `python -m src.data.run_warehouse_build` returned exit code 0. Last output: .\reports\data_quality\warehouse_reconciliation.md
- **Read all required Parquet artifacts:** Read 14 Parquet files successfully.
- **Processed row count and grain:** All eight processed tables have non-null, unique expected grains.
- **Sales reconciliation:** Interim, fact, store report, and family report totals compared with atol=1e-6.
- **Transactions reconciliation:** Interim, store-day fact, and store-level report totals compared exactly.
- **Transactions are not double-counted by family:** Interim rows=83,488; fact rows=83,488; family_key absent=True.
- **dim_store_date completeness:** Rows=92,016; expected date × store rows=92,016.
- **date_store_key contract:** Validated formula, uniqueness, and fact foreign-key coverage.
- **Holiday store mapping:** Validated 7,938 unique mapped store-day records against dim_store_date.
- **Forecast readiness flags:** Validated 1,782 store-family rows, binary flags, risk counts, and Ready rule.
- **Report CSV artifacts:** Read 19 report CSV files successfully.
- **PNG artifacts:** Decoded 41 valid, non-empty PNG files.
- **Raw SHA-256 unchanged during validation:** Compared 7 raw CSV files before and after pipeline execution.
- **Secret scan:** No real .env, private key, common token, or credential-bearing database URL found.
- **Absolute personal path scan:** No absolute personal-machine path found in tracked text files.
- **Git hygiene rules:** Required ignore rules exist; .gitkeep is retained; no Parquet is tracked.
- **SQL quality dry-run:** `python -m src.run_sql_quality_checks --dry-run` returned exit code 0. Last output: SQL quality dry-run passed: 5 files, 5 statements
- **Power BI dashboard:** `powerbi/store_sales_analytics.pbix` exists and its
  read-only layout metadata confirms eight analytical pages. Page 8 is `Forecast
  Readiness & Anomalies` and contains the `Forecast Readiness` and `Sales Anomalies`
  bookmarks with inverse `FR_Group`/`AN_Group` visibility.
- **Completed DA scope:** store/family performance, trend/seasonality, promotion,
  holiday/event, transaction/oil, forecast-readiness, and anomaly outputs are
  present in the report and dashboard layers.

## KNOWN LIMITATIONS

- **Raw SHA-256 baseline:** No baseline hash file exists; pre/post hashes were still compared.
- **Ignored artifacts still tracked:** Raw CSV tracked=6; large reproducible report CSV tracked=3. Run documented git rm --cached commands manually.
- **Sales semantics:** `sales` is Sales Volume, not revenue or profit; price, cost,
  margin, inventory, stockout, and lead-time data are unavailable.
- **Causality:** Promotion and holiday/event comparisons are descriptive
  associations or proxy differences, not causal uplift estimates.
- **Partial year:** Historical actual sales end on 2017-08-15, so 2017 is not a
  complete actual-sales year even though the date dimension extends through the
  2017-08-31 test horizon.
- **Power BI operations:** The local PBIX is complete, but Power BI Service
  publication, gateway, scheduled refresh, credentials, and production access are
  not verified by repository evidence.

## FAIL

- None.

## NOT RUN

- **PostgreSQL runtime:** PostgreSQL runtime was not requested; missing configuration: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD.

## Data reconciliation

| Measure/check | Source | Value | Status |
|---|---|---:|---|
| dim_date row count | `data\processed\dim_date.parquet` | 1704 | PASS |
| dim_store row count | `data\processed\dim_store.parquet` | 54 | PASS |
| dim_family row count | `data\processed\dim_family.parquet` | 33 | PASS |
| dim_store_date row count | `data\processed\dim_store_date.parquet` | 92016 | PASS |
| fact_daily_sales row count | `data\processed\fact_daily_sales.parquet` | 3000888 | PASS |
| fact_store_transactions row count | `data\processed\fact_store_transactions.parquet` | 83488 | PASS |
| fact_oil_price row count | `data\processed\fact_oil_price.parquet` | 1704 | PASS |
| bridge_store_holiday row count | `data\processed\bridge_store_holiday.parquet` | 7938 | PASS |
| Sales volume | `interim train` | 1073644952.2030685 | PASS |
| Sales volume | `sales fact` | 1073644952.2030685 | PASS |
| Sales volume | `store report` | 1073644952.2030684 | PASS |
| Sales volume | `family report` | 1073644952.2030684 | PASS |
| Transactions | `interim transactions` | 141478945 | PASS |
| Transactions | `transaction fact` | 141478945 | PASS |
| Transactions | `store report` | 141478945 | PASS |
| Transactions | `transaction store report` | 141478945 | PASS |

## Artifacts generated

- Generated by cleaning: 6 interim Parquet files
- Generated by warehouse build: 8 processed Parquet files
- Validated existing artifacts: 19 report CSV files
- Validated existing artifacts: 41 report PNG files
- Generated by validator: reports\da_project_validation.md

## Git hygiene

- Tracked files scanned: 169.
- Raw CSV files still tracked: 6.
- Large reproducible report CSV files still tracked: 3.
- Tracked Parquet files: 0.
- No files were deleted, untracked, or committed by the validator.

## Power BI readiness

- Processed model tables and date_store_key relationships are file-validated.
- The single-direction `DimStoreDate` filter architecture is documented.
- The local Power BI dashboard is complete and contains eight pages.
- Forecast readiness and sales anomaly analysis are both present on Page 8 with
  bookmark navigation.
- Power BI Service runtime/publication/refresh is not claimed or validated.

## FUTURE WORK

- Configure and validate PostgreSQL DDL, load, marts, and runtime SQL quality checks.
- Remove ignored raw/report artifacts from the Git index manually if still tracked.
- If deployment is required, validate Power BI Service publication, refresh,
  gateway, credentials, access, and production reconciliation.
- Begin the separate Data Science phase with 16-day temporal backtests, baselines,
  leakage-safe feature engineering, model evaluation, and final prediction
  generation. No forecast model or approved accuracy result exists yet.

## Commands to reproduce

```powershell
python -m src.validate_da_project
```

The validator itself runs `pytest`, cleaning, warehouse build, SQL quality dry-run,
and all file/artifact checks. It never runs PostgreSQL runtime, deletes files,
changes business rules, modifies the Git index, or creates a commit.
