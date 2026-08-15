# End-to-End Project Validation

- Execution timestamp: `2026-08-15T20:00:21.412475+07:00`
- Overall status: **PASS**
- DA scope: PASS=18, WARNING=2, FAIL=0, NOT RUN=1
- DS scope: PASS=9, WARNING=0, FAIL=0, NOT RUN=0

## Scope ownership

- `src.validate_da_project`: raw/processed data, warehouse, EDA/report artifacts, repository hygiene, and local Power BI artifact existence.
- `src.validate_ds_project`: temporal splits, baselines, current modeling reports, chosen metadata, final model, and final submission contracts.

## Detailed evidence

- `reports/da_project_validation.md`
- `reports/ds_project_validation.md`

## Command to reproduce

```powershell
python -m src.validate_project
```
