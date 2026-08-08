# Power BI Readiness Checklist

## Overall status

**READY WITH WARNINGS**

File-based warehouse và toàn bộ DA regression đã pass, không có lỗi dữ liệu/model
nghiêm trọng. Chưa đánh dấu `READY` tuyệt đối vì PostgreSQL runtime chưa chạy,
baseline SHA-256 cố định chưa tồn tại, và một số raw/large report CSV vẫn còn trong
Git index dù đã được `.gitignore` bao phủ.

Quy ước trạng thái:

- **READY:** mọi kiểm tra quan trọng pass và không còn warning cần xử lý trước khi
  giao/publish.
- **READY WITH WARNINGS:** model file-based có thể bắt đầu được dựng, nhưng còn
  warning vận hành/repository cần được ghi nhận.
- **NOT READY:** còn lỗi grain, key, reconciliation, artifact, test hoặc semantic
  model quan trọng.

## Regression execution

Các command sau đã chạy thành công theo thứ tự ngày 2026-08-06:

| Command | Result |
|---|---|
| `python -m src.data.run_cleaning` | PASS |
| `python -m src.data.run_warehouse_build` | PASS |
| `python -m pytest -v` | PASS — 135 tests |
| `python notebooks/04a_sales_trend_seasonality.py` | PASS |
| `python notebooks/04_business_eda.py` | PASS |
| `python notebooks/05_promotion_analysis.py` | PASS |
| `python notebooks/06_holiday_analysis.py` | PASS |
| `python notebooks/07_transactions_analysis.py` | PASS |
| `python notebooks/08_anomaly_review.py` | PASS |
| `python notebooks/09_forecast_readiness.py` | PASS |
| `python -m src.run_sql_quality_checks --dry-run` | PASS — 5 files/5 statements |
| `python -m src.validate_da_project` | PASS — 19 PASS, 2 WARNING, 0 FAIL, 1 NOT RUN |

Notebook 09 được chạy bổ sung vì đây là producer của forecast-readiness artifact;
không notebook DA downstream nào bị bỏ qua.

## Critical data checks

| Check | Status | Evidence |
|---|---|---|
| Cleaning chỉ ghi data artifacts vào `data/interim/` | PASS | `OUTPUT_PATHS` gồm đúng 6 interim Parquet; contract test pass |
| Warehouse chỉ ghi data artifacts vào `data/processed/` | PASS | `WAREHOUSE_PATHS` gồm đúng 8 processed Parquet |
| Sales dtype | PASS | Raw, clean và fact đều là `float64` |
| Sales reconciliation | PASS | Raw `1,073,644,952.2030684`; clean/fact `1,073,644,952.2030685`; khớp với tolerance `1e-6` |
| Transactions reconciliation | PASS | Clean và fact cùng `141,478,945` |
| Không double-count transactions | PASS | Clean/fact cùng 83,488 rows; grain date–store unique; fact không có `family_key` |
| Database module | PASS | `src.database` resolve tới `src/database.py` |
| `dim_store_date` full grid | PASS | 92,016 rows = 1,704 dates × 54 stores |
| `date_store_key` | PASS | Formula đúng, unique; mọi sales/transaction fact key đều map |
| Holiday slicer population | PASS | `is_holiday` chứa cả 0 và 1 |
| Promotion metric naming | PASS | Contract tests xác nhận tên non-causal mới và legacy names không còn |
| Sales-trend artifacts | PASS | Đủ 3 CSV và 5 PNG bắt buộc; artifacts đọc/giải mã được |
| Forecast readiness overlap | PASS | 1,782 series; 438 rows có ít nhất 2 risk flags |
| Store growth | PASS | Có proxy được đổi tên, `recent_90d_growth`, `recent_90d_yoy_growth`, và `has_yoy_comparison` |
| README/documentation | PASS | `README.md` và ba tài liệu trong `docs/` tồn tại; internal links đã kiểm tra |
| Business insights freshness | PASS | Dùng recent-growth window mới, correlation `0.837`, readiness `709/635/380/58`, và overlapping count 438 |
| Report/artifact validation | PASS | Validator đọc 14 Parquet, 19 CSV và giải mã 41 PNG |
| SQL quality dry-run | PASS | 5 quality SQL files, 5 statements; không yêu cầu database |
| PostgreSQL runtime | NOT RUN | Thiếu cấu hình kết nối; không bị ghi PASS hoặc FAIL giả |

## Git and reproducibility warnings

| Check | Status | Action |
|---|---|---|
| Raw hash không đổi trong regression | PASS | 7 raw CSV có SHA-256 giống nhau trước/sau validator |
| Baseline SHA-256 cố định | WARNING | Chưa có `raw_sha256.json`/`SHA256SUMS`; tạo baseline có review nếu muốn đối chiếu giữa máy/lần clone |
| Ignore rules | PASS | Raw CSV, ba large report CSV và generated Parquet đã được ignore; `.gitkeep` được giữ |
| Unwanted artifacts còn trong Git index | WARNING | 6 raw CSV và 3 large report CSV vẫn tracked; cần người dùng tự chạy `git rm --cached` trước commit |
| Secrets và absolute paths | PASS | Không phát hiện secret thật, `.env`, hoặc absolute personal path trong tracked text files |

Các lệnh untrack giữ nguyên file local:

```powershell
git rm --cached data/raw/*.csv
git rm --cached reports/tables/holiday_analysis.csv
git rm --cached reports/tables/promotion_analysis_matched.csv
git rm --cached reports/tables/transactions_analysis.csv
git status --short
```

Không lệnh nào ở trên được tự động chạy trong regression này.

## Tables recommended for Power BI import

Core semantic model:

1. `data/processed/dim_date.parquet` → `DimDate`
2. `data/processed/dim_store.parquet` → `DimStore`
3. `data/processed/dim_family.parquet` → `DimFamily`
4. `data/processed/dim_store_date.parquet` → `DimStoreDate`
5. `data/processed/fact_daily_sales.parquet` → `FactDailySales`
6. `data/processed/fact_store_transactions.parquet` → `FactStoreTransactions`
7. `data/processed/fact_oil_price.parquet` → `FactOilPrice`

Optional, with restricted roles:

- `data/processed/bridge_store_holiday.parquet` → audit/detail only; không dùng làm
  primary holiday slicer hoặc active fact-filter path.
- `reports/tables/forecast_readiness.csv` → trang readiness riêng; cần composite
  store-family key/bridge được kiểm định nếu nối vào model.

Không import raw CSV hoặc ba report CSV lớn tái tạo được vào semantic model chính.

## Relationship gate

- `DimDate 1 → * DimStoreDate` bằng `date_key`, single direction.
- `DimStore 1 → * DimStoreDate` bằng `store_key`, single direction.
- `DimStoreDate 1 → * FactDailySales` bằng `date_store_key`, single direction.
- `DimStoreDate 1 → * FactStoreTransactions` bằng `date_store_key`, single direction.
- `DimFamily 1 → * FactDailySales` bằng `family_key`, single direction.
- `DimDate 1 → * FactOilPrice` bằng `date_key`, single direction.
- Không tạo active direct path thứ hai từ `DimDate`/`DimStore` tới store-day facts.
- Không dùng many-to-many hoặc bidirectional filtering để xử lý holiday.

## Go/no-go decision

**READY WITH WARNINGS** cho bước thiết kế/import Power BI file-based.

Không có critical data failure ngoài PostgreSQL runtime chưa chạy. Trước khi commit
hoặc publish, cần xử lý Git-index warnings. Nếu nguồn Power BI sẽ là PostgreSQL thay
vì Parquet, trạng thái cho nguồn đó vẫn là **NOT READY** cho tới khi DDL, load, marts,
quality runtime và reconciliation thực tế đều pass.

Checklist này không tạo hoặc bắt đầu xây Power BI report.
