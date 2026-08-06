# DA Project Validation

Ngày kiểm định: 2026-08-06 (Asia/Bangkok)

## Passed checks

- Cấu trúc dự án đầy đủ: toàn bộ thư mục yêu cầu trong `data/`, `notebooks/`, `src/`, `sql/`, `reports/`, `models/`, `tests/` và `powerbi/` đều tồn tại; bốn file `__init__.py` yêu cầu đều có mặt.
- Test suite: `94 passed` với `pytest -q`.
- Cleaning pipeline: `python -m src.data.run_cleaning` hoàn tất thành công và tạo lại 6 file Parquet ở `data/interim/` cùng các đầu ra làm sạch liên quan.
- Warehouse build: `python -m src.data.run_warehouse_build` hoàn tất thành công và tạo lại 7 file Parquet ở `data/processed/`.
- Cả 13 file Parquet đều đọc được. Không có giá trị thiếu ngoài 15 giá trị lag được kỳ vọng trong mỗi bảng dầu.
- Grain của `fact_daily_sales`, `fact_store_transactions` và bridge ngày-cửa hàng là duy nhất; không phát hiện nhân bản giao dịch theo `family`.
- Khóa ngày và cửa hàng của fact/bridge ánh xạ được sang dimension; không phát hiện orphan key trong các kiểm tra file.
- Tổng sales và transactions được đối soát thành công giữa dữ liệu sạch, fact và các report/mart dạng file (chi tiết ở phần Data reconciliation).
- Holiday được ánh xạ ở cấp cửa hàng: 7,938 dòng bridge có ánh xạ holiday; grain `date_key + store_key` là duy nhất. Có 239 trường hợp nhiều sự kiện được tổng hợp mà không nhân bản grain.
- Chạy tuần tự 9 notebook script từ `01_data_audit.py` đến `09_forecast_readiness.py`: 9/9 hoàn tất với exit code 0.
- Kiểm tra artifact: 16 CSV đều đọc được, 35 PNG đều giải mã được và không rỗng, 13 báo cáo Markdown hiện hữu trước báo cáo này đều không rỗng.
- `reports/business_insights.md` có 13 insight, mỗi insight đủ finding, evidence, business implication, recommended action và limitation; không có placeholder hoặc dùng sai khái niệm revenue. Các số liệu đại diện được tính lại trực tiếp từ report tables và khớp.
- Kiểm tra tĩnh SQL: 13/13 file SQL không rỗng và parse được.
- Không phát hiện password/credential thực trong các file được Git theo dõi. `DB_PASSWORD=change_me` chỉ xuất hiện trong `.env.example` và được nhận diện là placeholder.
- Không phát hiện absolute filesystem path trong source, notebook, SQL hoặc report được Git theo dõi.
- SHA-256 của 8 file trong `data/raw/` giống hệt trước và sau toàn bộ quy trình; dữ liệu raw không bị thay đổi.

## Failed checks

- SQL data-quality runtime không chạy được: `python -m src.run_sql_quality_checks` trả exit code 1 và báo `0 PASS, 0 WARNING, 1 FAIL`. Nguyên nhân là không có kết nối PostgreSQL vì thiếu `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`; file `.env` cũng không tồn tại. Vì vậy chưa thể xác nhận DDL, dữ liệu đã load, các mart SQL và quality queries trên PostgreSQL thực tế.

Không phát hiện lỗi logic nghiệp vụ nào khác trong phạm vi kiểm định file-based. Lỗi kết nối SQL trên không được tự động sửa hoặc che giấu.

## Warnings

- Bước load PostgreSQL được bỏ qua đúng theo điều kiện “nếu có connection”; không có database khả dụng trong môi trường kiểm định.
- `reports/data_quality/sql_quality_report.md` đã được cập nhật bởi lần chạy thất bại và hiện phản ánh lỗi kết nối database.
- Hai bảng dầu có cùng 15 giá trị thiếu do đặc trưng lag: `oil_change_1d` thiếu 1, `oil_change_7d` thiếu 7 và `oil_pct_change_7d` thiếu 7. Đây là biên đầu chuỗi thời gian, không phải lỗi đọc Parquet.
- Đối soát mart hiện mới được xác nhận qua các report tables được sinh từ pipeline/notebook. Chưa thể đối soát trực tiếp mart trong PostgreSQL.
- Phạm vi quét secret là các file được Git theo dõi; `.env` không tồn tại. Các file untracked cần tiếp tục được kiểm tra trước khi commit.

## Files generated

- 6 file Parquet trong `data/interim/`: `holiday_store_daily`, `oil_clean`, `stores_clean`, `test_clean`, `train_clean`, `transactions_clean`.
- 7 file Parquet trong `data/processed/`: `bridge_date_store_holiday`, `dim_date`, `dim_family`, `dim_store`, `fact_daily_sales`, `fact_oil_price`, `fact_store_transactions`.
- 16 CSV trong `reports/data_quality/` và `reports/tables/`, bao gồm audit khóa/grain, performance theo store/family, promotion, holiday, transactions, anomaly và forecast readiness.
- 35 biểu đồ PNG trong `reports/figures/`, bao phủ cleaning, EDA, promotion, holiday và transactions.
- Các báo cáo Markdown được làm mới bởi pipeline/notebook, gồm cleaning summary, warehouse reconciliation, data audit, EDA, promotion, holiday, transactions, anomaly, forecast readiness và business insights.
- Báo cáo kiểm định này: `reports/da_project_validation.md`.

## Data reconciliation

| Chỉ tiêu | Nguồn | Giá trị | Kết quả |
|---|---|---:|---|
| Sales | `data/interim/train_clean.parquet` | 1,073,644,952.1824812 | Khớp |
| Sales | `data/processed/fact_daily_sales.parquet` | 1,073,644,952.1824812 | Khớp |
| Sales | `reports/tables/store_performance.csv` | 1,073,644,952.1824810 | Khớp trong sai số floating point |
| Sales | `reports/tables/family_performance.csv` | 1,073,644,952.1824812 | Khớp |
| Sales | `reports/tables/holiday_analysis.csv` | 1,073,644,952.1824812 | Khớp |
| Transactions | `data/interim/transactions_clean.parquet` | 141,478,945 | Khớp |
| Transactions | `data/processed/fact_store_transactions.parquet` | 141,478,945 | Khớp |
| Transactions | `reports/tables/store_performance.csv` | 141,478,945 | Khớp |
| Transactions | `reports/tables/transactions_analysis.csv` | 141,478,945 | Khớp |
| Transactions | `reports/tables/transactions_store_summary.csv` | 141,478,945 | Khớp |

`fact_store_transactions` và `transactions_analysis` đều có 83,488 dòng ở grain ngày-cửa hàng. Tổng transactions không bị lặp theo 33 family. Đối soát sales dùng so sánh gần đúng để xử lý sai số biểu diễn số thực; không có chênh lệch nghiệp vụ.

## Commands to reproduce

Chạy từ thư mục gốc dự án:

```powershell
pytest -q
python -m src.data.run_cleaning
python -m src.data.run_warehouse_build

Get-ChildItem notebooks\[0-9][0-9]_*.py |
    Sort-Object Name |
    ForEach-Object { python $_.FullName }

python -m src.run_sql_quality_checks
```

Để kiểm định PostgreSQL đầy đủ, tạo `.env` từ `.env.example`, điền một database được phép sử dụng, sau đó chạy:

```powershell
python -m src.load_to_postgres --truncate
# Thực thi sql/marts theo thứ tự tên file bằng psql hoặc database client.
python -m src.run_sql_quality_checks
```

`--truncate` thay thế dữ liệu warehouse trong database đích; chỉ dùng với database đã được phê duyệt. Sau đó chạy lại notebook và các phép đối soát để xác nhận report khớp với mart SQL.

## Remaining work before Power BI

- Cấp cấu hình PostgreSQL hợp lệ và chạy lại load, mart SQL, data-quality SQL cho đến khi không còn FAIL.
- Đối soát trực tiếp tổng sales/transactions giữa PostgreSQL marts và các Parquet/report tables.
- Chốt schema, kiểu dữ liệu, quan hệ, hướng filter và grain cho model Power BI; giữ transactions ở grain ngày-cửa hàng và sales ở grain ngày-cửa hàng-family.
- Tạo date table/mark as date table, định nghĩa measures rõ ràng và tránh implicit measures gây double count.
- Xây dashboard, kiểm tra filter holiday theo store, refresh, performance và số liệu tổng trên từng trang.
- Thiết lập quy trình refresh, credential/gateway và tài liệu nguồn dữ liệu trước khi publish.

## Remaining work before Data Science

- Chốt target, forecast horizon, cấp dự báo và tiêu chí đánh giá theo bài toán kinh doanh.
- Thiết kế time-based split/backtest, chống leakage và xác định baseline phù hợp.
- Quyết định cách xử lý 15 giá trị lag dầu ở đầu chuỗi và các ngày không có giá dầu.
- Hoàn thiện feature pipeline có thể tái lập cho promotion, holiday theo store, transactions và oil; kiểm tra feature availability tại thời điểm dự báo.
- Thực hiện kiểm tra drift, outlier/anomaly policy, missing-data policy và cold-start cho store-family.
- Huấn luyện, đánh giá, theo dõi thí nghiệm và version hóa model/artifact; hiện chưa có kết quả model nào được phê duyệt.
