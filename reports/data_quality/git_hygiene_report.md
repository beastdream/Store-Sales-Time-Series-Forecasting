# Git Hygiene Report

Ngày kiểm tra: 2026-08-06

## Files ignored

`.gitignore` hiện bỏ qua:

- Môi trường và cache: `.venv/`, `venv/`, `.env`, `__pycache__`, `*.pyc`,
  `.ipynb_checkpoints/`, `.pytest_cache/`.
- Artifacts dữ liệu tái tạo được: `data/interim/*`, `data/processed/*`,
  `data/features/*`, `models/*`.
- Toàn bộ raw CSV: `data/raw/*.csv`.
- Ba report CSV lớn có thể tái tạo:
  `reports/tables/holiday_analysis.csv`,
  `reports/tables/promotion_analysis_matched.csv`, và
  `reports/tables/transactions_analysis.csv`.
- Rule `!**/.gitkeep` tiếp tục giữ các placeholder directory.

Các file đã tracked vẫn xuất hiện trong Git cho đến khi người dùng chủ động bỏ
theo dõi; thêm rule ignore không tự xóa file local hoặc thay đổi Git index.

Các tài liệu sau đã được kiểm tra và **không bị ignore**:

- `reports/business_insights.md`
- `reports/da_project_validation.md`
- `reports/forecast_readiness.md`
- Nội dung trong `reports/data_quality/`

## Large tracked files

Ngưỡng liệt kê: ít nhất 1 MiB theo kích thước working tree.

| File | Kích thước | Đánh giá |
|---|---:|---|
| `reports/tables/holiday_analysis.csv` | 25,80 MiB | CSV tái tạo được; đã thêm ignore rule |
| `reports/tables/promotion_analysis_matched.csv` | 23,36 MiB | CSV tái tạo được; đã thêm ignore rule |
| `reports/tables/transactions_analysis.csv` | 17,64 MiB | CSV tái tạo được; đã thêm ignore rule |
| `data/raw/transactions.csv` | 1,48 MiB | Raw data; đã được rule `data/raw/*.csv` bao phủ |
| `reports/figures/transactions_analysis/unusual_transaction_days.png` | 1,06 MiB | Figure đang tracked; không thuộc danh sách yêu cầu bỏ theo dõi |

Không file nào trong bảng trên bị xóa hoặc bị bỏ theo dõi tự động.

## Raw files tracked

Các raw CSV sau vẫn đang có trong Git index:

- `data/raw/holidays_events.csv`
- `data/raw/oil.csv`
- `data/raw/sample_submission.csv`
- `data/raw/stores.csv`
- `data/raw/test.csv`
- `data/raw/transactions.csv`

`data/raw/train.csv` không nằm trong Git index và được ignore. File
`data/raw/.gitkeep` vẫn được phép theo dõi.

## Secrets scan result

- Không tìm thấy `.env` trong working tree; `.env.example` là template đang tracked
  và dùng placeholder `change_me`, không phải password thật.
- Không tìm thấy `.venv/` hoặc `venv/` trong working tree.
- Không tìm thấy private-key header, token phổ biến, access key, hoặc database URL
  chứa username/password trong tracked files.
- `src/database.py` chỉ tham chiếu `settings["DB_PASSWORD"]`; đây là tên biến cấu
  hình, không phải password thật được hard-code.

Kết quả là clean theo các pattern đã kiểm tra. Đây là static pattern scan, không
thay thế secret scanner chuyên dụng hoặc việc rotate credential nếu từng bị commit.

## Absolute path scan result

Không tìm thấy đường dẫn tuyệt đối máy cá nhân theo các mẫu thư mục user/project
phổ biến của Windows, Linux hoặc macOS trong tracked files.

Không có generated Parquet nào đang được Git theo dõi. Rule
`data/processed/*` và `data/features/*` tiếp tục bao phủ các Parquet được tạo lại,
trong khi `.gitkeep` được giữ.

## Commands người dùng cần tự chạy

Các lệnh dưới đây chỉ bỏ file khỏi Git index; option `--cached` giữ nguyên file
trong working tree. Hãy review bằng `git status` trước khi commit:

```bash
git rm --cached data/raw/*.csv
git rm --cached reports/tables/holiday_analysis.csv
git rm --cached reports/tables/promotion_analysis_matched.csv
git rm --cached reports/tables/transactions_analysis.csv
git status --short
```

Không cần chạy lệnh cho `data/raw/train.csv` vì file này không được theo dõi.
Không có commit nào được tạo trong lần kiểm tra này.
