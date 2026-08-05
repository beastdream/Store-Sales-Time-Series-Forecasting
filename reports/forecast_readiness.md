# Forecast Readiness

> Phạm vi của bước này chỉ là đánh giá dữ liệu. Chưa huấn luyện, chọn hoặc đánh giá model.

## Dữ liệu và grain

Bảng chi tiết có `1,782` chuỗi tại grain `store_nbr × family`. Metrics được tính trong cửa sổ từ ngày sales dương đầu tiên đến ngày sales dương cuối cùng; các ngày zero sales bên trong cửa sổ vẫn được giữ. Báo cáo đối chiếu thêm các bảng EDA đã tạo: `family_performance.csv`, `store_performance.csv` và `sales_anomalies.csv`.

## Ngưỡng sử dụng

- Business rule: ít hơn `365` ngày lịch sử hoặc `90` active days là **Insufficient history**. Một năm lịch sử nhằm phủ ít nhất một chu kỳ mùa vụ; 90 active days là mức tối thiểu để có đủ quan sát sales dương cho pattern tuần/tháng.
- Business rule: **Ready** cần ít nhất `730` ngày, tương đương khoảng hai chu kỳ năm.
- Median zero-sales rate: `5.2876%`; Q75: `23.0478%`.
- Median coefficient of variation: `0.7558`; Q75: `1.1096`.
- Promotion-rate Q75: `43.2457%`.
- Missing-period Q75: `4` ngày.

Các median/Q75 chỉ được tính trên chuỗi đã có ít nhất một năm lịch sử và 90 active days, để chuỗi chưa hoạt động không làm méo ngưỡng.

## Quy tắc phân loại

Quy tắc được áp dụng theo thứ tự ưu tiên và mỗi chuỗi chỉ nhận một nhãn:

1. **Insufficient history:** history < 365 ngày hoặc active days < 90.
2. **Intermittent demand:** zero-sales rate ≥ Q75.
3. **Promotion dependent:** promotion rate ≥ Q75.
4. **High volatility:** coefficient of variation ≥ Q75.
5. **Ready:** history ≥ 730, zero-sales rate ≤ median, CV ≤ median và missing periods ≤ Q75.
6. **Ready with caution:** đủ lịch sử nhưng chưa đạt toàn bộ điều kiện Ready và không rơi vào các risk group Q75 ở trên.

## Phân bố readiness

| Nhóm | Số chuỗi | Tỷ lệ |
| --- | --- | --- |
| Ready | 364 | 20.4% |
| Ready with caution | 345 | 19.4% |
| Intermittent demand | 417 | 23.4% |
| Insufficient history | 144 | 8.1% |
| High volatility | 102 | 5.7% |
| Promotion dependent | 410 | 23.0% |

## Family thường gặp vấn đề

`issue_count` gồm Intermittent demand, Insufficient history, High volatility và Promotion dependent. Các metric family-level lấy trực tiếp từ `family_performance.csv`.

| family | issue_count | issue_rate | dominant_issue | zero_sales_rate | coefficient_of_variation | promotion_rate |
| --- | --- | --- | --- | --- | --- | --- |
| BABY CARE | 54 | 100.0% | Insufficient history | 94.1% | 6.162 | 0.1% |
| BOOKS | 54 | 100.0% | Insufficient history | 97.0% | 7.740 | 0.0% |
| GROCERY I | 54 | 100.0% | Promotion dependent | 8.1% | 0.761 | 62.6% |
| HOME APPLIANCES | 54 | 100.0% | Intermittent demand | 73.5% | 2.119 | 0.1% |
| SCHOOL AND OFFICE SUPPLIES | 54 | 100.0% | Intermittent demand | 74.1% | 7.343 | 4.5% |
| BEVERAGES | 53 | 98.1% | Promotion dependent | 8.1% | 0.967 | 56.9% |
| CLEANING | 53 | 98.1% | Promotion dependent | 8.1% | 0.685 | 54.9% |
| FROZEN FOODS | 51 | 94.4% | Promotion dependent | 8.1% | 2.100 | 39.4% |
| HARDWARE | 49 | 90.7% | Intermittent demand | 47.9% | 1.440 | 0.1% |
| DAIRY | 45 | 83.3% | Promotion dependent | 8.1% | 0.948 | 50.6% |

## Store thường gặp vấn đề

Store metrics lấy từ `store_performance.csv`; anomaly count lấy từ `sales_anomalies.csv` và chỉ là review flag, không phải lỗi dữ liệu.

| store_nbr | issue_count | issue_rate | dominant_issue | average_daily_sales | coefficient_of_variation | anomaly_count |
| --- | --- | --- | --- | --- | --- | --- |
| 52 | 33 | 100.0% | Insufficient history | 1,601.05 | 3.738 | 118 |
| 25 | 25 | 75.8% | High volatility | 6,782.07 | 0.702 | 117 |
| 36 | 25 | 75.8% | Intermittent demand | 9,098.83 | 0.478 | 68 |
| 35 | 24 | 72.7% | Intermittent demand | 4,558.60 | 0.516 | 48 |
| 22 | 23 | 69.7% | Promotion dependent | 2,428.86 | 1.305 | 16 |
| 28 | 23 | 69.7% | Intermittent demand | 10,916.36 | 0.468 | 45 |
| 33 | 23 | 69.7% | Intermittent demand | 8,419.54 | 0.340 | 54 |
| 53 | 23 | 69.7% | Promotion dependent | 6,660.43 | 0.877 | 19 |
| 10 | 22 | 66.7% | Intermittent demand | 5,708.97 | 0.297 | 63 |
| 16 | 22 | 66.7% | Intermittent demand | 6,524.00 | 0.315 | 102 |

## Ảnh hưởng tới forecasting

- **Ready:** phù hợp cho baseline/model chuẩn sau khi thiết kế validation theo thời gian.
- **Ready with caution:** cần kiểm tra thêm scale, recent regime và feature availability.
- **Intermittent demand:** nhiều zero; metric như MAE đơn thuần có thể che khuất khả năng dự báo occurrence. Nên cân nhắc intermittent-demand methods hoặc mô hình hai giai đoạn occurrence/size.
- **Insufficient history:** chưa phủ đủ mùa vụ; nên dùng pooled/global model, hierarchical information hoặc benchmark đơn giản thay vì fit riêng chuỗi.
- **High volatility:** prediction intervals cần rộng hơn; validation nhiều fold và robust loss có thể quan trọng hơn point accuracy đơn lẻ.
- **Promotion dependent:** forecast cần promotion plan tương lai đáng tin cậy; kịch bản thiếu promotion feature phải được đánh giá riêng.

## Bước tiếp theo

Chưa có model nào được huấn luyện trong bước này. Bước sau mới nên xác định forecast horizon, temporal split, baseline và metric theo từng readiness group.
