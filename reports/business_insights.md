# Business Insights

Tất cả số liệu dưới đây được đọc từ các artifact trong `reports/tables/` và
`reports/data_quality/`. “Finding” là mô tả trực tiếp từ dữ liệu. Khi đưa ra một
cách giải thích chưa được kiểm định, báo cáo ghi rõ đó là “Hypothesis”. Không có
phân tích nào dưới đây chứng minh quan hệ nhân quả.

## 1. Executive summary

### Insight 1.1 — Quy mô lớn nhưng mức độ sẵn sàng dự báo không đồng đều

- **Finding:** Dữ liệu chứa `1.073.644.952,18` tổng sales trên 54 store và 33
  family, nhưng chỉ `364/1.782` chuỗi store–family (`20,4%`) được xếp loại Ready;
  `438` chuỗi (`24,6%`) đồng thời mang ít nhất hai risk flags.
- **Evidence:** `warehouse_reconciliation.md` xác nhận 3.000.888 fact rows, 54
  store, 33 family và tổng sales được reconcile; `forecast_readiness.csv` ghi
  nhận primary classes gồm 364 Ready, 345 Ready with caution, 417 Intermittent
  demand, 144 Insufficient history, 102 High volatility và 410 Promotion
  dependent; phân bố overlap gồm 380 chuỗi có hai và 58 chuỗi có ba risk flags.
- **Business implication:** Một chiến lược forecast duy nhất cho mọi chuỗi sẽ bỏ
  qua khác biệt lớn về lịch sử, zero sales, volatility và promotion exposure.
- **Recommended action:** Thiết kế validation và benchmark riêng theo sáu nhóm
  readiness; bắt đầu với 709 chuỗi Ready/Ready with caution, đồng thời xây
  baseline chuyên biệt cho intermittent và insufficient-history series.
- **Limitation:** Readiness là phân loại theo rule và quantile, chưa phải kết quả
  backtest model.

### Insight 1.2 — Calendar, promotion và transactions đều chứa tín hiệu mô tả

- **Finding:** Sunday có daily system sales trung bình `825.218,12`, December có
  trung bình `808.565,34`; correlation store-day giữa sales và transactions là
  `0,837`; median promotion uplift proxy giảm từ `111,85%` unmatched xuống
  `19,35%` khi matched.
- **Evidence:** Calendar averages lấy trực tiếp từ `daily_sales_summary.csv` và
  `weekday_month_summary.csv`; các số còn lại đến từ
  `transactions_analysis.csv` và `promotion_analysis_matched.csv`.
- **Business implication:** Calendar, traffic và promotion context nên được đánh
  giá như forecast features hoặc scenario inputs, không chỉ dùng trend lịch sử.
- **Recommended action:** Trong bước modeling tiếp theo, benchmark mô hình chỉ có
  lag/calendar với mô hình bổ sung transactions proxy, holiday mapping và
  promotion plan; so sánh bằng temporal backtest.
- **Limitation:** Các quan hệ trên là association; transactions tương lai có thể
  chưa biết và promotion assignment không ngẫu nhiên.

## 2. Data-quality findings

### Insight 2.1 — Grain và reconciliation đã qua kiểm tra

- **Finding:** Không có duplicate grain hoặc missing surrogate key trong bảy bảng
  warehouse; totals sales, onpromotion, transactions, store và family đều PASS.
- **Evidence:** `warehouse_reconciliation.md` ghi 3.000.888 sales rows,
  83.488 transaction rows, 7.938 holiday-bridge rows, duplicate grain = 0 và
  missing surrogate key = 0 cho mọi bảng.
- **Business implication:** Các mart và phân tích có nền tảng grain rõ ràng; đặc
  biệt transactions không bị lặp theo family và holiday không bị nhân theo event.
- **Recommended action:** Giữ các reconciliation hiện tại làm quality gate bắt
  buộc trước mỗi lần refresh dữ liệu và trước scoring.
- **Limitation:** PASS về cấu trúc không bảo đảm mọi giá trị phản ánh đúng thực tế
  kinh doanh; anomaly vẫn cần review với người sở hữu dữ liệu.

### Insight 2.2 — Zero sales và oil imputation cần được giữ dấu vết

- **Finding:** Có `939.130` train rows bằng zero sales; 43/1.218 oil observations
  raw (`3,53%`) bị thiếu và cleaning tạo `529` daily oil prices được impute.
- **Evidence:** `audit_summary.md` ghi zero/missing raw; `cleaning_summary.md` ghi
  529 giá trị oil được impute, còn zero-sales rows được giữ nguyên.
- **Business implication:** Zero sales có thể là genuine no-sale, closure hoặc
  intermittent demand; imputed oil có uncertainty khác observed oil.
- **Recommended action:** Giữ zero sales trong target, thêm indicator cho closure/
  intermittency nếu có, và luôn mang `oil_was_imputed` vào feature audit.
- **Limitation:** Dataset hiện không có một trường xác nhận nguyên nhân zero sales;
  đây là vấn đề cần điều tra, không phải lỗi đã được chứng minh.

## 3. Sales trend

### Insight 3.1 — Daily sales scale tăng mạnh qua các năm đầy đủ

- **Finding:** Average daily system sales tăng từ `385.766,52` năm 2013 lên
  `575.478,70` năm 2014, `661.758,52` năm 2015 và `790.834,31` năm 2016; mức
  2016 cao hơn 2013 khoảng `105,0%`.
- **Evidence:** `daily_sales_summary.csv` tổng hợp trực tiếp từ sales fact, với 364
  ngày có observation ở 2013, 364 ở 2014, 364 ở 2015 và 365 ở 2016.
- **Business implication:** Forecast cần học cả level/trend thay đổi theo thời gian;
  một average cố định toàn kỳ sẽ có nguy cơ under-forecast giai đoạn gần đây.
- **Recommended action:** Dùng expanding-window backtest và baseline có trend;
  theo dõi bias theo năm/store/family thay vì chỉ một metric tổng.
- **Limitation:** Trend mô tả có thể kết hợp store openings, assortment và nominal
  scale; dữ liệu không đủ để tách riêng từng cơ chế.

### Insight 3.2 — Năm 2017 chỉ được so sánh theo cùng cutoff

- **Finding:** YTD sales đến `2017-08-15` là `194.217.068,37`, cao hơn cùng kỳ
  2016 (`176.562.909,57`) khoảng `10,00%`; average daily sales 2017 là
  `855.581,80` trên 227 ngày observation, không phải một năm đầy đủ.
- **Evidence:** `daily_sales_summary.csv` cung cấp calendar-aligned YTD; trong
  `monthly_sales_summary.csv`, tháng 07/2017 có MoM `+5,17%` và YoY `+15,13%`,
  còn tháng 08/2017 được đánh dấu incomplete nên không có growth hợp lệ.
- **Business implication:** Annual totals hoặc year-over-year 2017 có thể gây hiểu
  nhầm nếu không align cùng date range.
- **Recommended action:** Khi trình bày YoY, so sánh year-to-date cùng cutoff hoặc
  dùng daily/monthly average với nhãn partial period.
- **Limitation:** Daily average vẫn chịu season mix vì 2017 chưa có Sep–Dec.

### Insight 3.3 — Missing observation không đồng nghĩa sales bằng 0

- **Finding:** Có bốn ngày thiếu sales observation trong lịch sử: `2013-12-25`,
  `2014-12-25`, `2015-12-25` và `2016-12-25`; cả bốn đều có
  `has_sales_observation = 0` và `total_sales` để trống.
- **Evidence:** `daily_sales_summary.csv` giữ complete calendar từ `dim_date` và
  không zero-fill các ngày thiếu; moving averages loại các giá trị thiếu khỏi
  trailing 7/28 calendar-day windows.
- **Business implication:** Gán các ngày này bằng 0 sẽ tạo target giả và làm lệch
  trend, seasonality cũng như holiday comparison.
- **Recommended action:** Giữ `has_sales_observation` trong feature/quality layer và
  quyết định imputation riêng trong modeling, có audit rõ ràng.
- **Limitation:** Dữ liệu không cho biết nguyên nhân không có observation; không thể
  kết luận cửa hàng đóng cửa hay sales thực tế bằng 0.

## 4. Seasonality

### Insight 4.1 — Weekend, đặc biệt Sunday, có scale cao hơn

- **Finding:** Daily system sales trung bình cao nhất vào Sunday (`825.218,12`),
  kế đến Saturday (`772.205,59`); Thursday thấp nhất (`505.269,20`). Weekend
  average (`798.656,75`) cao hơn weekday (`573.143,02`) khoảng `39,3%`.
- **Evidence:** `daily_sales_summary.csv` và `weekday_month_summary.csv` tổng hợp
  trực tiếp observed daily sales; missing observations không được coi là 0.
- **Business implication:** Weekly seasonality là tín hiệu forecast quan trọng và
  có thể ảnh hưởng kế hoạch staffing/vận hành theo ngày.
- **Recommended action:** Bắt buộc có weekday features và báo cáo forecast error
  theo weekday; dùng Sunday/Thursday làm hai slice kiểm tra bias.
- **Limitation:** Đây là average toàn hệ thống, có thể che khác biệt giữa store và
  family; không suy ra rằng weekday tự gây ra sales cao.

### Insight 4.2 — December nổi bật nhưng hypothesis cần kiểm định

- **Finding:** Theo month-of-year, December cao nhất (`808.565,34` average daily
  sales) và February thấp nhất (`571.895,24`), chênh khoảng `41,4%`.
- **Evidence:** `daily_sales_summary.csv` và `weekday_month_summary.csv` được tổng
  hợp theo calendar month qua toàn bộ observed years. Tháng cao tiếp theo là
  November (`669.464,90`) và July (`666.858,46`); payday average là `645.965,89`
  so với non-payday `636.962,96` (`+1,4%`).
- **Business implication:** Forecast horizon chạm các tháng cao/thấp cần seasonal
  calibration; cùng một mức safety buffer cho mọi tháng sẽ thiếu linh hoạt.
- **Recommended action:** Thêm month/year-season features và backtest riêng
  November–December so với February; chỉ điều chỉnh vận hành theo forecast và
  service target, không suy ra lượng nhập chính xác từ bảng này.
- **Limitation:** **Hypothesis:** holiday/payday mix có thể góp phần vào December;
  phân tích mô tả này chưa cô lập holiday, promotion hoặc trend năm.

## 5. Store performance

### Insight 5.1 — Store 44 dẫn đầu average daily sales

- **Finding:** Store 44 tại Quito có average daily sales `36.869,09`, cao nhất;
  tiếp theo là store 45 (`32.362,24`), store 47 (`30.254,34`) và store 3
  (`29.977,38`).
- **Evidence:** `store_performance.csv`; store 44 có CV `0,399`, growth proxy
  `106,7%` và sales volume per transaction `8,54`.
- **Business implication:** Store lớn cần được theo dõi bằng absolute error và bias,
  vì sai số phần trăm nhỏ vẫn có thể tạo chênh lệch sales volume lớn.
- **Recommended action:** Tạo dashboard forecast bias cho top-volume stores và
  đặt review cadence ngắn hơn cho store 44/45/47/3.
- **Limitation:** Ranking là sales volume, không phải profit hay hiệu quả vốn; bảng
  không có cost hoặc inventory.

### Insight 5.2 — Store 52 có scale thấp và volatility rất cao

- **Finding:** Store 52 tại Manta có average daily sales thấp nhất (`1.601,05`) và
  CV cao nhất (`3,738`); cả 33 store–family series của store này thuộc issue group
  trong readiness report.
- **Evidence:** `store_performance.csv` và `forecast_readiness.csv`; anomaly review
  cũng ghi 118 store-day review flags cho store 52.
- **Business implication:** Fit riêng từng series ở store 52 có nguy cơ không ổn
  định; prediction interval và fallback logic quan trọng hơn point forecast đơn lẻ.
- **Recommended action:** Ưu tiên pooled/global hoặc hierarchical baseline cho
  store 52, theo dõi interval coverage và review data/business regime trước khi
  dùng forecast cho vận hành.
- **Limitation:** Anomaly flag không phải lỗi dữ liệu; scale thấp có thể phản ánh
  store maturity hoặc assortment chưa có trong dataset.

## 6. Product family performance

### Insight 6.1 — Sales tập trung vào một số family lớn

- **Finding:** GROCERY I đóng góp `343.462.734,87` (`32,0%`) và BEVERAGES
  `216.954.486,00` (`20,2%`); riêng hai family chiếm khoảng `52,2%` total sales.
- **Evidence:** `family_performance.csv`; các family tiếp theo là PRODUCE `11,4%`,
  CLEANING `9,1%` và DAIRY `6,0%`.
- **Business implication:** Sai số forecast của GROCERY I/BEVERAGES có tác động lớn
  lên aggregate plan, nhưng long-tail family vẫn cần service-aware metrics.
- **Recommended action:** Báo cáo metric theo contribution tier; thêm weighted
  aggregate metric nhưng không bỏ macro metric trên 33 family.
- **Limitation:** Sales contribution không cho biết margin hoặc stock risk vì
  dataset không có cost và inventory.

### Insight 6.2 — Một số family có demand rất intermittent

- **Finding:** BOOKS có zero-sales rate `97,0%` và CV `7,740`; BABY CARE `94,1%`
  và CV `6,162`; SCHOOL AND OFFICE SUPPLIES `74,1%` và CV `7,343`.
- **Evidence:** `family_performance.csv`; forecast readiness xếp toàn bộ 54 series
  của BOOKS và BABY CARE vào issue groups.
- **Business implication:** Average-based forecast có thể tạo nonzero predictions
  liên tục cho demand vốn xuất hiện thưa, làm giảm tính hữu dụng vận hành.
- **Recommended action:** Benchmark Croston-style/two-stage occurrence-size hoặc
  pooled classifiers cho intermittent families; đánh giá zero/nonzero accuracy
  song song với magnitude error.
- **Limitation:** Zero sales không đồng nghĩa zero demand nếu có stockout; dataset
  không có inventory để phân biệt.

## 7. Promotion findings

> Promotion metrics are descriptive associations and do not establish causal effects.

### Insight 7.1 — Unmatched promotion proxy phóng đại nhiều family

- **Finding:** Median family promotion uplift proxy là `111,85%` ở unmatched nhưng
  `19,35%` khi match trong cùng store, family, year, month và weekday.
- **Evidence:** `promotion_analysis_matched.csv` có 102.648 matched cells và 32
  family summaries. SCHOOL AND OFFICE SUPPLIES giảm từ `4.256,98%` unmatched
  xuống `655,20%` matched trên 1.435 matched cells.
- **Business implication:** Overall comparison trộn composition/calendar khác nhau;
  dùng nó làm expected campaign effect sẽ quá lạc quan.
- **Recommended action:** Dùng matched proxy làm descriptive diagnostic và yêu cầu
  temporal/causal design chặt hơn trước khi đưa promotion effect vào planning.
- **Limitation:** Matched comparison đáng tin hơn về comparability nhưng vẫn không
  phải causal inference; price, campaign selection và demand chưa quan sát vẫn
  gây confounding.

### Insight 7.2 — Sample size không đồng đều

- **Finding:** BOOKS có 0 promotion observations nên proxy để trống; BABY CARE chỉ
  có 53 và HOME APPLIANCES 58 promotion observations, dưới ngưỡng cảnh báo 100.
- **Evidence:** `promotion_analysis_basic.csv` lưu count hai phía và
  `small_sample_warning`; non-promotion counts tương ứng là 90.936, 90.883 và
  90.878.
- **Business implication:** Extreme proxy ở cohort nhỏ có độ ổn định thấp và không
  nên dùng trực tiếp cho budget hoặc forecast adjustment.
- **Recommended action:** Hiển thị mọi proxy kèm sample size; đặt trạng thái
  “insufficient promotion evidence” cho cohort dưới 100 và không tự động fill
  BOOKS bằng effect của family khác.
- **Limitation:** Ngưỡng 100 là business warning rule, không phải confidence bound.

## 8. Holiday and event findings

### Insight 8.1 — Holiday windows có distribution dịch lên nhưng rất rộng

- **Finding:** Median difference so với baseline bốn tuần cùng weekday là
  `+13,69%` trong holiday, `−0,26%` trước holiday và `−1,81%` sau holiday.
- **Evidence:** `holiday_analysis.csv` có 4.054 during-holiday, 2.548 before và
  2.494 after observations có baseline; không row nào bị nhân theo multi-event.
- **Business implication:** Event-window feature có thể cải thiện forecast timing,
  nhưng một scalar holiday uplift chung sẽ bỏ qua heterogeneity.
- **Recommended action:** Encode before/during/after flags và interaction với locale/
  event type; backtest theo event category thay vì áp một multiplier cố định.
- **Limitation:** Baseline là approximation từ lag 7/14/21/28 ngày, không phải
  counterfactual và không chứng minh holiday gây thay đổi.

### Insight 8.2 — Locale/type có mức và độ phân tán khác nhau

- **Finding:** Median baseline difference là Additional `+23,42%`, Transfer
  `+15,20%`, National holiday `+15,16%`, Regional holiday `+13,74%`, Holiday
  `+6,18%` và Event `−2,31%`.
- **Evidence:** `holiday_event_summary.csv`; National holiday có Q25 `−2,89%` và
  Q75 `+42,29%`, còn Regional chỉ có 31 observations.
- **Business implication:** National/local/regional/type mapping cần giữ đúng
  store geography; date-only join có thể làm sai tín hiệu.
- **Recommended action:** Dùng bridge `date + store_nbr` làm nguồn feature chuẩn và
  cảnh báo category nhỏ như Regional trong model diagnostics.
- **Limitation:** Category có thể overlap nên counts không additive; 218 mapped
  special store-days ngoài observed sales range được giữ nhưng không có actual sales.

## 9. Transaction findings

### Insight 9.1 — Transactions liên hệ mạnh với sales ở store-day

- **Finding:** Pearson correlation giữa total sales và transactions là `0,837`
  trên 83.488 store-days.
- **Evidence:** `transactions_analysis.csv`; scatter dùng đúng một điểm/store-day và
  tổng transactions reconcile ở `141.478.945`.
- **Business implication:** Traffic volume là explanatory signal quan trọng, nhưng
  actual future transactions có thể không sẵn tại forecast time.
- **Recommended action:** So sánh mô hình dùng lag/rolling transactions với mô hình
  không dùng; chỉ dùng feature biết trước hoặc transaction forecast riêng để tránh
  leakage.
- **Limitation:** Correlation không phải causation; promotion, holiday và store scale
  có thể cùng tác động lên cả hai biến.

### Insight 9.2 — Store traffic và volume per transaction là hai chiều khác nhau

- **Finding:** Store 44 có nhiều transactions nhất (`7.273.093`), trong khi store
  51 dẫn đầu sales volume per transaction (`11,43`). Trong 33 tháng sales tăng,
  decomposition gắn 18 tháng với transaction effect trội và 15 tháng với
  sales-volume-per-transaction effect trội.
- **Evidence:** `transactions_store_summary.csv` và
  `transactions_monthly_driver.csv`; monthly arithmetic decomposition reconcile.
- **Business implication:** Sales growth không nên được diễn giải chỉ là tăng traffic;
  basket/volume dynamics cũng có vai trò mô tả đáng kể.
- **Recommended action:** Theo dõi hai KPI riêng theo store và month; review forecast
  miss bằng decomposition để xác định traffic forecast hay volume assumption lệch.
- **Limitation:** “Dominant driver” là decomposition số học, không chứng minh cơ chế
  hành vi hoặc nguyên nhân kinh doanh.

## 10. Forecast readiness

### Insight 10.1 — Primary class che một phần rủi ro chồng lấp

- **Finding:** Primary class ghi 417 Intermittent, 410 Promotion dependent và 102
  High volatility, nhưng các flag độc lập lần lượt là 530 (`29,7%`), 426 (`23,9%`)
  và 469 (`26,3%`). Có 709 chuỗi không mang risk flag, 635 có một, 380 có hai và
  58 có ba risk flags.
- **Evidence:** `forecast_readiness.csv` và `forecast_readiness.md`; threshold được
  giữ nguyên: zero-sales Q75 `23,05%`, CV Q75 `1,110`, promotion-rate Q75
  `43,25%`, history rule 365 ngày/90 active days và Ready rule 730 ngày.
- **Business implication:** Modeling roadmap nên ưu tiên nhóm có đủ signal rồi mở
  rộng bằng specialized methods, thay vì tối ưu một average score toàn bộ hierarchy.
- **Recommended action:** Xây benchmark matrix theo primary class nhưng báo cáo
  error/bias/coverage thêm theo từng flag và overlap count; không coi primary class
  là mô tả đầy đủ mọi rủi ro của series.
- **Limitation:** Quantile rules phản ánh dataset này; phải tái tính khi data refresh
  hoặc business scope thay đổi.

### Insight 10.2 — Overlapping risks tập trung theo family và store

- **Finding:** BOOKS có 54/54 series mang ít nhất hai risks và tổng 132 flags;
  BABY CARE có 53/54 và 128 flags; SCHOOL AND OFFICE SUPPLIES có 53/54. Store 52
  dẫn đầu store overlap với 20/33 series và tổng 54 flags.
- **Evidence:** Các bảng “Family/Store có nhiều overlapping risks” trong
  `forecast_readiness.md`, được tổng hợp từ các cột nhị phân của
  `forecast_readiness.csv`.
- **Business implication:** Issue mang tính cấu trúc theo hierarchy, phù hợp với
  pooled/global hoặc hierarchical learning hơn 1.782 local models độc lập.
- **Recommended action:** Thiết kế global model có store/family embeddings hoặc
  categorical effects, nhưng vẫn giữ fallback theo readiness class.
- **Limitation:** Flags dùng threshold rule/quantile và có correlation với nhau;
  overlap không chứng minh nhiều nguyên nhân độc lập. Chưa có model nào được huấn
  luyện ở giai đoạn EDA/readiness.

## 11. Data limitations

### Insight 11.1 — Không có inventory và cost

- **Finding:** Các bảng hiện có sales, promotion, transactions, oil, store metadata
  và event calendar; không có inventory, stockout, price, unit cost hoặc margin.
- **Evidence:** `column_audit.csv` liệt kê schema raw; các bảng warehouse/report
  không bổ sung các trường inventory/cost.
- **Business implication:** Không thể phân biệt zero demand với lost sales do
  stockout, không thể đề xuất số lượng nhập kho chính xác và không thể tính profit.
- **Recommended action:** Trước khi chuyển forecast thành replenishment decision,
  tích hợp on-hand, inbound, lead time, stockout, selling price và cost tables.
- **Limitation:** Ngay cả khi bổ sung dữ liệu, cần xác định service-level/capacity
  constraints; sales forecast không tự động trở thành inventory policy.

### Insight 11.2 — Nhiều context vẫn là proxy

- **Finding:** Có 2.741 sales anomaly review flags: 1.694 Business event, 820
  Unexplained anomaly và 227 Potential data issue.
- **Evidence:** `sales_anomalies.csv`; 25 flags ở system-day và 2.716 ở store-day.
- **Business implication:** Một phần biến động chưa có context rõ; tự động clean
  anomaly có thể xóa business signal thật.
- **Recommended action:** Thiết lập review queue cho Potential data issue và top
  Unexplained anomaly; lưu quyết định/metadata thay vì overwrite actual sales.
- **Limitation:** Potential data issue chỉ là heuristic label, không phải lỗi đã
  xác nhận; promotion/holiday mappings cũng có thể chưa bao phủ mọi hoạt động.

## 12. Business recommendations

### Recommendation 12.1 — Lập kế hoạch theo calendar slice

- **Finding:** Sunday average daily sales `825.218,12` so với Thursday
  `505.269,20`; December `808.565,34` so với February `571.895,24`.
- **Evidence:** Section 4, `daily_sales_summary.csv` và
  `weekday_month_summary.csv`.
- **Business implication:** Workload và sales volume có seasonality tuần/tháng rõ.
- **Recommended action:** Tạo forecast review theo weekday/month và dùng forecast
  interval để hỗ trợ staffing, allocation và vận hành; không quy đổi trực tiếp
  thành lượng nhập cụ thể khi chưa có inventory/lead time.
- **Limitation:** Aggregate system pattern cần được kiểm tra lại tại store–family.

### Recommendation 12.2 — Tách forecast strategy theo readiness

- **Finding:** 709/1.782 chuỗi không có serious risk flag; 438 chuỗi có ít nhất hai
  risks. Flag độc lập ghi 530 Intermittent, 469 High volatility, 426 Promotion
  dependent và 144 Insufficient history.
- **Evidence:** Section 10 và `forecast_readiness.csv`.
- **Business implication:** Local model cho chuỗi thưa/ngắn dễ thiếu ổn định.
- **Recommended action:** Dùng global/hierarchical baseline cho toàn bộ hierarchy,
  thêm intermittent benchmark cho 530 flagged series, cold-start fallback cho 144
  series và đánh giá riêng 438 series có overlapping risks.
- **Limitation:** Chọn phương pháp cuối cùng phải dựa trên temporal backtest 16-day
  horizon, không dựa riêng vào readiness label.

### Recommendation 12.3 — Quản trị promotion theo scenario, không dùng raw uplift

- **Finding:** Median proxy giảm từ 111,85% unmatched xuống 19,35% matched; ba
  family có promotion cohort dưới 100 hoặc bằng 0.
- **Evidence:** Section 7, `promotion_analysis_matched.csv` và
  `promotion_analysis_basic.csv`.
- **Business implication:** Raw average difference có thể làm kế hoạch promotion
  quá lạc quan.
- **Recommended action:** Yêu cầu promotion calendar 16 ngày tới, chạy forecast
  scenario with/without planned promotion và gắn sample-size warning vào output.
- **Limitation:** Scenario delta vẫn là prediction; matched proxy không phải causal
  effect và không thay thế experiment/quasi-experiment.

### Recommendation 12.4 — Giữ event mapping ở store geography

- **Finding:** National holiday median difference `+15,16%`, Local `+3,04%`, Event
  `−2,31%`; distributions rộng và category overlap.
- **Evidence:** Section 8 và `holiday_event_summary.csv`.
- **Business implication:** Date-only feature sẽ gán local/regional event sai store.
- **Recommended action:** Production feature pipeline phải join event bằng
  `date + store_nbr`, giữ multi-event aggregation và event-window flags.
- **Limitation:** Historical association có thể đổi theo năm và store operations.

### Recommendation 12.5 — Tạo monitoring theo traffic, volume và anomaly context

- **Finding:** Correlation sales–transactions `0,837`; monthly increases chia 18
  transaction-led và 15 volume-led; còn 820 anomalies chưa giải thích.
- **Evidence:** Sections 9 và 11 từ `transactions_*` và `sales_anomalies.csv`.
- **Business implication:** Forecast miss có thể đến từ traffic, volume per
  transaction hoặc context chưa map.
- **Recommended action:** Sau deployment, phân rã monthly miss theo hai driver và
  route extreme unexplained cases vào review queue trước khi retraining.
- **Limitation:** Driver decomposition là arithmetic và anomaly routing là heuristic.

## 13. Why forecasting is needed

### Insight 13.1 — Forecast tạo cầu nối từ EDA sang quyết định có kiểm soát

- **Finding:** Dataset có 1.782 store–family series với scale, seasonality,
  intermittency, promotion dependence và event exposure khác nhau; test horizon có
  đúng 16 ngày từ 16/08 đến 31/08/2017.
- **Evidence:** `forecast_readiness.csv` có đủ 54 × 33 series; `audit_summary.md`
  xác nhận train kết thúc 15/08 và test gồm 28.512 rows trong 16 ngày tiếp theo.
- **Business implication:** Historical averages không đủ để tạo kế hoạch nhất quán
  cho từng store–family; forecast cho phép lượng hóa expected sales và uncertainty
  theo horizon vận hành cụ thể.
- **Recommended action:** Xây temporal backtest mô phỏng đúng horizon 16 ngày,
  benchmark theo readiness class và xuất point forecast cùng interval/quality flag
  để hỗ trợ allocation và operations planning.
- **Limitation:** Forecast sales volume không trực tiếp quyết định inventory,
  replenishment hoặc profit; các quyết định đó cần thêm inventory, lead time, cost,
  service-level và capacity constraints.

**Bài toán Data Science:** Dự báo sales volume của từng tổ hợp store–family trong 16 ngày tiếp theo để hỗ trợ phân bổ hàng hóa và lập kế hoạch vận hành.
