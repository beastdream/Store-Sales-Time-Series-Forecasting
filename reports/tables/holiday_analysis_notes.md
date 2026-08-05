# Holiday Analysis Notes and Limitations

- Grain is one row per date and store. The holiday bridge is joined on both date_key and store_key with a one-to-one validation.
- 239 store-days contain multiple mapped events. Their descriptions/types stay aggregated in one row, so sales is not multiplied.
- 218 mapped special store-days fall outside the observed sales fact range. They remain in the table with missing actual sales and are not used as sales observations in summaries.
- Baseline sales is the mean of sales exactly 7, 14, 21, and 28 days earlier for the same store. It uses no future observations and therefore compares the same weekday approximately.
- Prior special days are not removed from the baseline. Their count is recorded in `baseline_special_day_count`.
- 2,862 calendar-store rows lack all four historical sales observations because of early dates or gaps outside the observed fact range. They remain in the output with an incomplete-baseline status and undefined baseline comparison.
- Events below 30 observations are flagged; 35 distinct event labels receive this warning.
- Distribution charts retain all observations and outliers. Log scale is used only for readability in sales-distribution charts.
- Event categories can overlap, so category-summary counts are not additive.
- Results are descriptive associations, not causal estimates. Holidays, promotions, store operations, trends, and other contemporaneous factors may all contribute to differences.
