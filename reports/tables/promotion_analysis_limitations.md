# Promotion Analysis Limitations

- Promotion metrics are descriptive associations and do not establish causal effects. Promotion assignment is not randomized.
- Product selection, store, seasonality, holidays, pricing, and underlying demand may differ between promotion and non-promotion observations.
- Aggregated comparisons can hide variation within a family, store, or calendar group.
- The matched comparison is more comparable than the unmatched overall comparison because it holds store, family, year, month, and day of week constant. It reduces observed composition and calendar-mix differences, but it is still not causal inference.
- Matching retains 102,648 cells containing both cohorts. It uses only observations inside each contemporaneous cell and does not use future data. Cells missing either cohort cannot contribute to matched results, so selection into the matched sample remains a limitation.
- No outlier is automatically removed. Extreme values remain in the averages and promotion uplift proxy; robust sensitivity analysis would be needed before choosing any documented exclusion rule.
- Cohorts with fewer than 100 observations are flagged; 3 unmatched grouped rows and 102,648 detailed matched cells have at least one small cohort. Aggregated family and store-type sample counts are also retained in the matched table.
- Division by a zero non-promotion average is left undefined; 1 unmatched grouped row has no finite promotion uplift proxy.
- `onpromotion` is a count of promoted items, not promotion depth, discount size, or campaign exposure. Its association with sales is not causal.
