# Sales Anomaly Review Notes

- 92,620 daily observations were reviewed; no observation was deleted or modified.
- Rolling z-score uses only the preceding 28 observations and requires at least 14. The threshold is absolute z-score >= 3.0.
- The second method compares each value with the median and 1.5×IQR fences for its same-weekday peer group. This is a descriptive full-sample peer comparison.
- Holiday and event context is joined at date_key + store_key before system aggregation. Promotion context is derived from the original family rows.
- Review categories are Business event: 1694, Unexplained anomaly: 820, Potential data issue: 227. `Potential data issue` means only that an extreme/zero observation lacks mapped business context; it is not a confirmed data error. `Unexplained anomaly` also requires investigation.
- Thresholds are review heuristics, not causal or probabilistic claims. Business events may be unmapped, and mapped context does not necessarily explain the observed sales value.
