# Family Performance and Forecast Readiness

Monthly growth is the median of valid month-over-month changes; a transition whose prior month has zero sales is retained in the source but excluded from division.

## Segmentation rules

Rules are applied in priority order so every family receives exactly one segment:

1. **Promotion dependent:** promotion rate is at or above Q75.
2. **High volume – stable:** total sales is at or above the median and CV is at or below the median.
3. **High volume – volatile:** total sales is at or above the median and CV is above the median.
4. **Low volume – intermittent:** total sales is below the median and zero-sales rate is at or above Q75.
5. **Low volume – stable:** remaining low-volume families.

Thresholds: total-sales median = 1962767.000000; CV median = 1.439613; zero-sales Q75 = 0.467846; promotion-rate Q75 = 0.357812.

## Findings

- GROCERY I contributes the most sales at 343,462,734.89 (32.0% of all sales).
- BOOKS has the highest normalized volatility, with a coefficient of variation of 7.740.
- BOOKS has the highest zero-sales rate at 97.0%; zero-sales rows remain in all metrics.
- GROCERY I has the highest promotion observation rate at 62.6%.
- Readiness segment counts are High volume – stable: 5, High volume – volatile: 3, Low volume – intermittent: 9, Low volume – stable: 7, Promotion dependent: 9; thresholds are median total sales 1,962,767.00, median CV 1.440, zero-sales Q75 46.8%, and promotion-rate Q75 35.8%.
