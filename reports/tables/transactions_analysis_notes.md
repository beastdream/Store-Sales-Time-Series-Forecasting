# Transactions Analysis Notes

- Grain is one point per date and store. Sales is aggregated across family before a one-to-one merge with transactions, so transactions is never repeated by family.
- Store-day Pearson correlation between sales and transactions is 0.837384. Correlation is association, not causation.
- 0 zero-transaction rows are retained. Their sales volume per transaction is undefined rather than infinite.
- 1,992 unusual store-days are flagged using an absolute modified z-score threshold of 3.5 within store. They are not removed.
- Across months with sales increases, dominant decomposition labels are Transactions: 18, Sales volume per transaction: 15. The identity decomposes arithmetic change; it does not prove a causal mechanism.
- Store rolling averages use 28 transaction observations with at least 7 observations. The overall trend figure uses a 28-calendar-day rolling mean.
