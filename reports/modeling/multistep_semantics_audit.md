# Multi-Step Forecasting and Lag-Semantics Audit

## Confirmed root cause

The previous training path attached target-history features directly to sparse
raw train rows. Grouped row shifts therefore meant “N previous observations,”
not necessarily calendar date t-N when an entire calendar date was absent.

The previous 16-day inference builder used a different rule. It masked all
post-origin targets, cycled lag references backward until they reached a
pre-origin date, and repeated the D+1 rolling snapshot across the full horizon.
It did not insert predictions. Consequently D+2 lag_1 could not equal predicted
D+1, and training and inference did not share one lag/rolling state transition.

## Corrected architecture

Both training and inference now begin with the canonical dense calendar frame:
one row for every date × store × family. Missing source observations retain
sales=NaN and sales_observed=0; observed zero remains sales=0 and
sales_observed=1.

Training builds lag and shifted rolling features on this dense frame and trains
only rows with observed targets. Therefore lag N means calendar date t-N.

Multi-step inference calls one shared recursive_forecast core. It masks every
target after the forecast origin, predicts D+1, appends only that prediction to a
private history, recomputes features for D+2, and repeats through the final day.
No actual validation/test target is appended.

## Artifact boundary

Existing baseline/model/tuning/error/interval reports, trained model files, and
final_submission.csv were produced before this semantics correction. They are
preserved unchanged as legacy evidence and must not be presented as results of
the recursive pipeline. Reproduce temporal backtests, selection, downstream
diagnostics, and final training in order before publishing replacement metrics or
a replacement submission.

No tuning, full backtest, model selection, final training, or submission
generation was run as part of this correction.
