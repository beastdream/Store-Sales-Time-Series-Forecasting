"""Leakage-safe rolling-origin date boundaries for forecast validation."""

from dataclasses import dataclass
from numbers import Integral

import pandas as pd


DEFAULT_HORIZON_DAYS = 16
DEFAULT_N_FOLDS = 4


@dataclass(frozen=True)
class TemporalSplit:
    """Inclusive date boundaries for one rolling-origin validation fold.

    Training may use observations on or before ``train_end``. Validation targets
    start on the following calendar day and continue through ``validation_end``.
    """

    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp

    @property
    def validation_days(self) -> int:
        """Return the inclusive number of calendar days in the validation window."""
        return (self.validation_end - self.validation_start).days + 1


def _positive_integer(value: int, name: str) -> int:
    """Return a positive Python integer or raise a clear configuration error."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def make_rolling_splits(
    last_actual_date: str | pd.Timestamp,
    horizon: int = DEFAULT_HORIZON_DAYS,
    n_folds: int = DEFAULT_N_FOLDS,
) -> tuple[TemporalSplit, ...]:
    """Generate contiguous rolling-origin validation windows in time order.

    The final validation fold ends on ``last_actual_date``. Earlier folds are
    placed immediately before it, each with exactly ``horizon`` calendar days.
    For every fold, ``train_end`` is the day before ``validation_start``; therefore
    no validation target date can be part of that fold's training period.

    This function returns date boundaries only. It performs no random split and
    does not read target values, making the boundaries reusable across all
    store-family series.
    """
    horizon_days = _positive_integer(horizon, "horizon")
    fold_count = _positive_integer(n_folds, "n_folds")
    last_date = pd.Timestamp(last_actual_date).normalize()
    if pd.isna(last_date):
        raise ValueError("last_actual_date must be a valid date")

    first_validation_start = last_date - pd.Timedelta(
        days=horizon_days * fold_count - 1
    )
    splits: list[TemporalSplit] = []
    for fold_index in range(fold_count):
        validation_start = first_validation_start + pd.Timedelta(
            days=fold_index * horizon_days
        )
        validation_end = validation_start + pd.Timedelta(days=horizon_days - 1)
        splits.append(
            TemporalSplit(
                train_end=validation_start - pd.Timedelta(days=1),
                validation_start=validation_start,
                validation_end=validation_end,
            )
        )
    return tuple(splits)
