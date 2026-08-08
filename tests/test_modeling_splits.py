"""Contracts for reusable rolling-origin temporal validation boundaries."""

import pandas as pd
import pytest

from src.modeling.splits import make_rolling_splits


LAST_ACTUAL_DATE = pd.Timestamp("2017-08-15")


def test_default_splits_match_four_final_sixteen_day_windows() -> None:
    splits = make_rolling_splits(LAST_ACTUAL_DATE)

    assert len(splits) == 4
    assert [
        (split.validation_start, split.validation_end) for split in splits
    ] == [
        (pd.Timestamp("2017-06-13"), pd.Timestamp("2017-06-28")),
        (pd.Timestamp("2017-06-29"), pd.Timestamp("2017-07-14")),
        (pd.Timestamp("2017-07-15"), pd.Timestamp("2017-07-30")),
        (pd.Timestamp("2017-07-31"), pd.Timestamp("2017-08-15")),
    ]


def test_each_validation_window_has_sixteen_calendar_days() -> None:
    splits = make_rolling_splits(LAST_ACTUAL_DATE)

    assert all(split.validation_days == 16 for split in splits)
    assert all(
        len(pd.date_range(split.validation_start, split.validation_end, freq="D"))
        == 16
        for split in splits
    )


def test_folds_are_chronological_contiguous_and_end_on_last_actual_date() -> None:
    splits = make_rolling_splits(LAST_ACTUAL_DATE)

    assert list(splits) == sorted(splits, key=lambda split: split.validation_start)
    assert all(
        current.validation_end + pd.Timedelta(days=1) == following.validation_start
        for current, following in zip(splits, splits[1:])
    )
    assert splits[-1].validation_end == LAST_ACTUAL_DATE


def test_training_period_ends_before_validation_without_future_leakage() -> None:
    all_dates = pd.Series(pd.date_range("2017-01-01", LAST_ACTUAL_DATE, freq="D"))

    for split in make_rolling_splits(LAST_ACTUAL_DATE):
        training_dates = all_dates.loc[all_dates.le(split.train_end)]
        validation_dates = all_dates.loc[
            all_dates.between(split.validation_start, split.validation_end)
        ]

        assert split.train_end < split.validation_start
        assert training_dates.max() < validation_dates.min()
        assert set(training_dates).isdisjoint(validation_dates)


def test_custom_horizon_and_fold_count_are_generated_not_hard_coded() -> None:
    splits = make_rolling_splits("2024-01-31", horizon=7, n_folds=2)

    assert [(split.validation_start, split.validation_end) for split in splits] == [
        (pd.Timestamp("2024-01-18"), pd.Timestamp("2024-01-24")),
        (pd.Timestamp("2024-01-25"), pd.Timestamp("2024-01-31")),
    ]


@pytest.mark.parametrize(
    ("argument", "value"),
    [("horizon", 0), ("horizon", -1), ("horizon", 1.5), ("n_folds", 0)],
)
def test_invalid_split_configuration_raises(argument: str, value: object) -> None:
    kwargs = {argument: value}

    with pytest.raises(ValueError, match=argument):
        make_rolling_splits(LAST_ACTUAL_DATE, **kwargs)
