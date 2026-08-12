from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modeling.error_analysis import (
    attach_readiness_labels,
    score_failure_flags,
    score_segments,
    validate_oof_predictions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _oof() -> pd.DataFrame:
    rows = []
    for fold in range(1, 5):
        for store_nbr, family, actual, prediction in [
            (1, "A", 0.0, 1.0),
            (1, "B", 10.0, 8.0),
            (2, "A", 20.0, 18.0),
        ]:
            rows.append(
                {
                    "fold": fold,
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=fold),
                    "store_nbr": store_nbr,
                    "family": family,
                    "store_type": "X" if store_nbr == 1 else "Y",
                    "promotion_active": fold % 2,
                    "is_holiday": int(fold == 1),
                    "sales": actual,
                    "prediction": prediction,
                }
            )
    return pd.DataFrame(rows)


def _readiness() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "store_nbr": [1, 1, 2],
            "family": ["A", "B", "A"],
            "readiness_class": ["Intermittent demand", "High volatility", "Ready"],
            "is_high_volatility": [0, 1, 0],
            "is_intermittent": [1, 0, 0],
            "is_promotion_dependent": [0, 1, 0],
            "is_insufficient_history": [0, 0, 0],
        }
    )


def test_segment_scores_match_direct_metric_formulas() -> None:
    oof = _oof()
    scores = score_segments(oof, ["store_nbr"]).set_index("store_nbr")
    store_one = oof.loc[oof["store_nbr"].eq(1)]
    expected_rmsle = np.sqrt(
        np.mean(np.square(np.log1p(store_one["prediction"]) - np.log1p(store_one["sales"])))
    )
    expected_mae = np.mean(np.abs(store_one["prediction"] - store_one["sales"]))
    expected_wape = np.abs(store_one["prediction"] - store_one["sales"]).sum() / store_one["sales"].sum()

    assert scores.loc[1, "rmsle"] == pytest.approx(expected_rmsle)
    assert scores.loc[1, "mae"] == pytest.approx(expected_mae)
    assert scores.loc[1, "wape"] == pytest.approx(expected_wape)
    assert scores.loc[1, "series_count"] == 2
    assert scores.loc[1, "fold_count"] == 4


def test_readiness_is_attached_posthoc_without_changing_oof_values_or_rows() -> None:
    oof = _oof()
    labeled = attach_readiness_labels(oof, _readiness())

    assert len(labeled) == len(oof)
    pd.testing.assert_frame_equal(labeled[oof.columns], oof)
    assert labeled["readiness_class"].notna().all()
    assert set(score_segments(labeled, ["readiness_class"])["readiness_class"]) == {
        "Intermittent demand", "High volatility", "Ready"
    }


def test_overlapping_failure_flags_are_scored_independently() -> None:
    labeled = attach_readiness_labels(_oof(), _readiness())
    flags = score_failure_flags(labeled)

    assert flags["risk_flag"].nunique() == 4
    assert set(flags["flag_active"]) == {0, 1}
    active = flags.loc[flags["flag_active"].eq(1)].set_index("risk_flag")
    assert active.loc["is_high_volatility", "series_count"] == 1
    assert active.loc["is_promotion_dependent", "series_count"] == 1


def test_oof_validation_rejects_duplicate_or_incomplete_folds() -> None:
    oof = _oof()
    validate_oof_predictions(oof)
    with pytest.raises(ValueError, match="unique"):
        validate_oof_predictions(pd.concat([oof, oof.iloc[[0]]], ignore_index=True))
    with pytest.raises(ValueError, match="exactly 4"):
        validate_oof_predictions(oof.loc[oof["fold"].ne(4)])


def test_error_analysis_loads_readiness_only_after_oof_creation() -> None:
    source = (PROJECT_ROOT / "notebooks" / "17_forecast_error_analysis.py").read_text(
        encoding="utf-8"
    )
    main_body = source[source.index("def main()") :]
    oof_position = min(
        position for position in [main_body.find("pd.read_parquet(OOF_PATH)"), main_body.find("reproduce_oof_predictions()")]
        if position >= 0
    )
    readiness_position = main_body.find('pd.read_csv(TABLES_DIR / "forecast_readiness.csv")')

    reproduction_source = source[
        source.index("def reproduce_oof_predictions") : source.index("def markdown_table")
    ]
    assert "forecast_readiness.csv" not in reproduction_source
    assert "attach_readiness_labels" not in reproduction_source
    assert oof_position < readiness_position
    assert "load_test" not in source
