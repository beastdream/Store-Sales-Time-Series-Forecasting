"""Post-hoc segment error analysis for out-of-fold sales forecasts."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


GRAIN = ["date", "store_nbr", "family", "fold"]
REQUIRED_OOF_COLUMNS = [*GRAIN, "sales", "prediction"]
READINESS_LABEL_COLUMNS = [
    "store_nbr",
    "family",
    "readiness_class",
    "is_high_volatility",
    "is_intermittent",
    "is_promotion_dependent",
    "is_insufficient_history",
]


def validate_oof_predictions(oof: pd.DataFrame, expected_folds: int = 4) -> None:
    """Validate complete finite OOF predictions before any segmentation."""
    missing = [column for column in REQUIRED_OOF_COLUMNS if column not in oof]
    if missing:
        raise KeyError("OOF predictions are missing columns: " + ", ".join(missing))
    if oof.empty or oof.duplicated(GRAIN).any():
        raise ValueError("OOF predictions must have a non-empty unique fold grain")
    if oof["fold"].nunique() != expected_folds:
        raise ValueError(f"OOF predictions must cover exactly {expected_folds} folds")
    numeric = oof[["sales", "prediction"]].to_numpy(dtype="float64")
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError("OOF actuals and predictions must be finite and nonnegative")


def score_segments(oof: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    """Calculate RMSLE, MAE and WAPE at any post-hoc segment grain."""
    groups = list(group_columns)
    if not groups:
        raise ValueError("group_columns must not be empty")
    missing = [column for column in groups if column not in oof]
    if missing:
        raise KeyError("OOF predictions are missing group columns: " + ", ".join(missing))

    working = oof.copy()
    working["_series_id"] = (
        working["store_nbr"].astype(str) + "\x1f" + working["family"].astype(str)
    )
    working["squared_log_error"] = np.square(
        np.log1p(working["prediction"]) - np.log1p(working["sales"])
    )
    working["absolute_error"] = (working["sales"] - working["prediction"]).abs()
    result = (
        working.groupby(groups, as_index=False, observed=True, dropna=False)
        .agg(
            observation_count=("sales", "size"),
            series_count=("_series_id", "nunique"),
            fold_count=("fold", "nunique"),
            actual_sales=("sales", "sum"),
            predicted_sales=("prediction", "sum"),
            mean_squared_log_error=("squared_log_error", "mean"),
            mae=("absolute_error", "mean"),
            absolute_error_sum=("absolute_error", "sum"),
        )
        .reset_index(drop=True)
    )
    result["rmsle"] = np.sqrt(result.pop("mean_squared_log_error"))
    result["wape"] = result["absolute_error_sum"].div(
        result["actual_sales"].replace(0, np.nan)
    )
    return result.drop(columns="absolute_error_sum")


def attach_readiness_labels(oof: pd.DataFrame, readiness: pd.DataFrame) -> pd.DataFrame:
    """Join full-history readiness labels strictly after predictions exist."""
    missing = [column for column in READINESS_LABEL_COLUMNS if column not in readiness]
    if missing:
        raise KeyError("readiness is missing label columns: " + ", ".join(missing))
    labels = readiness[READINESS_LABEL_COLUMNS].copy()
    if labels.duplicated(["store_nbr", "family"]).any():
        raise ValueError("readiness labels must be unique by store-family")
    result = oof.merge(
        labels,
        on=["store_nbr", "family"],
        how="left",
        validate="many_to_one",
    )
    if result[READINESS_LABEL_COLUMNS[2:]].isna().any().any():
        raise ValueError("readiness labels must map every OOF store-family series")
    return result


def score_failure_flags(labeled_oof: pd.DataFrame) -> pd.DataFrame:
    """Score each overlapping readiness risk flag against its complement."""
    rows: list[pd.DataFrame] = []
    for flag in READINESS_LABEL_COLUMNS[3:]:
        segment = labeled_oof.assign(flag_active=labeled_oof[flag].astype("uint8"))
        scored = score_segments(segment, ["flag_active"])
        scored.insert(0, "risk_flag", flag)
        rows.append(scored)
    return pd.concat(rows, ignore_index=True)
