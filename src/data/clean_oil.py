"""Daily oil-price calendar cleaning and change features."""

import pandas as pd


def clean_oil(
    oil_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Build and impute a complete daily oil calendar for an inclusive date range."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("start_date must be less than or equal to end_date")

    required_columns = ["date", "dcoilwtico"]
    missing_columns = [
        column for column in required_columns if column not in oil_df.columns
    ]
    if missing_columns:
        raise KeyError(f"oil: required columns not found: {', '.join(missing_columns)}")

    oil = oil_df.loc[:, required_columns].copy(deep=True)
    oil["date"] = pd.to_datetime(oil["date"]).dt.normalize()
    if oil["date"].duplicated().any():
        raise ValueError("oil: duplicate date grain detected")

    calendar = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    cleaned = calendar.merge(oil, on="date", how="left", validate="one_to_one")
    cleaned = cleaned.rename(columns={"dcoilwtico": "oil_price"})
    cleaned["oil_was_imputed"] = cleaned["oil_price"].isna().astype("uint8")

    if cleaned["oil_price"].notna().sum() == 0:
        raise ValueError("oil: no observed oil price is available for imputation")

    cleaned["oil_price"] = (
        cleaned["oil_price"]
        .interpolate(method="linear", limit_area="inside")
        .ffill()
        .bfill()
    )
    if cleaned["oil_price"].isna().any():
        raise ValueError("oil: oil_price still contains missing values after imputation")

    cleaned["oil_change_1d"] = cleaned["oil_price"].diff(1)
    cleaned["oil_change_7d"] = cleaned["oil_price"].diff(7)
    cleaned["oil_pct_change_7d"] = cleaned["oil_price"].pct_change(
        periods=7,
        fill_method=None,
    )
    return cleaned
