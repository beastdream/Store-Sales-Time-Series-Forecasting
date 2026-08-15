# %% [markdown]
# # Forecast Problem Definition and Feature Availability
#
# This notebook verifies the forecasting contract and prediction-time feature
# availability. It does not build features, train a model, or create predictions.

# %%
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED, DATA_RAW, REPORTS_DIR, TABLES_DIR


REPORT_PATH = REPORTS_DIR / "modeling" / "forecast_problem_definition.md"


def load_sources() -> dict[str, pd.DataFrame]:
    """Load only sources needed to verify the forecast and availability contract."""
    return {
        "train": pd.read_csv(DATA_RAW / "train.csv", parse_dates=["date"]),
        "test": pd.read_csv(DATA_RAW / "test.csv", parse_dates=["date"]),
        "stores": pd.read_csv(DATA_RAW / "stores.csv"),
        "holidays": pd.read_csv(
            DATA_RAW / "holidays_events.csv", parse_dates=["date"]
        ),
        "oil": pd.read_csv(DATA_RAW / "oil.csv", parse_dates=["date"]),
        "transactions": pd.read_csv(
            DATA_RAW / "transactions.csv", parse_dates=["date"]
        ),
        "dim_store_date": pd.read_parquet(
            DATA_PROCESSED / "dim_store_date.parquet",
            columns=["date_key", "store_key", "has_sales_observation"],
        ),
        "readiness": pd.read_csv(TABLES_DIR / "forecast_readiness.csv"),
        "anomalies": pd.read_csv(TABLES_DIR / "sales_anomalies.csv"),
    }


def build_forecast_contract(sources: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Derive and validate the competition forecasting contract from source data."""
    train = sources["train"]
    test = sources["test"]
    if "sales" not in train.columns or "sales" in test.columns:
        raise AssertionError("sales must exist only as the historical training target")
    if "id" not in test.columns or not test["id"].is_unique:
        raise AssertionError("test id must exist and be unique")
    grain = ["date", "store_nbr", "family"]
    if train.duplicated(grain).any() or test.duplicated(grain).any():
        raise AssertionError("train/test forecast grain must be unique")

    historical_start = train["date"].min().normalize()
    last_actual_date = train["date"].max().normalize()
    forecast_start = test["date"].min().normalize()
    forecast_end = test["date"].max().normalize()
    horizon = int(test["date"].nunique())
    store_count = int(train["store_nbr"].nunique())
    family_count = int(train["family"].nunique())
    series_count = int(train[["store_nbr", "family"]].drop_duplicates().shape[0])
    expected_predictions = store_count * family_count * horizon

    if forecast_start != last_actual_date + pd.Timedelta(days=1):
        raise AssertionError("forecast period must begin after the final actual date")
    if len(pd.date_range(forecast_start, forecast_end, freq="D")) != horizon:
        raise AssertionError("test forecast dates must form a continuous calendar")
    if len(test) != expected_predictions:
        raise AssertionError("test rows must equal stores × families × horizon")
    if series_count != store_count * family_count:
        raise AssertionError("historical store-family coverage is incomplete")

    return {
        "forecast_target": "sales",
        "forecast_grain": "store × family × day",
        "historical_start": historical_start,
        "last_actual_date": last_actual_date,
        "forecast_start": forecast_start,
        "forecast_end": forecast_end,
        "forecast_horizon_days": horizon,
        "store_count": store_count,
        "family_count": family_count,
        "series_count": series_count,
        "expected_predictions": expected_predictions,
        "test_id_column": "id",
    }


def build_feature_availability(
    sources: dict[str, pd.DataFrame],
    contract: dict[str, object],
) -> pd.DataFrame:
    """Classify feature availability at the historical forecast origin."""
    test = sources["test"]
    stores = sources["stores"]
    holidays = sources["holidays"]
    oil = sources["oil"]
    transactions = sources["transactions"]
    readiness = sources["readiness"]
    anomalies = sources["anomalies"]
    test_start = pd.Timestamp(contract["forecast_start"])
    test_end = pd.Timestamp(contract["forecast_end"])

    onpromotion_known = (
        "onpromotion" in test.columns and test["onpromotion"].notna().all()
    )
    stores_cover_test = test["store_nbr"].isin(stores["store_nbr"]).all()
    holiday_source_reaches_horizon = holidays["date"].max() >= test_end
    oil_source_reaches_horizon = oil["date"].max() >= test_end
    future_transactions_available = transactions["date"].max() >= test_start

    rows = [
        {
            "Feature": "Calendar",
            "Source": "test.date / DimDate",
            "Available historically?": "Yes",
            "Available for future forecast horizon?": "Yes",
            "Allowed in initial model?": "Yes",
            "Leakage risk?": "Low",
            "Notes": "Deterministic from the prediction date; derive without target data.",
        },
        {
            "Feature": "Store metadata",
            "Source": "stores.csv / DimStore",
            "Available historically?": "Yes",
            "Available for future forecast horizon?": "Yes" if stores_cover_test else "No",
            "Allowed in initial model?": "Yes" if stores_cover_test else "No",
            "Leakage risk?": "Low",
            "Notes": "Static city, state, type, and cluster cover all test stores.",
        },
        {
            "Feature": "Family",
            "Source": "train.csv, test.csv / DimFamily",
            "Available historically?": "Yes",
            "Available for future forecast horizon?": "Yes",
            "Allowed in initial model?": "Yes",
            "Leakage risk?": "Low",
            "Notes": "Product-family identity is part of the forecast grain.",
        },
        {
            "Feature": "onpromotion",
            "Source": "train.csv and Kaggle test.csv",
            "Available historically?": "Yes",
            "Available for future forecast horizon?": "Yes" if onpromotion_known else "No",
            "Allowed in initial model?": "Yes" if onpromotion_known else "No",
            "Leakage risk?": "Low for competition; deployment caveat",
            "Notes": "Future-known in the supplied Kaggle test. Production use requires an available promotion plan.",
        },
        {
            "Feature": "Holiday / event",
            "Source": "holidays_events.csv / DimStoreDate",
            "Available historically?": "Yes",
            "Available for future forecast horizon?": "Yes" if holiday_source_reaches_horizon else "No",
            "Allowed in initial model?": "Yes" if holiday_source_reaches_horizon else "No",
            "Leakage risk?": "Low to medium",
            "Notes": "Calendar-known events cover the horizon; preserve national/regional/local and transfer rules.",
        },
        {
            "Feature": "Oil",
            "Source": "oil.csv / FactOilPrice",
            "Available historically?": "Yes",
            "Available for future forecast horizon?": "Competition-known: Yes" if oil_source_reaches_horizon else "No",
            "Allowed in initial model?": "Conditional",
            "Leakage risk?": "Medium to high",
            "Notes": "Competition data reaches the horizon, but production-realistic future oil availability differs. Any interpolation must be causal within each fold.",
        },
        {
            "Feature": "Transactions",
            "Source": "transactions.csv / FactStoreTransactions",
            "Available historically?": "Yes",
            "Available for future forecast horizon?": "Yes" if future_transactions_available else "No",
            "Allowed in initial model?": "Historical lags only",
            "Leakage risk?": "High for current-day values",
            "Notes": "Future transactions are not supplied. Do not use current-day transactions unless they are forecast separately.",
        },
        {
            "Feature": "Historical sales lags",
            "Source": "train.sales / FactDailySales",
            "Available historically?": "Yes",
            "Available for future forecast horizon?": "At forecast origin only",
            "Allowed in initial model?": "Yes",
            "Leakage risk?": "High if not shifted",
            "Notes": "Use shift before rolling calculations; multi-step forecasts cannot read future actual sales.",
        },
        {
            "Feature": "Observation status",
            "Source": "DimStoreDate.has_sales_observation",
            "Available historically?": "Yes",
            "Available for future forecast horizon?": "Known as no actual observation",
            "Allowed in initial model?": "Historical context only",
            "Leakage risk?": "Medium",
            "Notes": "Missing observation is not zero sales. Preserve the flag and never impute target zero automatically.",
        },
        {
            "Feature": "ForecastReadiness outputs",
            "Source": "forecast_readiness.csv",
            "Available historically?": "Yes",
            "Available for future forecast horizon?": "Artifact exists",
            "Allowed in initial model?": "No",
            "Leakage risk?": "High in temporal backtests",
            "Notes": f"Computed over the full historical window for {len(readiness):,} series; do not use automatically unless recalculated causally at every cutoff.",
        },
        {
            "Feature": "SalesAnomalies outputs",
            "Source": "sales_anomalies.csv",
            "Available historically?": "Yes",
            "Available for future forecast horizon?": "Artifact exists",
            "Allowed in initial model?": "No",
            "Leakage risk?": "High in temporal backtests",
            "Notes": f"Full-history review output ({len(anomalies):,} rows); do not use automatically unless recomputed causally at every cutoff.",
        },
    ]
    return pd.DataFrame(rows)


def audit_missing_observations(
    sources: dict[str, pd.DataFrame],
    contract: dict[str, object],
) -> dict[str, object]:
    """Quantify missing sales dates without converting them to zero targets."""
    train = sources["train"]
    store_date = sources["dim_store_date"]
    historical_start = pd.Timestamp(contract["historical_start"])
    last_actual_date = pd.Timestamp(contract["last_actual_date"])
    forecast_start = pd.Timestamp(contract["forecast_start"])
    forecast_end = pd.Timestamp(contract["forecast_end"])
    observed_dates = pd.DatetimeIndex(train["date"].drop_duplicates())
    missing_dates = pd.date_range(
        historical_start, last_actual_date, freq="D"
    ).difference(observed_dates)

    historical_key_start = int(historical_start.strftime("%Y%m%d"))
    historical_key_end = int(last_actual_date.strftime("%Y%m%d"))
    forecast_key_start = int(forecast_start.strftime("%Y%m%d"))
    forecast_key_end = int(forecast_end.strftime("%Y%m%d"))
    historical_store_dates = store_date.loc[
        store_date["date_key"].between(historical_key_start, historical_key_end)
    ]
    forecast_store_dates = store_date.loc[
        store_date["date_key"].between(forecast_key_start, forecast_key_end)
    ]
    if set(store_date["has_sales_observation"].unique()) != {0, 1}:
        raise AssertionError("has_sales_observation must remain a binary status")

    return {
        "missing_calendar_dates": tuple(missing_dates),
        "historical_store_days": len(historical_store_dates),
        "historical_store_days_without_observation": int(
            historical_store_dates["has_sales_observation"].eq(0).sum()
        ),
        "forecast_store_days": len(forecast_store_dates),
        "forecast_store_days_without_actual": int(
            forecast_store_dates["has_sales_observation"].eq(0).sum()
        ),
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame without requiring an optional Markdown dependency."""
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(
    contract: dict[str, object],
    feature_audit: pd.DataFrame,
    missing: dict[str, object],
) -> str:
    """Create the verified forecast problem definition report."""
    contract_rows = pd.DataFrame(
        [
            ("Forecast target", contract["forecast_target"]),
            ("Forecast grain", contract["forecast_grain"]),
            ("Historical period", f"{contract['historical_start']:%Y-%m-%d} through {contract['last_actual_date']:%Y-%m-%d}"),
            ("Final actual-sales date", f"{contract['last_actual_date']:%Y-%m-%d}"),
            ("Test forecast period", f"{contract['forecast_start']:%Y-%m-%d} through {contract['forecast_end']:%Y-%m-%d}"),
            ("Forecast horizon", f"{contract['forecast_horizon_days']} calendar days"),
            ("Stores", f"{contract['store_count']:,}"),
            ("Product families", f"{contract['family_count']:,}"),
            ("Store-family series", f"{contract['series_count']:,}"),
            ("Expected predictions", f"{contract['expected_predictions']:,}"),
            ("Test ID", str(contract["test_id_column"])),
        ],
        columns=["Contract item", "Verified value"],
    )
    missing_dates = missing["missing_calendar_dates"]
    missing_text = ", ".join(date.strftime("%Y-%m-%d") for date in missing_dates)
    return "\n".join(
        [
            "# Forecast Problem Definition",
            "",
            "> Scope: this entrypoint audits only the forecasting contract and feature availability; it does not itself train a model or create predictions. Later validated pipeline stages and artifacts are documented in the project README.",
            "",
            "## Verified forecasting contract",
            "",
            _markdown_table(contract_rows),
            "",
            f"The supplied test is a complete `{contract['store_count']} × {contract['family_count']} × {contract['forecast_horizon_days']}` grid, giving `{contract['expected_predictions']:,}` unique `id` rows. `sales` exists in train and is absent from test.",
            "",
            "## Feature availability audit",
            "",
            _markdown_table(feature_audit),
            "",
            "## Initial-model availability boundary",
            "",
            "- Allowed directly: calendar, store metadata, family, and Kaggle-test `onpromotion`.",
            "- Allowed with causal construction: historical sales lags/rolling features and historical transaction lags.",
            "- Conditional: holiday/event context and oil, subject to forecast-origin and production-availability policies.",
            "- Not allowed automatically: current-day future transactions, full-window ForecastReadiness outputs, and full-window SalesAnomalies outputs.",
            "",
            "## Missing-date and observation-status rule",
            "",
            "**Missing observation date != zero sales.**",
            "",
            f"The historical calendar contains `{len(missing_dates)}` dates with no sales observation: {missing_text}. Across the historical date-store grid, `{missing['historical_store_days_without_observation']:,}` of `{missing['historical_store_days']:,}` store-days have `has_sales_observation = 0`. The 16-day forecast grid has `{missing['forecast_store_days_without_actual']:,}` of `{missing['forecast_store_days']:,}` store-days without an actual sales observation by construction.",
            "",
            "`has_sales_observation` must be preserved (or represented by an equivalent explicit status). A missing row must not be silently materialized as `sales = 0`; observed zero Sales Volume and absent observations have different meanings.",
            "",
            "## Oil: competition versus production",
            "",
            "The competition oil source contains dated information through the test end, so a competition experiment may evaluate it under explicit causal imputation rules. A production forecast may not know future oil prices at forecast origin. Production use must instead define lagged availability, an external oil forecast, or a scenario. Full-series interpolation that uses later dates is not valid inside temporal backtests.",
            "",
            "## This entrypoint's boundaries",
            "",
            "- This contract audit does not create predictions.",
            "- It does not persist a final feature dataset.",
            "- It does not train or evaluate a forecasting model.",
            "- No test target was read or inferred.",
            "",
        ]
    )


def main() -> None:
    """Run the audit and write its single Markdown output."""
    sources = load_sources()
    contract = build_forecast_contract(sources)
    feature_audit = build_feature_availability(sources, contract)
    missing = audit_missing_observations(sources, contract)
    report = build_report(contract, feature_audit, missing)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Forecast problem definition validated.")
    print(
        f"Actual: {contract['historical_start']:%Y-%m-%d} -> "
        f"{contract['last_actual_date']:%Y-%m-%d}"
    )
    print(
        f"Forecast: {contract['forecast_start']:%Y-%m-%d} -> "
        f"{contract['forecast_end']:%Y-%m-%d} "
        f"({contract['forecast_horizon_days']} days, "
        f"{contract['expected_predictions']:,} rows)"
    )
    print(
        "Missing historical observation dates: "
        f"{len(missing['missing_calendar_dates'])}"
    )
    print(f"Feature audit rows: {len(feature_audit)}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
