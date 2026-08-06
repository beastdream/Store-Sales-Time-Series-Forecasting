# %% [markdown]
# # Store-Family Forecast Readiness
#
# This notebook profiles readiness only. It does not train, fit, select, or
# evaluate any forecasting model.

# %%
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED, REPORTS_DIR, TABLES_DIR

OUTPUT_PATH = TABLES_DIR / "forecast_readiness.csv"
REPORT_PATH = REPORTS_DIR / "forecast_readiness.md"
INSUFFICIENT_HISTORY_DAYS = 365
MIN_ACTIVE_DAYS = 90
READY_HISTORY_DAYS = 730
ISSUE_CLASSES = {
    "Intermittent demand",
    "Insufficient history",
    "High volatility",
    "Promotion dependent",
}
RISK_FLAG_COLUMNS = [
    "is_insufficient_history",
    "is_intermittent",
    "is_promotion_dependent",
    "is_high_volatility",
]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide numeric series and return NaN for zero denominators."""
    result = numerator.astype("float64").div(
        denominator.astype("float64").replace(0, np.nan)
    )
    return result.where(np.isfinite(result))


def load_sources() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load processed facts/dimensions and previously generated EDA tables."""
    fact = pd.read_parquet(
        DATA_PROCESSED / "fact_daily_sales.parquet",
        columns=["date_key", "store_key", "family_key", "sales", "is_promotion"],
    )
    dates = pd.read_parquet(
        DATA_PROCESSED / "dim_date.parquet",
        columns=["date_key", "full_date"],
    )
    stores = pd.read_parquet(DATA_PROCESSED / "dim_store.parquet")
    families = pd.read_parquet(DATA_PROCESSED / "dim_family.parquet")
    family_performance = pd.read_csv(TABLES_DIR / "family_performance.csv")
    store_performance = pd.read_csv(TABLES_DIR / "store_performance.csv")
    anomalies = pd.read_csv(TABLES_DIR / "sales_anomalies.csv")
    return (
        fact,
        dates,
        stores,
        families,
        family_performance,
        store_performance,
        anomalies,
    )


def build_series_metrics(
    fact: pd.DataFrame,
    dates: pd.DataFrame,
    stores: pd.DataFrame,
    families: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate metrics inside each series' first-to-last positive-sales window."""
    enriched = fact.merge(dates, on="date_key", validate="many_to_one")
    grain = ["store_key", "family_key"]
    all_series = (
        stores[["store_key", "store_nbr", "city", "state", "store_type", "cluster"]]
        .merge(families[["family_key", "family"]], how="cross")
    )
    active = (
        enriched.loc[enriched["sales"].gt(0)]
        .groupby(grain, as_index=False, observed=True)
        .agg(
            history_start=("full_date", "min"),
            history_end=("full_date", "max"),
            active_days=("full_date", "nunique"),
        )
    )
    with_windows = enriched.merge(active, on=grain, how="left", validate="many_to_one")
    in_window = with_windows.loc[
        with_windows["full_date"].ge(with_windows["history_start"])
        & with_windows["full_date"].le(with_windows["history_end"])
    ]
    metrics = (
        in_window.groupby(grain, as_index=False, observed=True)
        .agg(
            observed_period_count=("full_date", "nunique"),
            average_sales=("sales", "mean"),
            sales_std=("sales", "std"),
            zero_sales_rate=("sales", lambda values: values.eq(0).mean()),
            promotion_rate=("is_promotion", "mean"),
        )
    )
    result = (
        all_series.merge(active, on=grain, how="left", validate="one_to_one")
        .merge(metrics, on=grain, how="left", validate="one_to_one")
    )
    result["history_length"] = (
        result["history_end"] - result["history_start"]
    ).dt.days.add(1)
    inactive = result["history_start"].isna()
    result.loc[inactive, "history_length"] = 0
    result.loc[inactive, "active_days"] = 0
    result.loc[inactive, "observed_period_count"] = 0
    result.loc[inactive, "average_sales"] = 0.0
    result.loc[inactive, "sales_std"] = 0.0
    result.loc[inactive, "zero_sales_rate"] = 1.0
    result.loc[inactive, "promotion_rate"] = 0.0
    result["history_length"] = result["history_length"].astype("int64")
    result["active_days"] = result["active_days"].astype("int64")
    result["observed_period_count"] = result["observed_period_count"].astype("int64")
    result["missing_period_count"] = (
        result["history_length"] - result["observed_period_count"]
    ).clip(lower=0)
    result["coefficient_of_variation"] = safe_divide(
        result["sales_std"], result["average_sales"]
    )
    result["has_positive_sales"] = (~inactive).astype("int8")
    if len(result) != len(stores) * len(families):
        raise AssertionError("Store-family matrix is incomplete")
    if result.duplicated(["store_nbr", "family"]).any():
        raise AssertionError("Store-family readiness grain is not unique")
    return result


def classify_series(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Apply ordered business and data-derived readiness rules."""
    eligible = metrics.loc[
        metrics["history_length"].ge(INSUFFICIENT_HISTORY_DAYS)
        & metrics["active_days"].ge(MIN_ACTIVE_DAYS)
    ]
    thresholds = {
        "zero_sales_rate_median": float(eligible["zero_sales_rate"].median()),
        "zero_sales_rate_q75": float(eligible["zero_sales_rate"].quantile(0.75)),
        "cv_median": float(eligible["coefficient_of_variation"].median()),
        "cv_q75": float(eligible["coefficient_of_variation"].quantile(0.75)),
        "promotion_rate_q75": float(eligible["promotion_rate"].quantile(0.75)),
        "missing_period_count_q75": float(
            eligible["missing_period_count"].quantile(0.75)
        ),
    }
    insufficient = (
        metrics["history_length"].lt(INSUFFICIENT_HISTORY_DAYS)
        | metrics["active_days"].lt(MIN_ACTIVE_DAYS)
    )
    intermittent = metrics["zero_sales_rate"].ge(thresholds["zero_sales_rate_q75"])
    promotion_dependent = metrics["promotion_rate"].ge(
        thresholds["promotion_rate_q75"]
    )
    high_volatility = metrics["coefficient_of_variation"].ge(thresholds["cv_q75"])
    ready_criteria = (
        metrics["history_length"].ge(READY_HISTORY_DAYS)
        & metrics["zero_sales_rate"].le(thresholds["zero_sales_rate_median"])
        & metrics["coefficient_of_variation"].le(thresholds["cv_median"])
        & metrics["missing_period_count"].le(
            thresholds["missing_period_count_q75"]
        )
    )
    result = metrics.copy()
    result["is_insufficient_history"] = insufficient.astype("uint8")
    result["is_intermittent"] = intermittent.astype("uint8")
    result["is_promotion_dependent"] = promotion_dependent.astype("uint8")
    result["is_high_volatility"] = high_volatility.astype("uint8")
    result["risk_flag_count"] = result[RISK_FLAG_COLUMNS].sum(axis=1).astype("uint8")
    result["is_ready"] = (
        ready_criteria & result["risk_flag_count"].eq(0)
    ).astype("uint8")
    result["readiness_class"] = np.select(
        [
            insufficient,
            intermittent,
            promotion_dependent,
            high_volatility,
            result["is_ready"].eq(1),
        ],
        [
            "Insufficient history",
            "Intermittent demand",
            "Promotion dependent",
            "High volatility",
            "Ready",
        ],
        default="Ready with caution",
    )
    result["classification_rule"] = np.select(
        [
            insufficient,
            intermittent,
            promotion_dependent,
            high_volatility,
            result["is_ready"].eq(1),
        ],
        [
            f"history_length < {INSUFFICIENT_HISTORY_DAYS} or active_days < {MIN_ACTIVE_DAYS}",
            "zero_sales_rate >= eligible-series Q75",
            "promotion_rate >= eligible-series Q75",
            "coefficient_of_variation >= eligible-series Q75",
            (
                f"history_length >= {READY_HISTORY_DAYS}, zero-sales <= median, "
                "CV <= median, and missing periods <= Q75"
            ),
        ],
        default="Sufficient history without an extreme Q75 risk flag",
    )
    for name, value in thresholds.items():
        result[name] = value
    return result, thresholds


def _overlap_summary(readiness: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Rank dimensions by series carrying two or more independent risk flags."""
    working = readiness.assign(
        has_overlapping_risks=readiness["risk_flag_count"].ge(2)
    )
    summary = (
        working.groupby(dimension, as_index=False, observed=True)
        .agg(
            series_count=("risk_flag_count", "size"),
            overlapping_risk_series=("has_overlapping_risks", "sum"),
            total_risk_flags=("risk_flag_count", "sum"),
            average_risk_flags=("risk_flag_count", "mean"),
            maximum_risk_flags=("risk_flag_count", "max"),
        )
    )
    summary["overlap_rate"] = safe_divide(
        summary["overlapping_risk_series"], summary["series_count"]
    )
    return summary.sort_values(
        ["overlapping_risk_series", "total_risk_flags", "overlap_rate"],
        ascending=False,
    )


def _problem_summary(
    readiness: pd.DataFrame, dimension: str
) -> pd.DataFrame:
    """Rank dimensions by count and rate of material readiness issues."""
    working = readiness.assign(is_issue=readiness["readiness_class"].isin(ISSUE_CLASSES))
    summary = (
        working.groupby(dimension, as_index=False, observed=True)
        .agg(
            series_count=("readiness_class", "size"),
            issue_count=("is_issue", "sum"),
        )
    )
    summary["issue_rate"] = safe_divide(summary["issue_count"], summary["series_count"])
    dominant = (
        working.loc[working["is_issue"]]
        .groupby([dimension, "readiness_class"], observed=True)
        .size()
        .rename("count")
        .reset_index()
        .sort_values([dimension, "count"], ascending=[True, False])
        .drop_duplicates(dimension)
        .rename(columns={"readiness_class": "dominant_issue"})
    )
    return summary.merge(
        dominant[[dimension, "dominant_issue"]],
        on=dimension,
        how="left",
        validate="one_to_one",
    ).sort_values(["issue_count", "issue_rate"], ascending=False)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact DataFrame without requiring a Markdown extension."""
    columns = frame.columns.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    readiness: pd.DataFrame,
    thresholds: dict[str, float],
    family_performance: pd.DataFrame,
    store_performance: pd.DataFrame,
    anomalies: pd.DataFrame,
) -> None:
    """Write an evidence-based readiness report using generated EDA tables."""
    class_counts = (
        readiness["readiness_class"]
        .value_counts()
        .reindex(
            [
                "Ready",
                "Ready with caution",
                "Intermittent demand",
                "Insufficient history",
                "High volatility",
                "Promotion dependent",
            ],
            fill_value=0,
        )
        .rename_axis("Nhóm")
        .reset_index(name="Số chuỗi")
    )
    class_counts["Tỷ lệ"] = (
        class_counts["Số chuỗi"] / len(readiness) * 100
    ).map(lambda value: f"{value:.1f}%")

    flag_labels = {
        "is_insufficient_history": "Insufficient history",
        "is_intermittent": "Intermittent demand",
        "is_promotion_dependent": "Promotion dependent",
        "is_high_volatility": "High volatility",
        "is_ready": "Ready (no serious risk flags)",
    }
    flag_counts = pd.DataFrame(
        {
            "Flag": [flag_labels[column] for column in [*RISK_FLAG_COLUMNS, "is_ready"]],
            "Số chuỗi": [
                int(readiness[column].sum())
                for column in [*RISK_FLAG_COLUMNS, "is_ready"]
            ],
        }
    )
    flag_counts["Tỷ lệ"] = (flag_counts["Số chuỗi"] / len(readiness) * 100).map(
        lambda value: f"{value:.1f}%"
    )
    risk_distribution = (
        readiness["risk_flag_count"]
        .map(lambda value: str(value) if value < 3 else "3+")
        .value_counts()
        .reindex(["0", "1", "2", "3+"], fill_value=0)
        .rename_axis("Số risk flags")
        .reset_index(name="Số chuỗi")
    )
    risk_distribution["Tỷ lệ"] = (
        risk_distribution["Số chuỗi"] / len(readiness) * 100
    ).map(lambda value: f"{value:.1f}%")

    family_overlap = _overlap_summary(readiness, "family").head(10).copy()
    store_overlap = _overlap_summary(readiness, "store_nbr").head(10).copy()
    for frame in (family_overlap, store_overlap):
        frame["overlap_rate"] = frame["overlap_rate"].map(lambda value: f"{value:.1%}")
        frame["average_risk_flags"] = frame["average_risk_flags"].map(
            lambda value: f"{value:.2f}"
        )

    family_issues = _problem_summary(readiness, "family").head(10)
    family_context = family_performance[
        ["family", "zero_sales_rate", "coefficient_of_variation", "promotion_rate"]
    ]
    family_issues = family_issues.merge(
        family_context, on="family", validate="one_to_one"
    )
    family_display = family_issues[
        [
            "family",
            "issue_count",
            "issue_rate",
            "dominant_issue",
            "zero_sales_rate",
            "coefficient_of_variation",
            "promotion_rate",
        ]
    ].copy()
    for column in ["issue_rate", "zero_sales_rate", "promotion_rate"]:
        family_display[column] = family_display[column].map(lambda value: f"{value:.1%}")
    family_display["coefficient_of_variation"] = family_display[
        "coefficient_of_variation"
    ].map(lambda value: f"{value:.3f}")

    store_issues = _problem_summary(readiness, "store_nbr").head(10)
    anomaly_counts = (
        anomalies.loc[anomalies["analysis_level"].eq("store_day")]
        .groupby("store_nbr", observed=True)
        .size()
        .rename("anomaly_count")
        .reset_index()
    )
    store_context = store_performance[
        ["store_nbr", "average_daily_sales", "coefficient_of_variation"]
    ]
    store_issues = (
        store_issues.merge(store_context, on="store_nbr", validate="one_to_one")
        .merge(anomaly_counts, on="store_nbr", how="left", validate="one_to_one")
    )
    store_issues["anomaly_count"] = store_issues["anomaly_count"].fillna(0).astype(int)
    store_display = store_issues[
        [
            "store_nbr",
            "issue_count",
            "issue_rate",
            "dominant_issue",
            "average_daily_sales",
            "coefficient_of_variation",
            "anomaly_count",
        ]
    ].copy()
    store_display["issue_rate"] = store_display["issue_rate"].map(
        lambda value: f"{value:.1%}"
    )
    store_display["average_daily_sales"] = store_display[
        "average_daily_sales"
    ].map(lambda value: f"{value:,.2f}")
    store_display["coefficient_of_variation"] = store_display[
        "coefficient_of_variation"
    ].map(lambda value: f"{value:.3f}")

    lines = [
        "# Forecast Readiness",
        "",
        "> Phạm vi của bước này chỉ là đánh giá dữ liệu. Chưa huấn luyện, chọn hoặc đánh giá model.",
        "",
        "## Dữ liệu và grain",
        "",
        f"Bảng chi tiết có `{len(readiness):,}` chuỗi tại grain `store_nbr × family`. "
        "Metrics được tính trong cửa sổ từ ngày sales dương đầu tiên đến ngày sales "
        "dương cuối cùng; các ngày zero sales bên trong cửa sổ vẫn được giữ. Báo cáo "
        "đối chiếu thêm các bảng EDA đã tạo: `family_performance.csv`, "
        "`store_performance.csv` và `sales_anomalies.csv`.",
        "",
        "## Ngưỡng sử dụng",
        "",
        f"- Business rule: ít hơn `{INSUFFICIENT_HISTORY_DAYS}` ngày lịch sử hoặc "
        f"`{MIN_ACTIVE_DAYS}` active days là **Insufficient history**. Một năm lịch sử "
        "nhằm phủ ít nhất một chu kỳ mùa vụ; 90 active days là mức tối thiểu để có "
        "đủ quan sát sales dương cho pattern tuần/tháng.",
        f"- Business rule: **Ready** cần ít nhất `{READY_HISTORY_DAYS}` ngày, tương "
        "đương khoảng hai chu kỳ năm.",
        f"- Median zero-sales rate: `{thresholds['zero_sales_rate_median']:.4%}`; "
        f"Q75: `{thresholds['zero_sales_rate_q75']:.4%}`.",
        f"- Median coefficient of variation: `{thresholds['cv_median']:.4f}`; "
        f"Q75: `{thresholds['cv_q75']:.4f}`.",
        f"- Promotion-rate Q75: `{thresholds['promotion_rate_q75']:.4%}`.",
        f"- Missing-period Q75: `{thresholds['missing_period_count_q75']:.0f}` ngày.",
        "",
        "Các median/Q75 chỉ được tính trên chuỗi đã có ít nhất một năm lịch sử và "
        "90 active days, để chuỗi chưa hoạt động không làm méo ngưỡng.",
        "",
        "## Quy tắc phân loại",
        "",
        "Các risk flag được tính độc lập nên một chuỗi có thể đồng thời intermittent, "
        "promotion dependent và high volatility. `risk_flag_count` là tổng bốn risk "
        "flags và không tính `is_ready`. `is_ready = 1` chỉ khi chuỗi đạt rule Ready "
        "và không có risk flag nghiêm trọng.",
        "",
        "`readiness_class` vẫn là nhãn chính duy nhất. Khi nhiều rule cùng đúng, nhãn "
        "chính được chọn theo thứ tự ưu tiên đã công bố sau:",
        "",
        f"1. **Insufficient history:** history < {INSUFFICIENT_HISTORY_DAYS} ngày hoặc active days < {MIN_ACTIVE_DAYS}.",
        "2. **Intermittent demand:** zero-sales rate ≥ Q75.",
        "3. **Promotion dependent:** promotion rate ≥ Q75.",
        "4. **High volatility:** coefficient of variation ≥ Q75.",
        f"5. **Ready:** history ≥ {READY_HISTORY_DAYS}, zero-sales rate ≤ median, "
        "CV ≤ median và missing periods ≤ Q75.",
        "6. **Ready with caution:** đủ lịch sử nhưng chưa đạt toàn bộ điều kiện Ready "
        "và không rơi vào các risk group Q75 ở trên.",
        "",
        "## Phân bố readiness",
        "",
        _markdown_table(class_counts),
        "",
        "## Phân bố overlapping flags",
        "",
        "### Số chuỗi theo từng flag độc lập",
        "",
        _markdown_table(flag_counts),
        "",
        "### Số chuỗi theo số lượng risk flags",
        "",
        _markdown_table(risk_distribution),
        "",
        "### Family có nhiều overlapping risks",
        "",
        _markdown_table(family_overlap),
        "",
        "### Store có nhiều overlapping risks",
        "",
        _markdown_table(store_overlap),
        "",
        "## Family thường gặp vấn đề",
        "",
        "`issue_count` gồm Intermittent demand, Insufficient history, High volatility "
        "và Promotion dependent. Các metric family-level lấy trực tiếp từ "
        "`family_performance.csv`.",
        "",
        _markdown_table(family_display),
        "",
        "## Store thường gặp vấn đề",
        "",
        "Store metrics lấy từ `store_performance.csv`; anomaly count lấy từ "
        "`sales_anomalies.csv` và chỉ là review flag, không phải lỗi dữ liệu.",
        "",
        _markdown_table(store_display),
        "",
        "## Ảnh hưởng tới forecasting",
        "",
        "- **Ready:** phù hợp cho baseline/model chuẩn sau khi thiết kế validation theo thời gian.",
        "- **Ready with caution:** cần kiểm tra thêm scale, recent regime và feature availability.",
        "- **Intermittent demand:** nhiều zero; metric như MAE đơn thuần có thể che khuất "
        "khả năng dự báo occurrence. Nên cân nhắc intermittent-demand methods hoặc "
        "mô hình hai giai đoạn occurrence/size.",
        "- **Insufficient history:** chưa phủ đủ mùa vụ; nên dùng pooled/global model, "
        "hierarchical information hoặc benchmark đơn giản thay vì fit riêng chuỗi.",
        "- **High volatility:** prediction intervals cần rộng hơn; validation nhiều "
        "fold và robust loss có thể quan trọng hơn point accuracy đơn lẻ.",
        "- **Promotion dependent:** forecast cần promotion plan tương lai đáng tin cậy; "
        "kịch bản thiếu promotion feature phải được đánh giá riêng.",
        "",
        "## Bước tiếp theo",
        "",
        "Chưa có model nào được huấn luyện trong bước này. Bước sau mới nên xác định "
        "forecast horizon, temporal split, baseline và metric theo từng readiness group.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


# %% [markdown]
# ## Run readiness assessment

# %%
def main() -> None:
    """Build detailed readiness data and its Markdown assessment."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (
        fact,
        dates,
        stores,
        families,
        family_performance,
        store_performance,
        anomalies,
    ) = load_sources()
    metrics = build_series_metrics(fact, dates, stores, families)
    readiness, thresholds = classify_series(metrics)
    output_columns = [
        "store_nbr",
        "family",
        "city",
        "state",
        "store_type",
        "cluster",
        "history_start",
        "history_end",
        "history_length",
        "active_days",
        "zero_sales_rate",
        "average_sales",
        "sales_std",
        "coefficient_of_variation",
        "promotion_rate",
        "observed_period_count",
        "missing_period_count",
        "has_positive_sales",
        "is_insufficient_history",
        "is_intermittent",
        "is_promotion_dependent",
        "is_high_volatility",
        "is_ready",
        "risk_flag_count",
        "readiness_class",
        "classification_rule",
        "zero_sales_rate_median",
        "zero_sales_rate_q75",
        "cv_median",
        "cv_q75",
        "promotion_rate_q75",
        "missing_period_count_q75",
    ]
    readiness[output_columns].sort_values(["store_nbr", "family"]).to_csv(
        OUTPUT_PATH, index=False
    )
    write_report(
        readiness,
        thresholds,
        family_performance,
        store_performance,
        anomalies,
    )
    print(readiness["readiness_class"].value_counts().to_string())
    print("\nIndependent flags:")
    print(readiness[[*RISK_FLAG_COLUMNS, "is_ready"]].sum().to_string())
    print("\nRisk flag count:")
    print(readiness["risk_flag_count"].value_counts().sort_index().to_string())
    print(f"Table: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
