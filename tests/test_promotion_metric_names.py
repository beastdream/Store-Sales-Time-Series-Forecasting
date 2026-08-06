"""Terminology contract tests for descriptive promotion metrics."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_daily_store_mart_uses_noncausal_promotion_metric_names() -> None:
    sql = (
        PROJECT_ROOT / "sql" / "marts" / "01_daily_store_performance.sql"
    ).read_text(encoding="utf-8")

    assert "sales_on_promotion_active_rows" in sql
    assert "sales_on_nonpromotion_rows" in sql
    assert "promotion_active_sales_share_proxy" in sql
    assert (
        "Promotion metrics are descriptive associations and do not establish "
        "causal effects."
    ) in sql
    assert (
        "SUM(CASE WHEN is_promotion = 1 THEN sales ELSE 0::NUMERIC END)"
        in sql
    )
    assert (
        "SUM(CASE WHEN is_promotion = 0 THEN sales ELSE 0::NUMERIC END)"
        in sql
    )


def test_legacy_promotion_metric_names_are_absent_from_repository() -> None:
    legacy_names = [
        "promoted_" + "sales",
        "non_promoted_" + "sales",
        "promotion_" + "sales_share",
        "Promoted " + "Sales",
        "Promotion " + "Sales Share",
    ]
    checked_suffixes = {".py", ".sql", ".md", ".csv", ".txt"}
    occurrences: list[str] = []

    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in checked_suffixes:
            continue
        if ".git" in path.parts or "data" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for name in legacy_names:
            if name in text:
                occurrences.append(f"{path.relative_to(PROJECT_ROOT)}: {name}")

    assert not occurrences, "Legacy promotion names found:\n" + "\n".join(occurrences)
