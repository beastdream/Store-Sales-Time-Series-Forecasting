"""Cross-platform contracts for persisted repository-relative paths."""

import json
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _walk_strings(value: object, key: str = "") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for child_key, child in value.items():
            child_path = f"{key}.{child_key}" if key else str(child_key)
            found.extend(_walk_strings(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_strings(child, f"{key}[{index}]"))
    elif isinstance(value, str):
        found.append((key, value))
    return found


def test_persisted_json_repository_paths_use_posix_separators() -> None:
    checked = 0
    for root in (PROJECT_ROOT / "models", PROJECT_ROOT / "reports"):
        for path in root.rglob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            for key, value in _walk_strings(data):
                is_path_field = (
                    key.endswith(".path")
                    or key == "path"
                    or key.endswith("_path")
                    or key.endswith("model_artifact")
                    or value.startswith(("models/", "reports/", "data/"))
                )
                if is_path_field:
                    checked += 1
                    assert "\\" not in value, f"{path}:{key} is not POSIX-style"
    assert checked > 0


def test_path_generators_and_generated_reports_use_posix_style() -> None:
    tuning_source = (
        PROJECT_ROOT / "notebooks" / "16_global_lightgbm_tuning.py"
    ).read_text(encoding="utf-8")
    final_source = (
        PROJECT_ROOT / "notebooks" / "20_final_competition_forecast.py"
    ).read_text(encoding="utf-8")
    assert ".as_posix()" in tuning_source
    assert final_source.count(".as_posix()") >= 6

    for relative_path in (
        "reports/da_project_validation.md",
        "reports/modeling/final_forecast_report.md",
    ):
        text = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        assert "\\" not in text, f"{relative_path} contains Windows separators"


def test_direct_runtime_dependencies_are_declared_explicitly() -> None:
    lines = [
        line.strip().lower()
        for line in (PROJECT_ROOT / "requirements.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    declared = {re.split(r"[<>=!~]", line, maxsplit=1)[0] for line in lines}
    direct_runtime = {
        "numpy",
        "pandas",
        "matplotlib",
        "lightgbm",
        "sqlalchemy",
        "sqlparse",
        "python-dotenv",
    }
    assert direct_runtime <= declared
    assert len(lines) == len(set(lines))
