"""Central project paths and directory initialization helpers."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_FEATURES = PROJECT_ROOT / "data" / "features"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
MODELS_DIR = PROJECT_ROOT / "models"
SQL_DIR = PROJECT_ROOT / "sql"


def ensure_project_directories() -> None:
    """Create project directories when they do not already exist."""
    directories = (
        DATA_RAW,
        DATA_INTERIM,
        DATA_PROCESSED,
        DATA_FEATURES,
        REPORTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        MODELS_DIR,
        SQL_DIR,
    )

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
