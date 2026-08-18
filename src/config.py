"""Canonical paths for the creditbridge-ai project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

DUCKDB_PATH = PROCESSED_DIR / "creditbridge.duckdb"

# Raw CSVs loaded into DuckDB, keyed by the table name they land in.
RAW_TABLES = {
    "application_train": RAW_DIR / "application_train.csv",
    "bureau": RAW_DIR / "bureau.csv",
}

EXT_SOURCE_COLS = ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
