"""Load the raw Home Credit CSVs into a DuckDB database.

DuckDB reads the CSVs directly, so the ~340 MB of source data never has to be
materialised in pandas.

    python src/load_to_duckdb.py
"""

from __future__ import annotations

import sys

import duckdb

from config import DUCKDB_PATH, PROCESSED_DIR, RAW_TABLES


def missing_sources() -> list[str]:
    """Return the raw CSVs that have not been downloaded yet."""
    return [str(path) for path in RAW_TABLES.values() if not path.exists()]


def load(con: duckdb.DuckDBPyConnection) -> None:
    """(Re)create one table per raw CSV."""
    for table, path in RAW_TABLES.items():
        print(f"loading {path.name} -> {table} ...", flush=True)
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.execute(
            f"CREATE TABLE {table} AS SELECT * FROM read_csv_auto(?, header=true)",
            [str(path)],
        )
        rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        cols = len(con.execute(f"SELECT * FROM {table} LIMIT 0").description)
        print(f"  {table}: {rows:,} rows x {cols} columns")


def main() -> int:
    absent = missing_sources()
    if absent:
        print("Raw data not found. Expected:", file=sys.stderr)
        for path in absent:
            print(f"  {path}", file=sys.stderr)
        print(
            "\nDownload it first (see README step 1-2):\n"
            "  kaggle auth login\n"
            "  kaggle competitions download -c home-credit-default-risk -p data/raw",
            file=sys.stderr,
        )
        return 1

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(DUCKDB_PATH)) as con:
        load(con)
    print(f"\nDatabase written to {DUCKDB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
