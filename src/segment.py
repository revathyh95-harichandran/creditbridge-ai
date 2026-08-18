"""Flag each applicant as thin-file or thick-file, and report segment risk.

An applicant is **thin-file** when both of these hold:

  * they have 0 or 1 records in ``bureau`` (almost no external credit history), and
  * at least one of EXT_SOURCE_1 / EXT_SOURCE_2 / EXT_SOURCE_3 is missing.

Everyone else is **thick-file**. The point of the split is to isolate the applicants a
conventional scorecard cannot see clearly, and compare their realised default rate against
the well-documented majority.

Reads from application_train_clean (src/clean.py), not the raw table: clean.py deliberately
leaves EXT_SOURCE_1/2/3 NULL where missing (see its docstring), so the NULL-count check below
is unaffected by cleaning and gives identical results either way — but application_train_clean
is the canonical per-applicant feature table for everything downstream (see
src/compare_segments.py), so segmentation is defined against it for consistency.

    python src/clean.py    # first, if application_train_clean doesn't exist yet
    python src/segment.py
"""

from __future__ import annotations

import sys

import duckdb

from config import DUCKDB_PATH, EXT_SOURCE_COLS

SEGMENT_TABLE = "applicant_segments"

MAX_BUREAU_RECORDS_FOR_THIN = 1
MIN_MISSING_EXT_SOURCES_FOR_THIN = 1


def _missing_ext_source_expr(alias: str = "a") -> str:
    """SQL counting how many EXT_SOURCE scores are missing for a row."""
    return " + ".join(
        f"CAST({alias}.{col} IS NULL AS INTEGER)" for col in EXT_SOURCE_COLS
    )


def build_segments(con: duckdb.DuckDBPyConnection, table: str = SEGMENT_TABLE) -> None:
    """Create the per-applicant segment table from ``application_train_clean`` + ``bureau``."""
    con.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        WITH bureau_counts AS (
            SELECT SK_ID_CURR, COUNT(*) AS n_bureau_records
            FROM bureau
            GROUP BY SK_ID_CURR
        ),
        flagged AS (
            SELECT
                a.SK_ID_CURR,
                a.TARGET,
                COALESCE(b.n_bureau_records, 0) AS n_bureau_records,
                {_missing_ext_source_expr('a')} AS n_missing_ext_sources
            FROM application_train_clean a
            LEFT JOIN bureau_counts b ON a.SK_ID_CURR = b.SK_ID_CURR
        )
        SELECT
            *,
            CASE
                WHEN n_bureau_records <= {MAX_BUREAU_RECORDS_FOR_THIN}
                 AND n_missing_ext_sources >= {MIN_MISSING_EXT_SOURCES_FOR_THIN}
                THEN 'thin-file'
                ELSE 'thick-file'
            END AS segment
        FROM flagged
    """)


def segment_summary(
    con: duckdb.DuckDBPyConnection, table: str = SEGMENT_TABLE
) -> list[tuple]:
    """Return (segment, applicants, share, defaults, default_rate) ordered thin-file first."""
    return con.execute(f"""
        SELECT
            segment,
            COUNT(*)                                   AS applicants,
            COUNT(*) * 1.0 / SUM(COUNT(*)) OVER ()     AS share_of_book,
            SUM(TARGET)                                AS defaults,
            AVG(TARGET * 1.0)                          AS default_rate
        FROM {table}
        GROUP BY segment
        ORDER BY segment DESC
    """).fetchall()


def format_summary(rows: list[tuple]) -> str:
    """Render the segment summary as a fixed-width table."""
    header = (
        f"{'segment':<12}{'applicants':>12}{'share':>9}"
        f"{'defaults':>10}{'default rate':>14}"
    )
    lines = [header, "-" * len(header)]
    for segment, applicants, share, defaults, rate in rows:
        lines.append(
            f"{segment:<12}{applicants:>12,}{share:>8.1%}"
            f"{int(defaults):>10,}{rate:>13.2%}"
        )

    totals = _overall(rows)
    if totals is not None:
        applicants, defaults, rate = totals
        lines.append("-" * len(header))
        lines.append(
            f"{'overall':<12}{applicants:>12,}{1.0:>8.1%}"
            f"{defaults:>10,}{rate:>13.2%}"
        )
    return "\n".join(lines)


def _overall(rows: list[tuple]) -> tuple[int, int, float] | None:
    if not rows:
        return None
    applicants = sum(r[1] for r in rows)
    defaults = sum(int(r[3]) for r in rows)
    return applicants, defaults, defaults / applicants if applicants else 0.0


def _lift_note(rows: list[tuple]) -> str:
    """One line putting the thin-file default rate in context."""
    rates = {segment: rate for segment, _, _, _, rate in rows}
    thin, thick = rates.get("thin-file"), rates.get("thick-file")
    if thin is None or thick is None or not thick:
        return ""
    return (
        f"\nThin-file applicants default at {thin / thick:.2f}x the thick-file rate "
        f"({thin:.2%} vs {thick:.2%})."
    )


def main() -> int:
    if not DUCKDB_PATH.exists():
        print(
            f"No database at {DUCKDB_PATH}.\nRun: python src/load_to_duckdb.py",
            file=sys.stderr,
        )
        return 1

    with duckdb.connect(str(DUCKDB_PATH)) as con:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "application_train_clean" not in tables:
            print(
                "application_train_clean not found.\nRun: python src/clean.py",
                file=sys.stderr,
            )
            return 1
        build_segments(con)
        rows = segment_summary(con)

    print("\nThin-file rule: <= "
          f"{MAX_BUREAU_RECORDS_FOR_THIN} bureau record(s) AND >= "
          f"{MIN_MISSING_EXT_SOURCES_FOR_THIN} missing EXT_SOURCE score\n")
    print(format_summary(rows))
    print(_lift_note(rows))
    print(f"Per-applicant flags written to {DUCKDB_PATH} :: {SEGMENT_TABLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
