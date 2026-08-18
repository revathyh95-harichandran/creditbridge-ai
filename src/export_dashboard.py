"""Export the cleaned, segmented applicant dataset to a single dashboard-ready CSV.

Business-readable columns, not raw dataset codes — `TARGET` becomes `defaulted`, the
segment flag becomes `file_type`, `DAYS_EMPLOYED`/`DAYS_BIRTH` become whole years, etc.
Boolean fields (`defaulted`, `owns_car`, `owns_realty`, `prior_applicant`) are encoded as
0/1 rather than Yes/No so a BI tool can average them directly into a rate without a
calculated field.

Alongside income/employment/age/family status, five non-bureau features are included —
the ones identified in reports/segment_findings.md §4 as fully populated (0% missing) for
thin-file applicants: asset ownership (owns_car, owns_realty), employment type
(income_type), household composition (household_size), and prior relationship with this
lender (prior_applicant). The three near-constant contact flags from that same section
(FLAG_MOBIL 100%, FLAG_CONT_MOBILE 99.8%, FLAG_EMP_PHONE 82%) are left out — they barely
vary across applicants, so they don't discriminate between them on a dashboard.

    python src/export_dashboard.py
"""

from __future__ import annotations

import sys

import duckdb

from config import DUCKDB_PATH, PROCESSED_DIR

EXPORT_PATH = PROCESSED_DIR / "dashboard_export.csv"

EXPORT_QUERY = """
    SELECT
        s.SK_ID_CURR AS applicant_id,
        s.segment AS file_type,
        s.TARGET AS defaulted,
        CAST(ROUND(c.AMT_INCOME_TOTAL, 0) AS BIGINT) AS income,
        ROUND(-c.DAYS_EMPLOYED / 365.25, 1) AS employment_years,
        CAST(ROUND(-c.DAYS_BIRTH / 365.25, 0) AS INTEGER) AS age,
        c.NAME_FAMILY_STATUS AS family_status,
        CASE WHEN c.FLAG_OWN_CAR = 'Y' THEN 1 ELSE 0 END AS owns_car,
        CASE WHEN c.FLAG_OWN_REALTY = 'Y' THEN 1 ELSE 0 END AS owns_realty,
        c.NAME_INCOME_TYPE AS income_type,
        CAST(ROUND(c.CNT_FAM_MEMBERS, 0) AS INTEGER) AS household_size,
        CASE WHEN p.SK_ID_CURR IS NOT NULL THEN 1 ELSE 0 END AS prior_applicant
    FROM applicant_segments s
    JOIN application_train_clean c USING (SK_ID_CURR)
    LEFT JOIN (SELECT DISTINCT SK_ID_CURR FROM previous_application) p USING (SK_ID_CURR)
    ORDER BY s.SK_ID_CURR
"""


def main() -> int:
    if not DUCKDB_PATH.exists():
        print(f"No database at {DUCKDB_PATH}.\nRun: python src/load_to_duckdb.py", file=sys.stderr)
        return 1

    with duckdb.connect(str(DUCKDB_PATH)) as con:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        missing = {"application_train_clean", "applicant_segments", "previous_application"} - tables
        if missing:
            print(
                f"Missing tables: {missing}.\nRun: python src/clean.py && python src/segment.py",
                file=sys.stderr,
            )
            return 1

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        con.execute(f"COPY ({EXPORT_QUERY}) TO '{EXPORT_PATH.as_posix()}' (HEADER, DELIMITER ',')")
        n_rows = con.execute(f"SELECT COUNT(*) FROM ({EXPORT_QUERY})").fetchone()[0]

    print(f"{n_rows:,} rows exported to {EXPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
