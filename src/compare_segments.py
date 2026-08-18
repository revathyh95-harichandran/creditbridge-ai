"""Compare thin-file vs thick-file applicants on cleaned data.

Answers two questions:

  1. Do the segments actually look different on income, employment length, age, and
     application channel — using application_train_clean (src/clean.py) so the comparison
     reflects imputed/capped values, not raw nulls and outliers?
  2. Which non-bureau features (asset ownership, employment stability, family status,
     contact completeness, previous internal applications) are still fully populated for
     thin-file applicants — i.e. usable scorecard signal even where bureau/EXT_SOURCE data
     is sparse?

Completeness (question 2) is checked against the RAW application_train / previous_application
tables, not the cleaned table: application_train_clean has no missing values left by
construction (that's what clean.py is for), so checking completeness there would trivially
say "100% populated" for everything and defeat the point of the question.

    python src/compare_segments.py
"""

from __future__ import annotations

import sys

import duckdb

from config import DUCKDB_PATH

NON_BUREAU_FEATURE_GROUPS = {
    "asset ownership": ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"],
    "employment stability": ["NAME_INCOME_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "DAYS_EMPLOYED"],
    "family status": ["NAME_FAMILY_STATUS", "CNT_FAM_MEMBERS", "CNT_CHILDREN"],
    "contact completeness": ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"],
}


def income_by_segment(con: duckdb.DuckDBPyConnection) -> list[tuple]:
    return con.execute("""
        SELECT s.segment, COUNT(*),
               ROUND(AVG(c.AMT_INCOME_TOTAL), 0),
               ROUND(quantile_cont(c.AMT_INCOME_TOTAL, 0.5), 0),
               ROUND(quantile_cont(c.AMT_INCOME_TOTAL, 0.25), 0),
               ROUND(quantile_cont(c.AMT_INCOME_TOTAL, 0.75), 0)
        FROM applicant_segments s
        JOIN application_train_clean c USING (SK_ID_CURR)
        GROUP BY s.segment ORDER BY s.segment DESC
    """).fetchall()


def employment_length_by_segment(con: duckdb.DuckDBPyConnection) -> list[tuple]:
    return con.execute("""
        SELECT s.segment,
               ROUND(AVG(-c.DAYS_EMPLOYED / 365.25), 1),
               ROUND(quantile_cont(-c.DAYS_EMPLOYED / 365.25, 0.5), 1),
               ROUND(100.0 * AVG(c.DAYS_EMPLOYED_IS_RETIRED), 1)
        FROM applicant_segments s
        JOIN application_train_clean c USING (SK_ID_CURR)
        GROUP BY s.segment ORDER BY s.segment DESC
    """).fetchall()


def age_by_segment(con: duckdb.DuckDBPyConnection) -> list[tuple]:
    return con.execute("""
        SELECT s.segment,
               ROUND(AVG(-c.DAYS_BIRTH / 365.25), 1),
               ROUND(quantile_cont(-c.DAYS_BIRTH / 365.25, 0.5), 1)
        FROM applicant_segments s
        JOIN application_train_clean c USING (SK_ID_CURR)
        GROUP BY s.segment ORDER BY s.segment DESC
    """).fetchall()


def previous_application_coverage_by_segment(con: duckdb.DuckDBPyConnection) -> list[tuple]:
    """% of each segment with at least one prior application to this same lender."""
    return con.execute("""
        WITH has_prev AS (SELECT DISTINCT SK_ID_CURR FROM previous_application)
        SELECT s.segment, COUNT(*),
               ROUND(100.0 * AVG(CASE WHEN p.SK_ID_CURR IS NOT NULL THEN 1 ELSE 0 END), 1)
        FROM applicant_segments s
        LEFT JOIN has_prev p USING (SK_ID_CURR)
        GROUP BY s.segment ORDER BY s.segment DESC
    """).fetchall()


def channel_by_segment(con: duckdb.DuckDBPyConnection) -> list[tuple]:
    """Distribution of each applicant's most recent previous_application.CHANNEL_TYPE,
    among applicants who have at least one previous application."""
    return con.execute("""
        WITH ranked AS (
            SELECT SK_ID_CURR, CHANNEL_TYPE,
                   ROW_NUMBER() OVER (PARTITION BY SK_ID_CURR ORDER BY DAYS_DECISION DESC) rn
            FROM previous_application
        ),
        latest AS (SELECT SK_ID_CURR, CHANNEL_TYPE FROM ranked WHERE rn = 1)
        SELECT s.segment, l.CHANNEL_TYPE, COUNT(*),
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY s.segment), 1)
        FROM applicant_segments s
        JOIN latest l USING (SK_ID_CURR)
        GROUP BY s.segment, l.CHANNEL_TYPE
        ORDER BY s.segment DESC, COUNT(*) DESC
    """).fetchall()


def asset_ownership_rates_by_segment(con: duckdb.DuckDBPyConnection) -> list[tuple]:
    return con.execute("""
        SELECT s.segment,
               ROUND(100.0 * AVG(CASE WHEN a.FLAG_OWN_CAR = 'Y' THEN 1 ELSE 0 END), 1),
               ROUND(100.0 * AVG(CASE WHEN a.FLAG_OWN_REALTY = 'Y' THEN 1 ELSE 0 END), 1)
        FROM applicant_segments s JOIN application_train a USING (SK_ID_CURR)
        GROUP BY s.segment ORDER BY s.segment DESC
    """).fetchall()


def non_bureau_completeness(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """% missing per segment, per feature, checked against RAW application_train."""
    rows = []
    for group, cols in NON_BUREAU_FEATURE_GROUPS.items():
        for col in cols:
            by_segment = dict(con.execute(f"""
                SELECT s.segment, ROUND(100.0 * AVG(CASE WHEN a."{col}" IS NULL THEN 1 ELSE 0 END), 2)
                FROM applicant_segments s
                JOIN application_train a USING (SK_ID_CURR)
                GROUP BY s.segment
            """).fetchall())
            rows.append({
                "group": group, "feature": col,
                "thin_file_pct_missing": by_segment.get("thin-file"),
                "thick_file_pct_missing": by_segment.get("thick-file"),
            })
    return rows


def main() -> int:
    if not DUCKDB_PATH.exists():
        print(f"No database at {DUCKDB_PATH}.\nRun: python src/load_to_duckdb.py", file=sys.stderr)
        return 1

    with duckdb.connect(str(DUCKDB_PATH)) as con:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        missing = {"application_train_clean", "applicant_segments"} - tables
        if missing:
            print(f"Missing tables: {missing}.\nRun: python src/clean.py && python src/segment.py", file=sys.stderr)
            return 1

        print("=== Income (AMT_INCOME_TOTAL, cleaned) ===")
        for segment, n, mean, median, p25, p75 in income_by_segment(con):
            print(f"  {segment:<12} n={n:>7,}  mean={mean:>9,.0f}  median={median:>9,.0f}  p25={p25:>9,.0f}  p75={p75:>9,.0f}")

        print("\n=== Employment length (years, cleaned DAYS_EMPLOYED) ===")
        for segment, mean_yr, median_yr, pct_retired in employment_length_by_segment(con):
            print(f"  {segment:<12} mean={mean_yr:>5.1f}y  median={median_yr:>5.1f}y  retired={pct_retired:>5.1f}%")

        print("\n=== Age (years, DAYS_BIRTH) ===")
        for segment, mean_age, median_age in age_by_segment(con):
            print(f"  {segment:<12} mean={mean_age:>5.1f}y  median={median_age:>5.1f}y")

        print("\n=== Previous-application coverage (this lender) ===")
        for segment, n, pct in previous_application_coverage_by_segment(con):
            print(f"  {segment:<12} n={n:>7,}  has_previous_application={pct:>5.1f}%")

        print("\n=== Asset ownership rates (raw) ===")
        for segment, pct_car, pct_realty in asset_ownership_rates_by_segment(con):
            print(f"  {segment:<12} own_car={pct_car:>5.1f}%  own_realty={pct_realty:>5.1f}%")

        print("\n=== Application channel (most recent previous_application, among those with history) ===")
        for segment, channel, n, pct in channel_by_segment(con):
            print(f"  {segment:<12} {channel:<28} {pct:>5.1f}%  (n={n:,})")

        print("\n=== Non-bureau feature completeness by segment (raw data, % missing) ===")
        for row in non_bureau_completeness(con):
            print(
                f"  [{row['group']:<21}] {row['feature']:<20} "
                f"thin-file={row['thin_file_pct_missing']:>5.2f}%  thick-file={row['thick_file_pct_missing']:>5.2f}%"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
