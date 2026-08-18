"""Clean application_train into application_train_clean.

Scope is application_train only, matching the data-quality pass in
notebooks/01_data_quality.ipynb: this is the table that carries TARGET and becomes the
model matrix, so it is the table whose missing values and outliers actually need resolving
before modeling. bureau and previous_application are relational history tables that get
aggregated into application_train-level features later, not cleaned column-by-column here.

Decisions made (full rationale + alternatives considered in reports/segment_findings.md):

  * Continuous numeric columns, EXCEPT EXT_SOURCE_1/2/3: missing values imputed with the
    column median.
  * EXT_SOURCE_1/2/3: never imputed. The raw column (with its NULLs) passes through
    unchanged, plus a companion EXT_SOURCE_n_IS_MISSING flag (1/0) so a model can use
    missingness itself as a feature instead of it being silently erased.
  * Categorical (string/boolean) columns: missing values filled with the explicit string
    "Missing" rather than a mode/most-frequent value.
  * Binary 0/1 indicator columns (FLAG_*, FLAG_DOCUMENT_*, REG_*_NOT_*): passed through
    unchanged. Verified 0% missing in the source data, so no imputation decision applies;
    if that ever changes, build_clean_table raises rather than silently guessing.
  * Outliers: flagged via the IQR method (outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]). Where IQR
    is nonzero and at least one outlier is found, values are capped (not dropped) at the
    1st/99th percentile. Where IQR == 0 (degenerate — ordinal ratings, zero-inflated
    counts; see notebooks/01_data_quality.ipynb section 5), capping is skipped because the
    IQR test isn't measuring anything meaningful for that column.
  * DAYS_EMPLOYED is a special case, discovered in the data-quality notebook: 365243 is a
    sentinel meaning "not employed" (mostly pensioners), not a real tenure — and it's
    common enough (18% of applicants) that it single-handedly sets the column's 99th
    percentile. Generic percentile capping is therefore a no-op on it: the cap ceiling IS
    the sentinel. Before any other processing, 365243 is converted to NULL and captured in
    a DAYS_EMPLOYED_IS_RETIRED flag (1/0); the resulting NULLs then go through the normal
    median-impute + IQR-capping path like any other numeric column, computed only from
    genuine tenure values.

The source table is left untouched; results land in a new application_train_clean table
so the raw data stays available for audit.

    python src/clean.py
"""

from __future__ import annotations

import sys

import duckdb

from config import DUCKDB_PATH, EXT_SOURCE_COLS

RAW_TABLE = "application_train"
PREPPED_VIEW = "application_train_prepped"
CLEAN_TABLE = "application_train_clean"
ID_COLS = ("SK_ID_CURR", "TARGET")
NUMERIC_SQL_TYPES = {"BIGINT", "DOUBLE", "INTEGER"}
DAYS_EMPLOYED_SENTINEL = 365243

# All column-level functions below read from SOURCE_TABLE, which build_clean_table points
# at PREPPED_VIEW (raw data with the DAYS_EMPLOYED sentinel already resolved) rather than
# the raw table directly.
SOURCE_TABLE = PREPPED_VIEW


def prep_source(con: duckdb.DuckDBPyConnection) -> None:
    """Resolve the DAYS_EMPLOYED sentinel before anything else runs.

    365243 ("not employed", mostly pensioners) is common enough that it sets the column's
    own 99th percentile, so generic capping can't touch it. NULL it out and capture it in
    DAYS_EMPLOYED_IS_RETIRED so the rest of the pipeline treats it as ordinary missingness.
    """
    con.execute(f"""
        CREATE OR REPLACE VIEW {PREPPED_VIEW} AS
        SELECT
            * EXCLUDE (DAYS_EMPLOYED),
            CASE WHEN DAYS_EMPLOYED = {DAYS_EMPLOYED_SENTINEL} THEN NULL ELSE DAYS_EMPLOYED END AS DAYS_EMPLOYED,
            CASE WHEN DAYS_EMPLOYED = {DAYS_EMPLOYED_SENTINEL} THEN 1 ELSE 0 END AS DAYS_EMPLOYED_IS_RETIRED
        FROM {RAW_TABLE}
    """)


def classify_columns(con: duckdb.DuckDBPyConnection) -> tuple[list[str], list[str], list[str]]:
    """Continuous numeric (>2 distinct values) / binary flag (<=2) / categorical (string, bool).

    Same rule as the data-quality notebook: skewness and IQR fences are meaningless for a
    two-valued column, so numeric columns with <=2 distinct values are treated as flags,
    not continuous features.
    """
    desc = con.execute(f"DESCRIBE {SOURCE_TABLE}").fetchall()
    continuous, binary_flags, categorical = [], [], []
    for name, col_type, *_ in desc:
        if name in ID_COLS:
            continue
        if col_type in NUMERIC_SQL_TYPES:
            n = con.execute(f'SELECT COUNT(DISTINCT "{name}") FROM {SOURCE_TABLE}').fetchone()[0]
            (continuous if n > 2 else binary_flags).append(name)
        else:
            categorical.append(name)
    return continuous, binary_flags, categorical


def numeric_stats(con: duckdb.DuckDBPyConnection, col: str) -> dict:
    """Median, IQR fences, 1st/99th percentiles, and outlier count for one numeric column."""
    median, q1, q3, p1, p99, n_missing = con.execute(f'''
        SELECT
            quantile_cont("{col}", 0.5),
            quantile_cont("{col}", 0.25),
            quantile_cont("{col}", 0.75),
            quantile_cont("{col}", 0.01),
            quantile_cont("{col}", 0.99),
            SUM(CASE WHEN "{col}" IS NULL THEN 1 ELSE 0 END)
        FROM {SOURCE_TABLE}
    ''').fetchone()

    iqr = None if q1 is None else q3 - q1
    n_outliers = 0
    if iqr:
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = con.execute(f'''
            SELECT COUNT(*) FROM {SOURCE_TABLE}
            WHERE "{col}" IS NOT NULL AND ("{col}" < {lo} OR "{col}" > {hi})
        ''').fetchone()[0]

    return {"median": median, "p1": p1, "p99": p99, "iqr": iqr, "n_missing": n_missing, "n_outliers": n_outliers}


def numeric_column_plan(col: str, stats: dict, is_ext_source: bool, total_rows: int) -> tuple[str, list[str], dict]:
    """Return (SQL expr for the cleaned column, extra companion-column SQL exprs, decision log)."""
    pct_missing = round(100 * stats["n_missing"] / total_rows, 2)

    if is_ext_source:
        expr = f'"{col}"'
        extra = f'CASE WHEN "{col}" IS NULL THEN 1 ELSE 0 END AS "{col}_IS_MISSING"'
        decision = {
            "column": col, "action": "kept raw + added is_missing flag",
            "pct_missing": pct_missing, "capped": False,
        }
        return expr, [extra], decision

    should_cap = bool(stats["iqr"]) and stats["n_outliers"] > 0
    if should_cap:
        expr = (
            f'COALESCE(CASE '
            f'WHEN "{col}" < {stats["p1"]} THEN {stats["p1"]} '
            f'WHEN "{col}" > {stats["p99"]} THEN {stats["p99"]} '
            f'ELSE "{col}" END, {stats["median"]})'
        )
    else:
        expr = f'COALESCE("{col}", {stats["median"]})'

    decision = {
        "column": col, "action": "median-imputed" + (" + capped" if should_cap else ""),
        "pct_missing": pct_missing, "capped": should_cap,
        "median": stats["median"], "p1": stats["p1"], "p99": stats["p99"],
        "n_outliers": stats["n_outliers"],
    }
    return expr, [], decision


def categorical_column_plan(col: str, col_type: str, n_missing: int, total_rows: int) -> tuple[str, dict]:
    if col_type == "BOOLEAN":
        expr = f"COALESCE(CAST(\"{col}\" AS VARCHAR), 'Missing')"
    else:
        expr = f"COALESCE(\"{col}\", 'Missing')"
    decision = {
        "column": col, "action": "filled with explicit 'Missing' category",
        "pct_missing": round(100 * n_missing / total_rows, 2),
    }
    return expr, decision


def build_clean_table(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Create application_train_clean. Returns the list of cleaning decisions made."""
    prep_source(con)
    total_rows = con.execute(f"SELECT COUNT(*) FROM {SOURCE_TABLE}").fetchone()[0]

    continuous, binary_flags, categorical = classify_columns(con)
    col_types = {name: col_type for name, col_type, *_ in con.execute(f"DESCRIBE {SOURCE_TABLE}").fetchall()}

    select_parts = ['"SK_ID_CURR"', '"TARGET"']
    decisions = []

    for col in continuous:
        stats = numeric_stats(con, col)
        expr, extras, decision = numeric_column_plan(col, stats, col in EXT_SOURCE_COLS, total_rows)
        select_parts.append(f'{expr} AS "{col}"')
        select_parts.extend(extras)
        decisions.append(decision)

    for col in binary_flags:
        n_missing = con.execute(f'SELECT SUM(CASE WHEN "{col}" IS NULL THEN 1 ELSE 0 END) FROM {SOURCE_TABLE}').fetchone()[0]
        if n_missing:
            raise ValueError(
                f"{col} has {n_missing} missing values; binary-flag columns were assumed 0% "
                "missing at design time (see src/clean.py docstring). Add an explicit "
                "imputation rule for this column before proceeding."
            )
        select_parts.append(f'"{col}"')

    for col in categorical:
        n_missing = con.execute(f'SELECT SUM(CASE WHEN "{col}" IS NULL THEN 1 ELSE 0 END) FROM {SOURCE_TABLE}').fetchone()[0]
        expr, decision = categorical_column_plan(col, col_types[col], n_missing, total_rows)
        select_parts.append(f'{expr} AS "{col}"')
        decisions.append(decision)

    con.execute(f"CREATE OR REPLACE TABLE {CLEAN_TABLE} AS SELECT {', '.join(select_parts)} FROM {SOURCE_TABLE}")
    return decisions


def print_decisions(decisions: list[dict]) -> None:
    ext_source = [d for d in decisions if "is_missing flag" in d["action"]]
    capped = [d for d in decisions if d.get("capped")]
    imputed_only = [d for d in decisions if d["action"] == "median-imputed"]
    categorical = [d for d in decisions if "Missing" in d["action"]]

    print(f"\n{len(ext_source)} EXT_SOURCE columns: kept raw, missingness flagged")
    for d in ext_source:
        print(f"  {d['column']:<20} {d['pct_missing']:>5.1f}% missing -> {d['column']}_IS_MISSING")

    print(f"\n{len(capped)} numeric columns: median-imputed + capped at [p1, p99]")
    for d in capped:
        print(f"  {d['column']:<28} {d['pct_missing']:>5.1f}% missing, {d['n_outliers']:>6,} outliers -> capped to [{d['p1']:.2f}, {d['p99']:.2f}]")

    print(f"\n{len(imputed_only)} numeric columns: median-imputed, not capped (no IQR outliers / degenerate IQR)")
    for d in imputed_only:
        print(f"  {d['column']:<28} {d['pct_missing']:>5.1f}% missing")

    print(f"\n{len(categorical)} categorical columns: filled with 'Missing'")
    for d in categorical:
        print(f"  {d['column']:<28} {d['pct_missing']:>5.1f}% missing")


def main() -> int:
    if not DUCKDB_PATH.exists():
        print(f"No database at {DUCKDB_PATH}.\nRun: python src/load_to_duckdb.py", file=sys.stderr)
        return 1

    with duckdb.connect(str(DUCKDB_PATH)) as con:
        decisions = build_clean_table(con)
        n_retired = con.execute(f'SELECT SUM(DAYS_EMPLOYED_IS_RETIRED) FROM {CLEAN_TABLE}').fetchone()[0]
        n_rows, n_cols = con.execute(f"SELECT COUNT(*) FROM {CLEAN_TABLE}").fetchone()[0], len(
            con.execute(f"SELECT * FROM {CLEAN_TABLE} LIMIT 0").description
        )

    print(f"{CLEAN_TABLE}: {n_rows:,} rows x {n_cols} columns")
    print(
        f"\nDAYS_EMPLOYED sentinel (365243): {n_retired:,} rows resolved to NULL -> "
        "DAYS_EMPLOYED_IS_RETIRED=1, then median-imputed/capped like any other numeric column"
    )
    print_decisions(decisions)
    print(f"\nWritten to {DUCKDB_PATH} :: {CLEAN_TABLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
