"""Tests for the cleaning decisions in src/clean.py, against a synthetic in-memory table.

Covers the specific behaviors that matter for this dataset: EXT_SOURCE missingness is
preserved (not imputed), the DAYS_EMPLOYED sentinel is resolved before capping runs,
degenerate-IQR columns are left uncapped, and ordinary numeric/categorical imputation
follows the median / 'Missing' rules.
"""

import duckdb
import pytest

from clean import DAYS_EMPLOYED_SENTINEL, build_clean_table

N_ROWS = 40


@pytest.fixture
def con():
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE application_train (
            SK_ID_CURR INTEGER,
            TARGET INTEGER,
            DAYS_EMPLOYED INTEGER,
            EXT_SOURCE_1 DOUBLE,
            EXT_SOURCE_2 DOUBLE,
            EXT_SOURCE_3 DOUBLE,
            AMT_INCOME_TOTAL DOUBLE,
            REGION_RATING_CLIENT INTEGER,
            OCCUPATION_TYPE VARCHAR,
            FLAG_OWN_CAR VARCHAR,
            FLAG_EMP_PHONE INTEGER
        )
    """)

    for i in range(N_ROWS):
        sk_id = i + 1
        target = i % 2
        # 5 pensioners carrying the sentinel, rest genuine negative tenures
        days_employed = DAYS_EMPLOYED_SENTINEL if i < 5 else -(200 + i * 30)
        ext1 = None if i % 3 == 0 else 0.1 * (i % 10)
        ext2 = round(0.2 + 0.01 * i, 3)
        ext3 = None if i % 7 == 0 else round(0.1 + 0.01 * i, 3)
        # one extreme income outlier among otherwise tight values
        income = 5_000_000.0 if i == 39 else 100_000.0 + i * 500
        # near-degenerate ordinal: overwhelmingly 2, a couple of 1s and 3s
        rating = 2 if i < 35 else (1 if i < 38 else 3)
        occupation = None if i % 5 == 0 else "Laborers"
        flag_own_car = "Y" if i % 2 == 0 else "N"
        flag_emp_phone = 1

        con.execute(
            "INSERT INTO application_train VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [sk_id, target, days_employed, ext1, ext2, ext3, income, rating, occupation, flag_own_car, flag_emp_phone],
        )

    yield con
    con.close()


def test_ext_source_missingness_is_preserved_not_imputed(con):
    build_clean_table(con)
    raw_nulls = con.execute("SELECT COUNT(*) FROM application_train WHERE EXT_SOURCE_1 IS NULL").fetchone()[0]
    clean_nulls = con.execute("SELECT COUNT(*) FROM application_train_clean WHERE EXT_SOURCE_1 IS NULL").fetchone()[0]
    assert raw_nulls > 0
    assert clean_nulls == raw_nulls


def test_ext_source_is_missing_flag_matches_nulls(con):
    build_clean_table(con)
    mismatches = con.execute("""
        SELECT COUNT(*) FROM application_train_clean
        WHERE (EXT_SOURCE_1 IS NULL) != (EXT_SOURCE_1_IS_MISSING = 1)
    """).fetchone()[0]
    assert mismatches == 0


def test_ext_source_with_zero_missing_still_gets_flag_column(con):
    build_clean_table(con)
    always_zero = con.execute(
        "SELECT COUNT(*) FROM application_train_clean WHERE EXT_SOURCE_2_IS_MISSING != 0"
    ).fetchone()[0]
    assert always_zero == 0


def test_days_employed_sentinel_is_resolved(con):
    build_clean_table(con)
    remaining_sentinel = con.execute(
        f"SELECT COUNT(*) FROM application_train_clean WHERE DAYS_EMPLOYED = {DAYS_EMPLOYED_SENTINEL}"
    ).fetchone()[0]
    assert remaining_sentinel == 0


def test_days_employed_is_retired_flag_matches_sentinel_rows(con):
    build_clean_table(con)
    n_retired = con.execute("SELECT SUM(DAYS_EMPLOYED_IS_RETIRED) FROM application_train_clean").fetchone()[0]
    assert n_retired == 5


def test_days_employed_no_nulls_remain_after_impute(con):
    build_clean_table(con)
    nulls = con.execute("SELECT COUNT(*) FROM application_train_clean WHERE DAYS_EMPLOYED IS NULL").fetchone()[0]
    assert nulls == 0


def test_days_employed_genuine_values_untouched(con):
    """A genuine tenure value inside the normal range should survive unchanged."""
    build_clean_table(con)
    original = con.execute(
        "SELECT DAYS_EMPLOYED FROM application_train WHERE SK_ID_CURR = 20"
    ).fetchone()[0]
    cleaned = con.execute(
        "SELECT DAYS_EMPLOYED FROM application_train_clean WHERE SK_ID_CURR = 20"
    ).fetchone()[0]
    assert cleaned == pytest.approx(original)


def test_income_outlier_is_capped_not_dropped(con):
    build_clean_table(con)
    n_rows_clean = con.execute("SELECT COUNT(*) FROM application_train_clean").fetchone()[0]
    assert n_rows_clean == N_ROWS  # capping must not remove rows

    capped_value = con.execute(
        "SELECT AMT_INCOME_TOTAL FROM application_train_clean WHERE SK_ID_CURR = 40"
    ).fetchone()[0]
    assert capped_value < 5_000_000.0  # pulled down from the extreme raw value
    assert capped_value > 100_000.0    # still a plausible income, not zeroed out


def test_degenerate_iqr_column_is_not_capped(con):
    """REGION_RATING_CLIENT is overwhelmingly 2s -> Q1 == Q3 -> capping must be skipped."""
    build_clean_table(con)
    raw_values = dict(con.execute("SELECT SK_ID_CURR, REGION_RATING_CLIENT FROM application_train").fetchall())
    clean_values = dict(con.execute("SELECT SK_ID_CURR, REGION_RATING_CLIENT FROM application_train_clean").fetchall())
    assert raw_values == clean_values


def test_categorical_missing_filled_with_explicit_category(con):
    build_clean_table(con)
    remaining_nulls = con.execute(
        "SELECT COUNT(*) FROM application_train_clean WHERE OCCUPATION_TYPE IS NULL"
    ).fetchone()[0]
    assert remaining_nulls == 0

    missing_count = con.execute(
        "SELECT COUNT(*) FROM application_train_clean WHERE OCCUPATION_TYPE = 'Missing'"
    ).fetchone()[0]
    raw_missing_count = con.execute(
        "SELECT COUNT(*) FROM application_train WHERE OCCUPATION_TYPE IS NULL"
    ).fetchone()[0]
    assert missing_count == raw_missing_count > 0


def test_binary_flag_with_missing_raises(con):
    con.execute("UPDATE application_train SET FLAG_EMP_PHONE = NULL WHERE SK_ID_CURR = 1")
    with pytest.raises(ValueError, match="FLAG_EMP_PHONE"):
        build_clean_table(con)


def test_categorical_string_binary_column_uses_missing_category_not_raise(con):
    """FLAG_OWN_CAR is a string ('Y'/'N'), so it's categorical, not a binary-flag int column —
    a NULL there should be filled with 'Missing', not raise."""
    con.execute("UPDATE application_train SET FLAG_OWN_CAR = NULL WHERE SK_ID_CURR = 1")
    build_clean_table(con)
    value = con.execute(
        "SELECT FLAG_OWN_CAR FROM application_train_clean WHERE SK_ID_CURR = 1"
    ).fetchone()[0]
    assert value == "Missing"
