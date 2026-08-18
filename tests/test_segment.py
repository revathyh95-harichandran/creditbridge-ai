"""Segmentation tests against a synthetic in-memory database.

Each applicant below sits on a specific side of the thin-file rule, so the tests pin the
boundary conditions (1 vs 2 bureau records, zero vs one missing EXT_SOURCE) rather than
just the happy path.
"""

import duckdb
import pytest

from segment import build_segments, format_summary, segment_summary

# (SK_ID_CURR, TARGET, EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, n_bureau, expected segment)
APPLICANTS = [
    (1, 1, None, 0.5, 0.5, 0, "thin-file"),   # no bureau history, one score missing
    (2, 0, 0.4, None, None, 1, "thin-file"),  # single bureau record, two scores missing
    (3, 0, None, None, None, 1, "thin-file"), # boundary: exactly 1 record, all missing
    (4, 0, None, 0.5, 0.5, 2, "thick-file"),  # boundary: 2 records defeats the rule
    (5, 1, 0.1, 0.2, 0.3, 0, "thick-file"),   # no bureau history but fully scored
    (6, 0, 0.1, 0.2, 0.3, 5, "thick-file"),   # fully documented
]


@pytest.fixture
def con():
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE application_train_clean (
            SK_ID_CURR INTEGER, TARGET INTEGER,
            EXT_SOURCE_1 DOUBLE, EXT_SOURCE_2 DOUBLE, EXT_SOURCE_3 DOUBLE
        )
    """)
    con.execute("CREATE TABLE bureau (SK_ID_CURR INTEGER, SK_ID_BUREAU INTEGER)")

    bureau_id = 1000
    for sk_id, target, e1, e2, e3, n_bureau, _ in APPLICANTS:
        con.execute("INSERT INTO application_train_clean VALUES (?, ?, ?, ?, ?)",
                    [sk_id, target, e1, e2, e3])
        for _ in range(n_bureau):
            bureau_id += 1
            con.execute("INSERT INTO bureau VALUES (?, ?)", [sk_id, bureau_id])

    yield con
    con.close()


@pytest.mark.parametrize("sk_id,expected", [(a[0], a[6]) for a in APPLICANTS])
def test_applicant_lands_in_expected_segment(con, sk_id, expected):
    build_segments(con)
    got = con.execute(
        "SELECT segment FROM applicant_segments WHERE SK_ID_CURR = ?", [sk_id]
    ).fetchone()[0]
    assert got == expected


def test_every_applicant_is_flagged_exactly_once(con):
    build_segments(con)
    total, distinct = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT SK_ID_CURR) FROM applicant_segments"
    ).fetchone()
    assert total == distinct == len(APPLICANTS)


def test_applicants_with_no_bureau_rows_count_as_zero_not_null(con):
    """The LEFT JOIN must yield 0, not NULL, or the <= 1 comparison silently drops them."""
    build_segments(con)
    assert con.execute(
        "SELECT n_bureau_records FROM applicant_segments WHERE SK_ID_CURR = 1"
    ).fetchone()[0] == 0
    assert con.execute(
        "SELECT COUNT(*) FROM applicant_segments WHERE n_bureau_records IS NULL"
    ).fetchone()[0] == 0


def test_missing_ext_source_count(con):
    build_segments(con)
    counts = dict(con.execute(
        "SELECT SK_ID_CURR, n_missing_ext_sources FROM applicant_segments"
    ).fetchall())
    assert counts[1] == 1
    assert counts[3] == 3
    assert counts[6] == 0


def test_summary_sizes_and_default_rates(con):
    build_segments(con)
    summary = {row[0]: row for row in segment_summary(con)}

    thin = summary["thin-file"]
    thick = summary["thick-file"]

    assert thin[1] == 3                      # applicants
    assert thin[3] == 1                      # defaults
    assert thin[4] == pytest.approx(1 / 3)   # default rate
    assert thick[1] == 3
    assert thick[3] == 1
    assert thick[4] == pytest.approx(1 / 3)
    assert thin[2] == pytest.approx(0.5)     # share of book


def test_summary_is_ordered_thin_file_first(con):
    build_segments(con)
    assert [row[0] for row in segment_summary(con)] == ["thin-file", "thick-file"]


def test_format_summary_renders_both_segments_and_total(con):
    build_segments(con)
    out = format_summary(segment_summary(con))
    assert "thin-file" in out and "thick-file" in out
    assert "overall" in out
    assert "33.33%" in out
