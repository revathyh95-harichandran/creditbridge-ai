"""Tests for the evaluation helpers in src/validate_model.py, on tiny hand-checkable data."""

import math

import numpy as np
import pandas as pd
import pytest

from validate_model import (
    accuracy_trap,
    age_band,
    approval_by_group,
    bootstrap_auc_range,
    calibration_table,
    downsample_curve,
    flag_riskiest,
    gains_curve,
    gini,
    ks_statistic,
    people_to_find_share_of_defaulters,
    people_to_find_share_randomly,
    subgroup_auc,
)


def test_accuracy_trap_lazy_model_scores_the_non_default_rate():
    out = accuracy_trap(np.array([0, 0, 0, 1]), np.array([0.1, 0.2, 0.6, 0.9]))
    assert out["lazy_accuracy"] == 0.75
    assert out["model_accuracy"] == 0.75  # right on 0, 1, 3; wrong on the false alarm at 0.6
    assert out["n_flagged"] == 2


def test_flag_riskiest_precision_recall_f1():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.1, 0.1])
    out = flag_riskiest(y, scores, 0.2)  # flags the top 2: one defaulter, one not
    assert out["n_flagged"] == 2
    assert out["precision"] == pytest.approx(0.5)
    assert out["recall"] == pytest.approx(0.5)
    assert out["f1"] == pytest.approx(0.5)


def test_flag_riskiest_with_no_hits_has_zero_f1():
    out = flag_riskiest(np.array([0, 0, 1]), np.array([0.9, 0.5, 0.1]), 0.34)
    assert out["precision"] == 0 and out["recall"] == 0 and out["f1"] == 0


def test_people_to_find_share_of_defaulters_follows_the_model_order():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.3])  # riskiest first: 1, 3, 2, 0
    assert people_to_find_share_of_defaulters(y, scores, 0.5) == 1
    assert people_to_find_share_of_defaulters(y, scores, 1.0) == 2


def test_people_to_find_share_randomly_is_the_expected_value():
    y = np.array([1, 1] + [0] * 18)  # 10% default rate
    assert people_to_find_share_randomly(y, 0.5) == 10  # 1 defaulter at a 10% hit rate


def test_gini_is_two_auc_minus_one():
    assert gini(0.5) == 0.0
    assert gini(1.0) == 1.0
    assert gini(0.685) == pytest.approx(0.37)


def test_ks_statistic_extremes():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert ks_statistic(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.0)


def test_calibration_table_groups_are_safest_first_with_extras_in_the_safest_groups():
    p = np.linspace(0.01, 0.12, 12)
    y = np.array([0] * 9 + [1] * 3)
    table = calibration_table(y, p, n_groups=5)
    assert table["n"].tolist() == [3, 3, 2, 2, 2]
    assert table["predicted"].is_monotonic_increasing
    assert table["actual"].iloc[-1] == 1.0
    assert table["n"].sum() == len(y)


def test_bootstrap_range_of_a_perfect_model_is_one():
    y = np.array([0] * 20 + [1] * 20)
    p = np.linspace(0, 1, 40)
    assert bootstrap_auc_range(y, p, n=50, seed=0) == pytest.approx((1.0, 1.0))


def test_bootstrap_range_is_ordered_and_reproducible():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.3 + rng.random(200) * 0.7, 0, 1)
    low, high = bootstrap_auc_range(y, p, n=200, seed=0)
    assert low <= high
    assert (low, high) == bootstrap_auc_range(y, p, n=200, seed=0)


def test_age_band_boundaries():
    bands = age_band(pd.Series([21.0, 29.9, 30.0, 44.9, 45.0, 69.0])).tolist()
    assert bands == ["Under 30", "Under 30", "30-45", "30-45", "45+", "45+"]


def test_subgroup_auc_is_computed_within_each_group():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])  # perfect within A and within B, imperfect overall
    groups = pd.Series(["A", "A", "B", "B"])
    assert subgroup_auc(y, p, groups, ["A", "B"]) == {"A": 1.0, "B": 1.0}


def test_approval_by_group_approves_the_safest_overall():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    table = approval_by_group(y, scores, pd.Series(["A", "A", "B", "B"]), ["A", "B"], share=0.5)
    a, b = table.set_index("group").loc["A"], table.set_index("group").loc["B"]
    assert a["approval_rate"] == 1.0 and b["approval_rate"] == 0.0
    assert a["default_rate_approved"] == 0.5
    assert math.isnan(b["default_rate_approved"])  # nobody approved, so no rate, no crash
    assert b["default_rate_overall"] == 0.5


def test_gains_curve_runs_from_zero_to_everyone():
    y = np.array([0, 1, 0, 1, 0])
    curve = gains_curve(y, np.array([0.1, 0.9, 0.2, 0.8, 0.3]), points=6)
    assert curve["share_found"].iloc[0] == 0.0
    assert curve["share_found"].iloc[-1] == 1.0
    assert curve["share_found"].is_monotonic_increasing
    assert curve["share_found"].iloc[2] == 1.0  # both defaulters are in the riskiest 40%


def test_downsample_curve_keeps_the_end_points():
    x, y = np.arange(1000), np.arange(1000) * 2
    xs, ys = downsample_curve(x, y, points=50)
    assert len(xs) <= 50
    assert (xs[0], xs[-1]) == (0, 999)
    assert np.array_equal(ys, xs * 2)
