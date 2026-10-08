"""Tests for src/model_thin_file.py that don't need the real database.

The important guarantees: no bureau data, external scores, or protected characteristics
ever become model inputs; the derived features are computed correctly; and the
"safest X% default rate" numbers in the report are calculated the way they're described.
"""

import numpy as np
import pandas as pd
import pytest

from model_thin_file import (
    CATEGORICAL_FEATURES,
    EXCLUDED_PREFIXES,
    FEATURES,
    NUMERIC_FEATURES,
    build_features,
    default_rate_of_riskiest,
    default_rate_of_safest,
)


def raw_row(**overrides):
    row = {
        "NAME_CONTRACT_TYPE": "Cash loans", "FLAG_OWN_CAR": "N", "FLAG_OWN_REALTY": "Y",
        "NAME_INCOME_TYPE": "Working", "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_HOUSING_TYPE": "House / apartment", "OCCUPATION_TYPE": "Missing",
        "ORGANIZATION_TYPE": "Business Entity Type 3",
        "AMT_INCOME_TOTAL": 100_000.0, "AMT_CREDIT": 300_000.0, "AMT_ANNUITY": 15_000.0,
        "AMT_GOODS_PRICE": 250_000.0,
        "DAYS_BIRTH": -365.25 * 40, "DAYS_EMPLOYED": -365.25 * 5, "DAYS_EMPLOYED_IS_RETIRED": 0,
        "DAYS_REGISTRATION": -365.25 * 10, "DAYS_ID_PUBLISH": -365.25 * 3,
        "DAYS_LAST_PHONE_CHANGE": -365.25 * 2,
        "CNT_CHILDREN": 1, "CNT_FAM_MEMBERS": 3, "REGION_POPULATION_RELATIVE": 0.02,
        "REGION_RATING_CLIENT": 2, "FLAG_EMP_PHONE": 1, "FLAG_WORK_PHONE": 0, "FLAG_PHONE": 1,
        "FLAG_EMAIL": 0, "REG_CITY_NOT_LIVE_CITY": 0, "REG_CITY_NOT_WORK_CITY": 0,
        "FLAG_DOCUMENT_3": 1, "prev_app_count": 4, "prev_approved": 3, "prev_refused": 1,
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize("feature", FEATURES)
def test_no_bureau_score_or_protected_feature_is_used(feature):
    assert not feature.startswith(EXCLUDED_PREFIXES)


def test_feature_lists_do_not_overlap():
    assert not set(CATEGORICAL_FEATURES) & set(NUMERIC_FEATURES)


def test_build_features_ratios_and_years():
    out = build_features(pd.DataFrame([raw_row()])).iloc[0]
    assert out["credit_to_income"] == pytest.approx(3.0)
    assert out["annuity_to_income"] == pytest.approx(0.15)
    assert out["credit_to_goods"] == pytest.approx(1.2)
    assert out["age_years"] == pytest.approx(40.0)
    assert out["employment_years"] == pytest.approx(5.0)
    assert out["years_since_phone_change"] == pytest.approx(2.0)
    assert out["prev_refused_share"] == pytest.approx(0.25)


def test_prev_refused_share_is_zero_with_no_previous_applications():
    row = raw_row(prev_app_count=0, prev_approved=0, prev_refused=0)
    out = build_features(pd.DataFrame([row])).iloc[0]
    assert out["prev_refused_share"] == 0.0


def test_build_features_produces_every_model_input():
    out = build_features(pd.DataFrame([raw_row()]))
    assert set(FEATURES) <= set(out.columns)
    assert not out[FEATURES].isna().any().any()


def test_default_rate_of_safest_uses_lowest_scores():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert default_rate_of_safest(y, scores, 0.5) == 0.0
    assert default_rate_of_safest(y, scores, 1.0) == 0.5


def test_default_rate_of_riskiest_uses_highest_scores():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert default_rate_of_riskiest(y, scores, 0.5) == 1.0


def test_safest_share_rounds_up_so_it_never_selects_nobody():
    y = np.array([1, 0, 0])
    scores = np.array([0.1, 0.5, 0.9])
    assert default_rate_of_safest(y, scores, 0.1) == 1.0  # ceil(0.3) = 1 applicant
