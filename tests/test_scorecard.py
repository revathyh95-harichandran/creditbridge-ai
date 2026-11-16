"""Tests for the scorecard's per-applicant reasons (src/scorecard.py)."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from fake_data import fake_applications
from pipeline import build_pipeline
from scorecard import readable, reasons

FIELDS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "CREDIT_TO_GOODS", "EMPLOYED_YEARS"]


@pytest.fixture(scope="module")
def fitted():
    df = fake_applications(n=1500)
    X, y = df.drop(columns="TARGET"), df["TARGET"]
    model = build_pipeline(LogisticRegression(max_iter=2000), imputer="median+flags", columns=FIELDS).fit(X, y)
    return model, X


def test_reasons_add_up_to_the_models_log_odds(fitted):
    model, X = fitted
    one = X.iloc[[0]]
    push = reasons(model, one)
    expected = model.decision_function(one)[0] - model.steps[-1][1].intercept_[0]
    assert push.sum() == pytest.approx(expected)


def test_flag_labels_follow_the_applicants_own_value(fitted):
    model, X = fitted
    has = X[X["EXT_SOURCE_1"].notna()].iloc[[0]]
    missing = X[X["EXT_SOURCE_1"].isna()].iloc[[0]]
    assert "External score 1 available" in reasons(model, has).index
    assert "External score 1 not available" in reasons(model, missing).index


def test_readable_labels():
    assert readable("EXT_SOURCE_3") == "External credit score 3"
    assert readable("missingindicator_EMPLOYED_YEARS", 1) == "Not employed"
    assert readable("missingindicator_EMPLOYED_YEARS", 0) == "Employed"
    assert np.isfinite(len(readable("SOMETHING_ELSE")))
