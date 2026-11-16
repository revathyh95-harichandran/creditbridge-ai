"""Tests for src/pipeline.py: the fixed rules, the drop rules, and - most importantly - that
every learned step learns only from the rows it is fitted on (no leakage)."""

import numpy as np
import pandas as pd
import pytest

from data import PROTECTED_COLS
from fake_data import fake_applications
from pipeline import (
    OutlierCapper,
    RoughReducer,
    build_pipeline,
    build_preparation,
    fix_known_issues,
    raw_feature,
)


def test_fix_known_issues_turns_the_365243_code_into_a_blank_and_a_flag():
    df = fake_applications()
    out = fix_known_issues(df)
    code = df["DAYS_EMPLOYED"] == 365243
    assert out.loc[code, "NOT_EMPLOYED"].eq(1).all()
    assert out.loc[code, "EMPLOYED_YEARS"].isna().all()
    assert (out.loc[~code, "EMPLOYED_YEARS"] > 0).all()        # days before application -> positive years


def test_fix_known_issues_flags_car_age_placeholders_and_renames_xna():
    out = fix_known_issues(fake_applications())
    assert not out["OWN_CAR_AGE"].isin([64, 65]).any()
    assert out["CAR_AGE_PLACEHOLDER"].sum() > 0
    assert "XNA" not in set(out["ORGANIZATION_TYPE"])


def test_fix_known_issues_never_uses_protected_columns_or_ids_and_leaves_input_alone():
    df = fake_applications()
    before = df.copy()
    out = fix_known_issues(df)
    for col in PROTECTED_COLS + ["SK_ID_CURR", "TARGET"]:
        assert col not in out.columns
    pd.testing.assert_frame_equal(df, before)


def test_caps_are_learned_from_training_rows_only():
    df = fix_known_issues(fake_applications(n=1000))
    train, test = df.iloc[:800].copy(), df.iloc[800:].copy()
    test.loc[test.index[0], "AMT_INCOME_TOTAL"] = 1e9          # a typo that exists only in the test rows
    capper = OutlierCapper().fit(train)
    assert capper.high_["AMT_INCOME_TOTAL"] == pytest.approx(train["AMT_INCOME_TOTAL"].quantile(0.999))
    capped = capper.transform(test)
    assert capped["AMT_INCOME_TOTAL"].max() <= capper.high_["AMT_INCOME_TOTAL"]


def test_imputer_medians_come_from_training_rows_only():
    df = fake_applications(n=1000)
    train, test = df.iloc[:800], df.iloc[800:].copy()
    test["EXT_SOURCE_2"] = 0.99                                 # a test-only shift must not move the median
    prep = build_preparation(imputer="median+flags", skew=False, cap=None, reduce=False).fit(train)
    imputer = prep.named_steps["encode"].named_transformers_["num"].named_steps["impute"]
    names = list(imputer.feature_names_in_)
    learned = imputer.statistics_[names.index("EXT_SOURCE_2")]
    assert learned == pytest.approx(fix_known_issues(train)["EXT_SOURCE_2"].median())


def test_rough_reducer_states_a_rule_for_every_dropped_column():
    df = fix_known_issues(fake_applications())
    df["MOSTLY_EMPTY"] = np.where(np.arange(len(df)) < 0.8 * len(df), np.nan, 1.0)
    df["COPY_OF_CREDIT"] = df["AMT_CREDIT"] * 2 + 1
    r = RoughReducer().fit(df)
    assert r.dropped_["MOSTLY_EMPTY"] == "too empty"
    assert r.dropped_["FLAG_MOBIL"] == "near-constant"
    assert r.dropped_["COPY_OF_CREDIT"].startswith("near-duplicate")
    assert "EXT_SOURCE_1" in r.kept_                            # blank > 50%, but an explained exception


def test_full_pipeline_gives_probabilities_and_survives_unseen_categories():
    df = fake_applications(n=800)
    train, new = df.iloc[:600], df.iloc[600:].copy()
    new.loc[new.index[:5], "ORGANIZATION_TYPE"] = "Never seen before"
    model = build_pipeline(imputer="median+flags").fit(train.drop(columns="TARGET"), train["TARGET"])
    p = model.predict_proba(new.drop(columns="TARGET"))[:, 1]
    assert len(p) == len(new) and np.all((p >= 0) & (p <= 1))


def test_raw_feature_maps_prepared_columns_back_to_their_field():
    raw = ["EXT_SOURCE_1", "NAME_EDUCATION_TYPE", "AMT_CREDIT"]
    assert raw_feature("missingindicator_EXT_SOURCE_1", raw) == "EXT_SOURCE_1"
    assert raw_feature("NAME_EDUCATION_TYPE_Higher education", raw) == "NAME_EDUCATION_TYPE"
    assert raw_feature("AMT_CREDIT", raw) == "AMT_CREDIT"
