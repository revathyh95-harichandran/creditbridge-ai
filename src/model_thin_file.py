"""Can we score thin-file applicants without any credit bureau data?

The segment analysis showed thin-file applicants default more (9.7% vs 7.6%) but are
fully documented on everything except credit history. This script tests whether that
non-bureau data is enough to tell the safer thin-file applicants from the riskier ones.

Population: thin-file applicants only (applicant_segments.segment = 'thin-file').

Features used (all available at application time, none from a credit bureau):
  * the loan itself: amount, annuity, goods price, contract type, and ratios to income
  * income, income type, education, occupation, employer type
  * age, employment length (+ the retired flag from src/clean.py), housing type
  * assets: owns car, owns realty
  * household: children, family size
  * contact/residence: phone/email flags, region rating, city mismatch flags
  * history with THIS lender (previous_application): number of past applications,
    how many were approved / refused

Deliberately left out:
  * EXT_SOURCE_1/2/3 and their is_missing flags: external credit scores, and their
    absence is part of what *defines* thin-file in the first place
  * AMT_REQ_CREDIT_BUREAU_*: counts of credit bureau enquiries, i.e. bureau data
  * CODE_GENDER, NAME_FAMILY_STATUS: protected characteristics under fair-lending law
    in many countries (e.g. the US Equal Credit Opportunity Act). Age is kept because it
    is a standard, legally permitted factor in many scorecards, but any real use would
    need a fairness/legal review.
  * building descriptors (*_AVG / *_MODE / *_MEDI) and OWN_CAR_AGE: 50-70% missing in
    the raw data, so after median imputation they mostly carry noise

Leak-free since the ML-pipeline upgrade: rows are read RAW from application_train (only the
fixed 365243 rule is applied in SQL); capping, median filling and scaling are learned inside
the model pipeline, on training rows only. Earlier versions read the pre-cleaned table, whose
medians and caps had been computed on all rows, including the test rows.

Two models are compared with 3-fold cross-validation on the training set: a logistic
regression (simple, explainable) and a gradient-boosted tree model. The better one is
then scored once on a held-out 25% test set it has never seen.

For context, the same logistic regression is also fitted on thick-file applicants twice:
once with the same non-bureau features, and once with EXT_SOURCE_1/2/3 added. That shows
(a) whether thin-file applicants are any harder to score than thick-file ones on the same
information, and (b) how much the external credit scores are actually worth.

    python src/model_thin_file.py
"""

from __future__ import annotations

import math
import sys

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from config import DUCKDB_PATH
from pipeline import OutlierCapper

SEED = 42
TEST_SHARE = 0.25

CATEGORICAL_FEATURES = [
    "NAME_CONTRACT_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
]

NUMERIC_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "credit_to_income",
    "annuity_to_income",
    "credit_to_goods",
    "age_years",
    "employment_years",
    "DAYS_EMPLOYED_IS_RETIRED",
    "years_since_registration",
    "years_since_id_change",
    "years_since_phone_change",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "REGION_POPULATION_RELATIVE",
    "REGION_RATING_CLIENT",
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "FLAG_DOCUMENT_3",
    "prev_app_count",
    "prev_approved",
    "prev_refused",
    "prev_refused_share",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Columns that must never be model inputs (see module docstring). Checked by the tests.
EXCLUDED_PREFIXES = ("EXT_SOURCE", "AMT_REQ_CREDIT_BUREAU", "CODE_GENDER", "NAME_FAMILY_STATUS")

# Only used by the thick-file benchmark, never by the thin-file model.
EXT_SOURCE_FEATURES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

LOAD_QUERY = """
    WITH prev AS (
        SELECT
            SK_ID_CURR,
            COUNT(*) AS prev_app_count,
            SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 ELSE 0 END) AS prev_approved,
            SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1 ELSE 0 END) AS prev_refused
        FROM previous_application
        GROUP BY SK_ID_CURR
    )
    SELECT
        c.SK_ID_CURR, c.TARGET,
        -- fixed rule (learns nothing): 365243 means "not employed", not a number of days
        CASE WHEN c.DAYS_EMPLOYED = 365243 THEN 1 ELSE 0 END AS DAYS_EMPLOYED_IS_RETIRED,
        CAST(c.EXT_SOURCE_1 AS DOUBLE) AS EXT_SOURCE_1,
        CAST(c.EXT_SOURCE_2 AS DOUBLE) AS EXT_SOURCE_2,
        CAST(c.EXT_SOURCE_3 AS DOUBLE) AS EXT_SOURCE_3,
        c.NAME_CONTRACT_TYPE, c.FLAG_OWN_CAR, c.FLAG_OWN_REALTY, c.NAME_INCOME_TYPE,
        c.NAME_EDUCATION_TYPE, c.NAME_HOUSING_TYPE, c.OCCUPATION_TYPE, c.ORGANIZATION_TYPE,
        CAST(c.AMT_INCOME_TOTAL AS DOUBLE) AS AMT_INCOME_TOTAL,
        CAST(c.AMT_CREDIT AS DOUBLE) AS AMT_CREDIT,
        CAST(c.AMT_ANNUITY AS DOUBLE) AS AMT_ANNUITY,
        CAST(c.AMT_GOODS_PRICE AS DOUBLE) AS AMT_GOODS_PRICE,
        CAST(c.DAYS_BIRTH AS DOUBLE) AS DAYS_BIRTH,
        CAST(CASE WHEN c.DAYS_EMPLOYED = 365243 THEN NULL ELSE c.DAYS_EMPLOYED END AS DOUBLE) AS DAYS_EMPLOYED,
        CAST(c.DAYS_REGISTRATION AS DOUBLE) AS DAYS_REGISTRATION,
        CAST(c.DAYS_ID_PUBLISH AS DOUBLE) AS DAYS_ID_PUBLISH,
        CAST(c.DAYS_LAST_PHONE_CHANGE AS DOUBLE) AS DAYS_LAST_PHONE_CHANGE,
        CAST(c.CNT_CHILDREN AS DOUBLE) AS CNT_CHILDREN,
        CAST(c.CNT_FAM_MEMBERS AS DOUBLE) AS CNT_FAM_MEMBERS,
        CAST(c.REGION_POPULATION_RELATIVE AS DOUBLE) AS REGION_POPULATION_RELATIVE,
        CAST(c.REGION_RATING_CLIENT AS DOUBLE) AS REGION_RATING_CLIENT,
        c.FLAG_EMP_PHONE, c.FLAG_WORK_PHONE, c.FLAG_PHONE, c.FLAG_EMAIL,
        c.REG_CITY_NOT_LIVE_CITY, c.REG_CITY_NOT_WORK_CITY, c.FLAG_DOCUMENT_3,
        COALESCE(p.prev_app_count, 0) AS prev_app_count,
        COALESCE(p.prev_approved, 0) AS prev_approved,
        COALESCE(p.prev_refused, 0) AS prev_refused
    FROM application_train c
    JOIN applicant_segments s USING (SK_ID_CURR)
    LEFT JOIN prev p USING (SK_ID_CURR)
    WHERE s.segment = ?
    ORDER BY c.SK_ID_CURR
"""


def load_segment(con: duckdb.DuckDBPyConnection, segment: str) -> pd.DataFrame:
    return build_features(con.execute(LOAD_QUERY, [segment]).fetchdf())


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw columns into readable model inputs (years instead of negative days, ratios)."""
    out = df.copy()
    out["credit_to_income"] = out["AMT_CREDIT"] / out["AMT_INCOME_TOTAL"]
    out["annuity_to_income"] = out["AMT_ANNUITY"] / out["AMT_INCOME_TOTAL"]
    out["credit_to_goods"] = out["AMT_CREDIT"] / out["AMT_GOODS_PRICE"]
    out["age_years"] = -out["DAYS_BIRTH"] / 365.25
    out["employment_years"] = -out["DAYS_EMPLOYED"] / 365.25
    out["years_since_registration"] = -out["DAYS_REGISTRATION"] / 365.25
    out["years_since_id_change"] = -out["DAYS_ID_PUBLISH"] / 365.25
    out["years_since_phone_change"] = -out["DAYS_LAST_PHONE_CHANGE"] / 365.25
    out["prev_refused_share"] = np.where(
        out["prev_app_count"] > 0, out["prev_refused"] / out["prev_app_count"].clip(lower=1), 0.0
    )
    for col in NUMERIC_FEATURES:
        out[col] = out[col].astype(float)
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].fillna("Missing").astype(str)
    return out


def logistic_model(numeric: list[str] = NUMERIC_FEATURES) -> Pipeline:
    # Cap -> fill -> scale, all learned on the training rows passed to fit (no leakage).
    numeric_steps = Pipeline([
        ("cap", OutlierCapper()),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CATEGORICAL_FEATURES),
        ("num", numeric_steps, numeric),
    ])
    return Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000))])


def boosted_model() -> Pipeline:
    prep = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_FEATURES),
        ("num", "passthrough", NUMERIC_FEATURES),
    ])
    is_categorical = [True] * len(CATEGORICAL_FEATURES) + [False] * len(NUMERIC_FEATURES)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=100,
        l2_regularization=1.0, categorical_features=is_categorical, random_state=SEED,
    )
    return Pipeline([("prep", prep), ("model", model)])


def default_rate_of_safest(y_true: np.ndarray, risk_scores: np.ndarray, share: float) -> float:
    """Default rate among the `share` of applicants with the lowest predicted risk."""
    order = np.argsort(risk_scores, kind="stable")
    n = math.ceil(share * len(order))
    return float(np.mean(np.asarray(y_true)[order[:n]]))


def default_rate_of_riskiest(y_true: np.ndarray, risk_scores: np.ndarray, share: float) -> float:
    """Default rate among the `share` of applicants with the highest predicted risk."""
    order = np.argsort(-np.asarray(risk_scores), kind="stable")
    n = math.ceil(share * len(order))
    return float(np.mean(np.asarray(y_true)[order[:n]]))


def holdout_auc(df: pd.DataFrame, numeric: list[str]) -> float:
    """Test AUC of the logistic regression on one segment, same split rules as the main model."""
    X, y = df[CATEGORICAL_FEATURES + numeric], df["TARGET"].astype(int).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SHARE, stratify=y, random_state=SEED
    )
    model = logistic_model(numeric).fit(X_train, y_train)
    return float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))


def thick_file_benchmark(con: duckdb.DuckDBPyConnection) -> dict:
    thick = load_segment(con, "thick-file")
    return {
        "same features": holdout_auc(thick, NUMERIC_FEATURES),
        "same features + EXT_SOURCE scores": holdout_auc(thick, NUMERIC_FEATURES + EXT_SOURCE_FEATURES),
    }


def run(con: duckdb.DuckDBPyConnection) -> dict:
    df = load_segment(con, "thin-file")
    X, y = df[FEATURES], df["TARGET"].astype(int).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SHARE, stratify=y, random_state=SEED
    )

    candidates = {"logistic regression": logistic_model(), "gradient boosting": boosted_model()}
    cv_auc = {
        name: float(cross_val_score(model, X_train, y_train, cv=3, scoring="roc_auc").mean())
        for name, model in candidates.items()
    }
    best_name = max(cv_auc, key=cv_auc.get)
    best = candidates[best_name].fit(X_train, y_train)
    scores = best.predict_proba(X_test)[:, 1]

    importance = permutation_importance(
        best, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=SEED
    )
    top_features = sorted(
        zip(FEATURES, importance.importances_mean), key=lambda kv: kv[1], reverse=True
    )[:10]

    return {
        "n_applicants": len(df),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "test_default_rate": float(y_test.mean()),
        "cv_auc": cv_auc,
        "best_model": best_name,
        "test_auc": float(roc_auc_score(y_test, scores)),
        "safest": {s: default_rate_of_safest(y_test, scores, s) for s in (0.25, 0.50, 0.75)},
        "riskiest_10": default_rate_of_riskiest(y_test, scores, 0.10),
        "top_features": top_features,
        "thick_file_benchmark": thick_file_benchmark(con),
    }


def main() -> int:
    if not DUCKDB_PATH.exists():
        print(f"No database at {DUCKDB_PATH}.\nRun: python src/load_to_duckdb.py", file=sys.stderr)
        return 1

    with duckdb.connect(str(DUCKDB_PATH), read_only=True) as con:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        missing = {"application_train", "applicant_segments", "previous_application"} - tables
        if missing:
            print(f"Missing tables: {missing}.\nRun: python src/clean.py && python src/segment.py", file=sys.stderr)
            return 1
        r = run(con)

    print(f"Thin-file applicants: {r['n_applicants']:,}  (train {r['n_train']:,} / test {r['n_test']:,})")
    print(f"Test-set default rate (no model): {r['test_default_rate']:.2%}\n")

    print("3-fold cross-validated AUC on the training set:")
    for name, auc in r["cv_auc"].items():
        print(f"  {name:<20} {auc:.3f}")
    print(f"\nChosen: {r['best_model']}  ->  held-out test AUC = {r['test_auc']:.3f}\n")

    print("If we only approved the applicants the model rates safest:")
    for share, rate in r["safest"].items():
        print(f"  safest {share:>4.0%}  default rate {rate:.2%}  (vs {r['test_default_rate']:.2%} for everyone)")
    print(f"  riskiest 10%  default rate {r['riskiest_10']:.2%}\n")

    print("Most useful features (drop in test AUC when the feature is shuffled):")
    for feature, drop in r["top_features"]:
        print(f"  {feature:<28} {drop:.4f}")

    print("\nBenchmark - same logistic regression on thick-file applicants (test AUC):")
    for name, auc in r["thick_file_benchmark"].items():
        print(f"  {name:<36} {auc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
