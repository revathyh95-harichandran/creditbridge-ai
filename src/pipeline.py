"""Leak-free preparation for every model in the project.

Earlier versions of this project cleaned the data in DuckDB (src/clean.py) *before* the
train/test split: the medians and the 1st/99th-percentile caps were computed on all
307,511 rows, so test rows quietly shaped the training data. src/clean.py is still used
for the descriptive work (segments, dashboard), but every model now goes through this
pipeline instead, which learns nothing outside the training rows it is fitted on.

Two kinds of steps:

1. `fix_known_issues` - fixed rules from the data-quality pass. They learn nothing, so they
   are safe to apply to any row (365243 code -> blank + flag, days -> years, ratios ...).
2. Transformer classes that DO learn from data (which columns to drop, where to cap, which
   columns are skewed, medians/imputation, scaling, categories). Inside a scikit-learn
   Pipeline they only ever learn from the rows passed to `fit`.

Choices below were each compared under 5-fold cross-validation (src/prep_experiments.py).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (switches IterativeImputer on)
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from data import PROTECTED_COLS, SEED

DAYS_EMPLOYED_CODE = 365243        # "not employed" (mostly pensioners), not a length of employment
CAR_AGE_PLACEHOLDERS = (64, 65)    # 2,636 car owners pile up at exactly 64/65 years
DAYS_TO_YEARS = {                  # negative days before application -> positive years
    "DAYS_BIRTH": "AGE_YEARS",
    "DAYS_EMPLOYED": "EMPLOYED_YEARS",
    "DAYS_REGISTRATION": "REGISTRATION_YEARS",
    "DAYS_ID_PUBLISH": "ID_CHANGE_YEARS",
    "DAYS_LAST_PHONE_CHANGE": "PHONE_CHANGE_YEARS",
}
# Columns whose blanks are explained, so the "too empty" rule must not drop them.
KEEP_DESPITE_BLANKS = ("EXT_SOURCE_1", "OWN_CAR_AGE")


# --- 1. Fixed rules (learn nothing) ------------------------------------------------------
def fix_known_issues(df):
    """Apply the data-quality fixes. Returns a new table; never touches the input."""
    out = df.drop(columns=["SK_ID_CURR", "TARGET"] + PROTECTED_COLS, errors="ignore").copy()

    out["NOT_EMPLOYED"] = (out["DAYS_EMPLOYED"] == DAYS_EMPLOYED_CODE).astype(int)
    out.loc[out["NOT_EMPLOYED"] == 1, "DAYS_EMPLOYED"] = np.nan
    out["ORGANIZATION_TYPE"] = out["ORGANIZATION_TYPE"].replace("XNA", "No employer")

    out["CAR_AGE_PLACEHOLDER"] = out["OWN_CAR_AGE"].isin(CAR_AGE_PLACEHOLDERS).astype(int)
    out.loc[out["CAR_AGE_PLACEHOLDER"] == 1, "OWN_CAR_AGE"] = np.nan

    for days, years in DAYS_TO_YEARS.items():
        out[years] = -out.pop(days) / 365.25

    out["CREDIT_TO_INCOME"] = out["AMT_CREDIT"] / out["AMT_INCOME_TOTAL"]
    out["ANNUITY_TO_INCOME"] = out["AMT_ANNUITY"] / out["AMT_INCOME_TOTAL"]
    out["CREDIT_TO_GOODS"] = out["AMT_CREDIT"] / out["AMT_GOODS_PRICE"]
    return out


def numeric_columns(df):
    return df.select_dtypes("number").columns.tolist()


def continuous_columns(df):
    """Numeric columns with more than 2 distinct values (0/1 flags are left alone)."""
    return [c for c in numeric_columns(df) if df[c].nunique() > 2]


# --- 2. Steps that learn from the rows they are fitted on --------------------------------
class RoughReducer(BaseEstimator, TransformerMixin):
    """Drop columns by three stated rules, learned on training rows only.

    1. too empty:      more than `max_missing` blank (except explained blanks)
    2. near-constant:  one value covers at least `max_top_share` of rows
    3. near-duplicate: |correlation| above `max_corr` with an earlier column
    """

    def __init__(self, max_missing=0.5, max_top_share=0.99, max_corr=0.95, keep=KEEP_DESPITE_BLANKS):
        self.max_missing = max_missing
        self.max_top_share = max_top_share
        self.max_corr = max_corr
        self.keep = keep

    def fit(self, X, y=None):
        self.dropped_ = {}
        missing = X.isna().mean()
        for col in X.columns:
            if missing[col] > self.max_missing and col not in self.keep:
                self.dropped_[col] = "too empty"
        for col in [c for c in X.columns if c not in self.dropped_]:
            if X[col].value_counts(normalize=True, dropna=False).iloc[0] >= self.max_top_share:
                self.dropped_[col] = "near-constant"
        num = [c for c in numeric_columns(X) if c not in self.dropped_]
        corr = X[num].corr().abs()
        for i, a in enumerate(num):        # keep the first of each pair, drop the later one
            if a in self.dropped_:
                continue
            for b in num[i + 1:]:
                if b not in self.dropped_ and corr.loc[a, b] > self.max_corr:
                    self.dropped_[b] = f"near-duplicate of {a}"
        self.kept_ = [c for c in X.columns if c not in self.dropped_]
        return self

    def transform(self, X):
        return X[self.kept_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.kept_)


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Cap (never delete) extreme values at percentiles learned on training rows."""

    def __init__(self, lower=0.001, upper=0.999):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self.columns_ = continuous_columns(X)
        self.low_ = X[self.columns_].quantile(self.lower)
        self.high_ = X[self.columns_].quantile(self.upper)
        return self

    def transform(self, X):
        X = X.copy()
        X[self.columns_] = X[self.columns_].clip(self.low_, self.high_, axis=1)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


class SkewFixer(BaseEstimator, TransformerMixin):
    """log(1 + x) for non-negative continuous columns with skewness above `threshold`."""

    def __init__(self, threshold=1.0, active=True):
        self.threshold = threshold
        self.active = active

    def fit(self, X, y=None):
        cont = continuous_columns(X)
        skew = X[cont].skew()
        self.columns_ = [c for c in cont if self.active and skew[c] > self.threshold and X[c].min() >= 0]
        return self

    def transform(self, X):
        X = X.copy()
        X[self.columns_] = np.log1p(X[self.columns_])
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


class ColumnPicker(BaseEstimator, TransformerMixin):
    """Keep only the listed columns (used for the small scorecard)."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[list(self.columns)]


def raw_feature(name, raw_columns):
    """Map a prepared column back to the input field it came from.

    'missingindicator_EXT_SOURCE_1' -> EXT_SOURCE_1,
    'NAME_EDUCATION_TYPE_Higher education' -> NAME_EDUCATION_TYPE.
    """
    name = name.removeprefix("missingindicator_")
    matches = [c for c in raw_columns if name == c or name.startswith(c + "_")]
    return max(matches, key=len) if matches else name


# --- 3. Putting it together ----------------------------------------------------------------
def make_imputer(method):
    if method == "median":            # plain median: the model can't tell filled from real
        return SimpleImputer(strategy="median")
    if method == "median+flags":      # + a 0/1 "was missing" column per gappy column
        return SimpleImputer(strategy="median", add_indicator=True)
    if method == "iterative+flags":   # predict each blank from the 10 most related columns
        return IterativeImputer(max_iter=5, n_nearest_features=10, random_state=SEED,
                                add_indicator=True, skip_complete=True)
    raise ValueError(method)


# Chosen by 5-fold CV (src/prep_experiments.py): flags win in 5/5 folds; iterative adds a little.
CHOSEN = dict(imputer="iterative+flags", skew=True, cap=(0.001, 0.999), reduce=True)


def build_preparation(imputer="iterative+flags", skew=True, cap=(0.001, 0.999), reduce=True, columns=None):
    """All preparation steps, in the order they have to run (no model)."""
    numeric = make_column_selector(dtype_include="number")
    text = make_column_selector(dtype_exclude="number")
    encode = ColumnTransformer([
        ("num", Pipeline([("impute", make_imputer(imputer)), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            # categories under 0.5% of rows are pooled, so rare employer types don't each get a weight
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005,
                                     sparse_output=False)),
        ]), text),
    ], verbose_feature_names_out=False)
    return Pipeline([
        ("fix", FunctionTransformer(fix_known_issues)),
        ("pick", ColumnPicker(columns) if columns else "passthrough"),
        ("reduce", RoughReducer() if reduce else "passthrough"),
        ("cap", OutlierCapper(*cap) if cap else "passthrough"),
        ("skew", SkewFixer(active=skew)),
        ("encode", encode),
    ])


def build_pipeline(model=None, **settings):
    """Preparation + model. Default model: logistic regression with scikit-learn defaults."""
    model = model if model is not None else LogisticRegression(max_iter=2000)
    prep = build_preparation(**{**CHOSEN, **settings})
    return Pipeline(prep.steps + [("model", model)])
