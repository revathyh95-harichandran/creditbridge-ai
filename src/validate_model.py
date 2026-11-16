"""Evaluate and stress-test the thin-file model from src/model_thin_file.py.

Same setup as the main model: thin-file applicants only, logistic regression, the same
features, a 75/25 stratified split with random_state=42. Every number in the README's
model-evaluation section comes from this script.

Checks:
  * accuracy trap        - a "nobody defaults" model vs ours at the 0.5 cut-off
  * flagging the riskiest 10/25/50% - precision, recall, F1
  * PR-AUC               - vs the random baseline (= the default rate)
  * Gini and KS          - the numbers banks usually quote
  * calibration          - Brier score, log loss, predicted vs actual per risk group
  * overfitting          - train AUC vs test AUC
  * stability            - AUC across 5 random splits + a bootstrap range on the test set
  * leakage              - previous applications must all be decided before today
  * simple-rule baseline - a 3-factor logistic regression
  * subgroups            - AUC by income type, age band and gender
  * fairness             - who gets approved if we take the safest 50%

Gender (CODE_GENDER) is loaded separately and used ONLY to report subgroup AUC and the
fairness table. It is never passed to the model - see EXCLUDED_PREFIXES in
model_thin_file.py and tests/test_model.py.

    python src/validate_model.py
"""

from __future__ import annotations

import math
import sys

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DUCKDB_PATH
from model_thin_file import (
    FEATURES,
    SEED,
    TEST_SHARE,
    default_rate_of_riskiest,
    default_rate_of_safest,
    load_segment,
    logistic_model,
    thick_file_benchmark,
)

FLAG_SHARES = (0.10, 0.25, 0.50)
APPROVE_SHARE = 0.50
N_RISK_GROUPS = 10
STABILITY_SEEDS = range(5)
N_BOOTSTRAP = 1000
SIMPLE_RULE_CATEGORICAL = ["NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE"]
SIMPLE_RULE_NUMERIC = ["REGION_RATING_CLIENT"]
SUBGROUP_INCOME_TYPES = ["Commercial associate", "State servant", "Working", "Pensioner"]
AGE_BANDS = ["Under 30", "30-45", "45+"]
GENDERS = {"F": "Women", "M": "Men"}


# --- small, testable helpers -------------------------------------------------------

def accuracy_trap(y: np.ndarray, p: np.ndarray, cutoff: float = 0.5) -> dict:
    """Accuracy of predicting 'nobody defaults' vs our model at a probability cut-off."""
    y = np.asarray(y)
    flagged = np.asarray(p) >= cutoff
    return {
        "lazy_accuracy": float(np.mean(y == 0)),
        "model_accuracy": float(np.mean(flagged == (y == 1))),
        "n_flagged": int(flagged.sum()),
    }


def flag_riskiest(y: np.ndarray, scores: np.ndarray, share: float) -> dict:
    """Precision, recall and F1 if we flag the `share` of people with the highest scores."""
    y = np.asarray(y)
    n = math.ceil(share * len(y))
    flagged = np.zeros(len(y), dtype=bool)
    flagged[np.argsort(-np.asarray(scores), kind="stable")[:n]] = True
    hits = int(y[flagged].sum())
    precision = hits / n
    recall = hits / int(y.sum())
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    return {"share": share, "n_flagged": n, "precision": precision, "recall": recall, "f1": f1}


def people_to_find_share_of_defaulters(y: np.ndarray, scores: np.ndarray, share: float = 0.5) -> int:
    """How many people you check, riskiest first, before finding `share` of all defaulters."""
    y = np.asarray(y)
    target = math.ceil(share * y.sum())
    found = np.cumsum(y[np.argsort(-np.asarray(scores), kind="stable")])
    return int(np.argmax(found >= target) + 1)


def people_to_find_share_randomly(y: np.ndarray, share: float = 0.5) -> int:
    """Expected people checked in random order to find `share` of defaulters (no luck either way).

    Deliberately the expected value, not one simulated shuffle: a single shuffle swings by
    several hundred people depending on the seed (2,000 shuffles of the test set ranged
    8,162 to 8,934 around this 8,546).
    """
    y = np.asarray(y)
    return math.ceil(math.ceil(share * y.sum()) / y.mean())


def gini(auc: float) -> float:
    return 2 * auc - 1


def ks_statistic(y: np.ndarray, scores: np.ndarray) -> float:
    """Largest gap between the score distributions of defaulters and non-defaulters."""
    fpr, tpr, _ = roc_curve(y, scores)
    return float(np.max(tpr - fpr))


def calibration_table(y: np.ndarray, p: np.ndarray, n_groups: int = N_RISK_GROUPS) -> pd.DataFrame:
    """Equal-sized risk groups, safest first: mean predicted vs actual default rate.

    When the count doesn't divide evenly (17,092 / 10), np.array_split gives the leftover
    people to the safest groups.
    """
    y, p = np.asarray(y), np.asarray(p)
    groups = np.array_split(np.argsort(p, kind="stable"), n_groups)
    return pd.DataFrame([
        {"group": i, "n": len(g), "predicted": float(p[g].mean()), "actual": float(y[g].mean())}
        for i, g in enumerate(groups, start=1)
    ])


def bootstrap_auc_range(y: np.ndarray, p: np.ndarray, n: int = N_BOOTSTRAP, seed: int = 0) -> tuple[float, float]:
    """95% range of test AUC when the test set is resampled with replacement."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            aucs.append(roc_auc_score(y[idx], p[idx]))
    low, high = np.percentile(aucs, [2.5, 97.5])
    return float(low), float(high)


def age_band(age_years: pd.Series) -> pd.Series:
    return pd.cut(age_years, [0, 30, 45, np.inf], right=False, labels=AGE_BANDS).astype(str)


def subgroup_auc(y: np.ndarray, p: np.ndarray, groups: pd.Series, keep: list[str]) -> dict:
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    return {g: float(roc_auc_score(y[groups == g], p[groups == g])) for g in keep}


def approval_by_group(y: np.ndarray, scores: np.ndarray, groups: pd.Series, keep: list[str],
                      share: float = APPROVE_SHARE) -> pd.DataFrame:
    """Approve the safest `share` overall, then report each group's approval and default rates."""
    y, groups = np.asarray(y), np.asarray(groups)
    n = math.ceil(share * len(y))
    approved = np.zeros(len(y), dtype=bool)
    approved[np.argsort(np.asarray(scores), kind="stable")[:n]] = True
    rows = []
    for g in keep:
        in_group = groups == g
        approved_in_group = in_group & approved
        rows.append({
            "group": g,
            "n": int(in_group.sum()),
            "approval_rate": float(approved[in_group].mean()),
            "default_rate_approved": float(y[approved_in_group].mean()) if approved_in_group.any() else float("nan"),
            "default_rate_overall": float(y[in_group].mean()),
        })
    return pd.DataFrame(rows)


def gains_curve(y: np.ndarray, scores: np.ndarray, points: int = 101) -> pd.DataFrame:
    """% of defaulters found vs % of people checked, riskiest first, at evenly spaced checkpoints."""
    y = np.asarray(y)
    found = np.concatenate([[0], np.cumsum(y[np.argsort(-np.asarray(scores), kind="stable")])])
    checked = np.linspace(0, 1, points)
    idx = np.round(checked * len(y)).astype(int)
    return pd.DataFrame({"share_checked": checked, "share_found": found[idx] / y.sum()})


def downsample_curve(x: np.ndarray, y: np.ndarray, points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Thin a curve to evenly spaced points so the charts only carry summary-level detail."""
    keep = np.unique(np.linspace(0, len(x) - 1, points).round().astype(int))
    return np.asarray(x)[keep], np.asarray(y)[keep]


# --- the model pieces --------------------------------------------------------------

def split(df: pd.DataFrame, seed: int = SEED):
    return train_test_split(df, test_size=TEST_SHARE, stratify=df["TARGET"], random_state=seed)


def simple_rule_model() -> Pipeline:
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), SIMPLE_RULE_CATEGORICAL),
        ("num", StandardScaler(), SIMPLE_RULE_NUMERIC),
    ])
    return Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000))])


def load_thin_file_with_gender(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = load_segment(con, "thin-file")
    gender = con.execute("SELECT SK_ID_CURR, CODE_GENDER FROM application_train_clean").fetchdf()
    return df.merge(gender, on="SK_ID_CURR", how="left")


def run_validation(con: duckdb.DuckDBPyConnection) -> dict:
    df = load_thin_file_with_gender(con)
    train, test = split(df)
    y_train, y_test = train["TARGET"].to_numpy(), test["TARGET"].to_numpy()

    model = logistic_model().fit(train[FEATURES], y_train)
    p_test = model.predict_proba(test[FEATURES])[:, 1]
    p_train = model.predict_proba(train[FEATURES])[:, 1]
    auc = float(roc_auc_score(y_test, p_test))
    base_rate = float(y_train.mean())

    stability = {}
    for seed in STABILITY_SEEDS:
        tr, te = split(df, seed)
        m = logistic_model().fit(tr[FEATURES], tr["TARGET"])
        stability[seed] = float(roc_auc_score(te["TARGET"], m.predict_proba(te[FEATURES])[:, 1]))

    simple = simple_rule_model().fit(train[SIMPLE_RULE_CATEGORICAL + SIMPLE_RULE_NUMERIC], y_train)
    simple_auc = float(roc_auc_score(
        y_test, simple.predict_proba(test[SIMPLE_RULE_CATEGORICAL + SIMPLE_RULE_NUMERIC])[:, 1]))

    test_age_band = age_band(test["age_years"])
    test_gender = test["CODE_GENDER"].map(GENDERS).fillna("Other")
    fpr, tpr, _ = roc_curve(y_test, p_test)
    precision, recall, _ = precision_recall_curve(y_test, p_test)

    return {
        "n_test": len(y_test),
        "n_test_defaulted": int(y_test.sum()),
        "test_default_rate": float(y_test.mean()),
        "auc": auc,
        "gini": gini(auc),
        "ks": ks_statistic(y_test, p_test),
        "pr_auc": float(average_precision_score(y_test, p_test)),
        "accuracy_trap": accuracy_trap(y_test, p_test),
        "safest": {s: default_rate_of_safest(y_test, p_test, s) for s in (0.25, 0.50, 0.75)},
        "riskiest_10": default_rate_of_riskiest(y_test, p_test, 0.10),
        "flagging": [flag_riskiest(y_test, p_test, s) for s in FLAG_SHARES],
        "half_defaulters_model": people_to_find_share_of_defaulters(y_test, p_test, 0.5),
        "half_defaulters_random": people_to_find_share_randomly(y_test, 0.5),
        "brier": float(brier_score_loss(y_test, p_test)),
        "brier_base": float(brier_score_loss(y_test, np.full(len(y_test), base_rate))),
        "log_loss": float(log_loss(y_test, p_test)),
        "log_loss_base": float(log_loss(y_test, np.full(len(y_test), base_rate))),
        "calibration": calibration_table(y_test, p_test),
        "train_auc": float(roc_auc_score(y_train, p_train)),
        "stability": stability,
        "bootstrap_range": bootstrap_auc_range(y_test, p_test),
        "max_days_decision": int(con.execute("SELECT MAX(DAYS_DECISION) FROM previous_application").fetchone()[0]),
        "simple_rule_auc": simple_auc,
        "thick_file_benchmark": thick_file_benchmark(con),
        "subgroup_auc": {
            "income type": subgroup_auc(y_test, p_test, test["NAME_INCOME_TYPE"], SUBGROUP_INCOME_TYPES),
            "age band": subgroup_auc(y_test, p_test, test_age_band, AGE_BANDS),
            "gender": subgroup_auc(y_test, p_test, test_gender, list(GENDERS.values())),
        },
        "fairness": {
            "age band": approval_by_group(y_test, p_test, test_age_band, AGE_BANDS),
            "gender": approval_by_group(y_test, p_test, test_gender, list(GENDERS.values())),
        },
        "curves": {
            "roc": downsample_curve(fpr, tpr),
            "pr": downsample_curve(recall[::-1], precision[::-1]),
            "gains": gains_curve(y_test, p_test),
        },
    }


def main() -> int:
    if not DUCKDB_PATH.exists():
        print(f"No database at {DUCKDB_PATH}.\nRun: python src/load_to_duckdb.py", file=sys.stderr)
        return 1
    with duckdb.connect(str(DUCKDB_PATH), read_only=True) as con:
        r = run_validation(con)

    print(f"Test set: {r['n_test']:,} people, {r['n_test_defaulted']:,} defaulted ({r['test_default_rate']:.2%})\n")

    t = r["accuracy_trap"]
    print("Accuracy trap")
    print(f"  'nobody defaults' model   {t['lazy_accuracy']:.2%}")
    print(f"  our model at 0.5 cut-off  {t['model_accuracy']:.2%}  (flags only {t['n_flagged']} people)\n")

    print(f"Ranking:  AUC {r['auc']:.3f}   Gini {r['gini']:.3f}   KS {r['ks']:.3f}")
    print(f"PR-AUC {r['pr_auc']:.3f}  vs random baseline {r['test_default_rate']:.3f}\n")

    print("Flagging the riskiest:           precision  recall   F1")
    for f in r["flagging"]:
        print(f"  {f['share']:>4.0%} ({f['n_flagged']:>5,} people)        {f['precision']:>7.1%}  {f['recall']:>6.1%}  {f['f1']:.3f}")
    print(f"To find half of all defaulters: model order {r['half_defaulters_model']:,} people, "
          f"random order {r['half_defaulters_random']:,}\n")

    print("Calibration")
    print(f"  Brier    {r['brier']:.4f}  vs base rate {r['brier_base']:.4f}")
    print(f"  log loss {r['log_loss']:.3f}   vs base rate {r['log_loss_base']:.3f}")
    print("  risk group   predicted   actual")
    for _, row in r["calibration"].iterrows():
        print(f"  {int(row['group']):>10}   {row['predicted']:>8.2%}   {row['actual']:>6.2%}")
    print()

    print(f"Overfitting: train AUC {r['train_auc']:.3f} vs test AUC {r['auc']:.3f}")
    print("Stability: AUC by split seed " + "  ".join(f"{s}:{a:.3f}" for s, a in r["stability"].items()))
    low, high = r["bootstrap_range"]
    print(f"Bootstrap 95% range ({N_BOOTSTRAP} resamples): {low:.3f} to {high:.3f}")
    leak = r["max_days_decision"]
    print(f"Leakage: max DAYS_DECISION = {leak}  ({'OK, all before the application' if leak < 0 else 'PROBLEM'})")
    print(f"Simple 3-factor rule AUC: {r['simple_rule_auc']:.3f}")
    for name, a in r["thick_file_benchmark"].items():
        print(f"Thick-file benchmark, {name}: {a:.3f}")
    print()

    print("AUC within subgroups")
    for family, scores in r["subgroup_auc"].items():
        print(f"  {family:<12} " + "   ".join(f"{g}: {a:.3f}" for g, a in scores.items()))
    print()

    print(f"Fairness - approve the safest {APPROVE_SHARE:.0%}")
    print("  group       approval   default rate of approved   group's overall default rate")
    for table in r["fairness"].values():
        for _, row in table.iterrows():
            print(f"  {row['group']:<10} {row['approval_rate']:>9.1%}   {row['default_rate_approved']:>24.2%}"
                  f"   {row['default_rate_overall']:>28.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
