"""A small, readable scorecard, plus "reasons" for any single decision.

The full logistic regression uses ~150 prepared columns. It ranks applicants well, but its
weights are hard to read one by one because many inputs move together (credit amount and
annuity correlate at 0.85). So:

1. An L1 (lasso) path, fitted inside the same 5 folds, shows how many *input fields* each
   penalty keeps and how well the model still ranks.
2. The largest model with at most MAX_FIELDS fields (a form a loan officer can fill in
   within a minute) is refitted as an ordinary logistic regression: the scorecard.
3. Its weights become odds ratios, and `reasons()` explains any single applicant.

    python src/scorecard.py
"""

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import style
from compare_models import CHOICES_JSON, OOF_CSV
from config import REPORTS_DIR
from cv import evaluate_on_folds, prepare_folds, summarise
from data import SEED, load_train_test, split_xy
from pipeline import build_pipeline, build_preparation, raw_feature

MAX_FIELDS = 10
C_PATH = [0.0002, 0.0003, 0.0005, 0.0007, 0.001, 0.002, 0.005, 0.01]

READABLE = {
    "EXT_SOURCE_1": "External credit score 1", "EXT_SOURCE_2": "External credit score 2",
    "EXT_SOURCE_3": "External credit score 3", "CREDIT_TO_GOODS": "Loan amount vs goods price",
    "EMPLOYED_YEARS": "Years employed", "AGE_YEARS": "Age", "ANNUITY_TO_INCOME": "Repayment vs income",
    "CREDIT_TO_INCOME": "Loan amount vs income",
}
# For "was missing" flags the label depends on whether THIS applicant's value is missing.
FLAG_LABELS = {
    "EXT_SOURCE_1": ("External score 1 not available", "External score 1 available"),
    "EXT_SOURCE_2": ("External score 2 not available", "External score 2 available"),
    "EXT_SOURCE_3": ("External score 3 not available", "External score 3 available"),
    "CREDIT_TO_GOODS": ("No goods price given", "Goods price given"),
    "EMPLOYED_YEARS": ("Not employed", "Employed"),
}


def readable(name: str, was_missing: int | None = None) -> str:
    """Plain-English label for a prepared column."""
    if name.startswith("missingindicator_"):
        field = name.removeprefix("missingindicator_")
        missing, present = FLAG_LABELS.get(field, (f"{field} missing", f"{field} given"))
        return missing if was_missing == 1 else present
    return READABLE.get(name, name)


def reasons(model: Pipeline, applicant: pd.DataFrame) -> pd.Series:
    """How much each input pushes THIS applicant's log-odds away from the average applicant.

    For a logistic regression on standardised inputs this is exact: weight x standardised value.
    Positive = raises the risk. Index = plain-English labels.
    """
    prepared = Pipeline(model.steps[:-1]).transform(applicant)[0]
    names = list(model.steps[-2][1].get_feature_names_out())
    push = pd.Series(model.steps[-1][1].coef_[0] * prepared, index=names)
    before_fill = Pipeline(model.steps[:-2]).transform(applicant)   # raw values, blanks still blank
    labels = []
    for n in names:
        field = n.removeprefix("missingindicator_")
        was_missing = int(pd.isna(before_fill[field].iloc[0])) if field in before_fill else None
        labels.append(readable(n, was_missing))
    push.index = labels
    return push.sort_values()


def main() -> int:
    train, _ = load_train_test()
    X, y = split_xy(train)

    # 1. L1 path inside the folds
    folds = prepare_folds(build_preparation(), X, y)
    raw_cols = folds[0]["prep"].named_steps["reduce"].kept_
    names = folds[0]["names"]
    path, results = [], {}
    for C in C_PATH:
        table, _, fitted = evaluate_on_folds(LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                                                random_state=SEED), folds)
        fields = [len({raw_feature(n, raw_cols) for n, w in zip(names, m.coef_[0]) if w != 0}) for m in fitted]
        path.append({"C": C, "fields": float(np.mean(fields)), "ROC-AUC": table["ROC-AUC"].mean()})
        print(f"C={C:<7} fields {np.mean(fields):5.1f}  ROC-AUC {table['ROC-AUC'].mean():.4f}")
    path = pd.DataFrame(path)
    del folds
    app_C = float(path[path["fields"] <= MAX_FIELDS]["C"].max())

    # 2. Which fields, from one fit on all training rows
    prep = build_preparation().fit(X, y)
    all_raw = prep.named_steps["reduce"].kept_
    names = list(prep[-1].get_feature_names_out())
    l1 = LogisticRegression(penalty="l1", solver="liblinear", C=app_C, random_state=SEED).fit(prep.transform(X), y)
    weight = pd.Series(np.abs(l1.coef_[0]), index=names).groupby([raw_feature(n, all_raw) for n in names]).max()
    fields = list(weight[weight > 0].sort_values(ascending=False).index)
    print(f"Scorecard fields (C={app_C}): {fields}")

    # 3. Refit as an ordinary logistic regression; cross-validate it on the same folds
    sc_folds = prepare_folds(build_preparation(columns=fields), X, y)
    table, oof, _ = evaluate_on_folds(LogisticRegression(max_iter=2000), sc_folds)
    print(f"Scorecard ROC-AUC {table['ROC-AUC'].mean():.4f} (sd {table['ROC-AUC'].std():.4f})")
    results[f"scorecard ({len(fields)} fields)"] = table
    summarise(results).round(5).to_csv(REPORTS_DIR / "scorecard_cv.csv")
    oof_all = pd.read_csv(OOF_CSV)
    oof_all[f"scorecard ({len(fields)} fields)"] = oof
    oof_all.to_csv(OOF_CSV, index=False)

    # 4. Odds ratios from the scorecard fitted on all training rows
    sc = build_pipeline(LogisticRegression(max_iter=2000), columns=fields).fit(X, y)
    card_names = list(sc.steps[-2][1].get_feature_names_out())
    logged = set(sc.named_steps["skew"].columns_)
    card = pd.DataFrame({"weight": sc.steps[-1][1].coef_[0], "odds ratio": np.exp(sc.steps[-1][1].coef_[0])},
                        index=card_names).sort_values("odds ratio")
    card["per"] = ["blank vs filled" if n.startswith("missingindicator_") else
                   ("+1 sd of log value" if n in logged else "+1 sd") for n in card.index]
    card.round(4).to_csv(REPORTS_DIR / "scorecard_odds_ratios.csv")
    path.round(5).to_csv(REPORTS_DIR / "scorecard_l1_path.csv", index=False)

    choices = json.loads(CHOICES_JSON.read_text())
    choices.update({"scorecard_fields": fields, "scorecard_C": app_C})
    CHOICES_JSON.write_text(json.dumps(choices, indent=2))
    draw_path(path, table["ROC-AUC"].mean())
    draw_card(card)
    print(card.round(3).to_string())
    return 0


def draw_path(path, scorecard_auc):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(path["fields"], path["ROC-AUC"], color=style.BLUE, marker="o", linewidth=2)
    ax.axvline(MAX_FIELDS, color=style.ORANGE, linestyle="--", linewidth=1, label=f"form limit: {MAX_FIELDS} fields")
    ax.set_xlabel("Input fields the L1 model keeps (5-fold average)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("A handful of fields carries most of the ranking power")
    ax.legend(frameon=False, loc="lower right")
    style.clean_axes(ax)
    style.save(fig, "scorecard_l1_path")


def draw_card(card):
    fig, ax = plt.subplots(figsize=(8.5, max(3.2, 0.34 * len(card))))
    yy = np.arange(len(card))
    colors = [style.BLUE if v < 1 else style.ORANGE for v in card["odds ratio"]]
    ax.hlines(yy, 1, card["odds ratio"], color=colors, linewidth=3)
    ax.scatter(card["odds ratio"], yy, color=colors, s=40, zorder=3)
    ax.axvline(1, color=style.INK_SECONDARY, linewidth=1)
    ax.set_xscale("log")
    ticks = [t for t in [0.5, 0.67, 0.8, 1, 1.25, 1.5] if card["odds ratio"].min() * 0.85 <= t <= card["odds ratio"].max() * 1.2]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_yticks(yy, [readable(n, 1) if n.startswith("missingindicator_") else readable(n) for n in card.index])
    ax.set_xlabel("Odds ratio (log scale): below 1 = safer, above 1 = riskier")
    ax.set_title("The scorecard: what pushes the risk of default down or up")
    style.clean_axes(ax, grid_axis="x")
    style.save(fig, "scorecard_odds_ratios")


if __name__ == "__main__":
    sys.exit(main())
