"""Three preparation experiments that justify src/pipeline.py, on the same 5 folds.

1. Missing values: median vs median + "was missing" flags vs iterative imputation + flags.
2. Class imbalance (8% defaults): none vs class weights vs SMOTE inside the folds, plus the
   WRONG way - SMOTE on all training rows before splitting - to show how leakage flatters.
3. The three column-dropping rules: on vs off.

Model: logistic regression (scikit-learn defaults). Training rows only; the test set is
never touched here.

    python src/prep_experiments.py
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression

import style
from config import REPORTS_DIR
from cv import evaluate_on_folds, prepare_folds, summarise
from data import SEED, load_train_test, split_xy
from pipeline import CHOSEN, build_preparation


def main() -> int:
    train, _ = load_train_test()
    X, y = split_xy(train)
    lr = LogisticRegression(max_iter=2000)
    results, extra = {}, {}

    def run(name, model, folds):
        table, _, _ = evaluate_on_folds(model, folds)
        results[name] = table
        print(f"{name:<40} ROC-AUC {table['ROC-AUC'].mean():.4f} (sd {table['ROC-AUC'].std():.4f})")

    # 1. Missing values
    for imputer in ["median", "median+flags", "iterative+flags"]:
        folds = prepare_folds(build_preparation(**{**CHOSEN, "imputer": imputer}), X, y)
        run(f"impute: {imputer}", lr, folds)
        if imputer == CHOSEN["imputer"]:
            chosen_folds = folds
        else:
            del folds

    # 2. Class imbalance, on the chosen preparation
    run("balance: none", lr, chosen_folds)
    run("balance: class weights", LogisticRegression(max_iter=2000, class_weight="balanced"), chosen_folds)
    run("balance: SMOTE inside the folds",
        ImbPipeline([("smote", SMOTE(random_state=SEED)), ("model", lr)]), chosen_folds)
    # The wrong way: oversample ALL training rows first, then cross-validate.
    prepared = build_preparation(**CHOSEN).fit_transform(X, y)
    Xs, ys = SMOTE(random_state=SEED).fit_resample(prepared, y)
    from sklearn.model_selection import cross_val_score

    from cv import CV
    wrong = cross_val_score(lr, Xs, ys, cv=CV, scoring="roc_auc", n_jobs=3)
    extra["balance: SMOTE before the split (wrong)"] = wrong
    print(f"{'balance: SMOTE before the split (wrong)':<40} ROC-AUC {wrong.mean():.4f}")

    # 3. Column rules
    del chosen_folds
    folds = prepare_folds(build_preparation(**{**CHOSEN, "reduce": False}), X, y)
    run("columns: keep all", lr, folds)

    summary = summarise(results)
    for name, aucs in extra.items():
        summary.loc[name, "ROC-AUC"] = aucs.mean()
        summary.loc[name, "ROC-AUC sd"] = aucs.std()
    summary.round(5).to_csv(REPORTS_DIR / "prep_experiments.csv")
    pd.concat(results, names=["setup", "fold"]).round(5).to_csv(REPORTS_DIR / "prep_experiments_folds.csv")
    paired = {k: results[k]["ROC-AUC"].to_numpy() - results["impute: iterative+flags"]["ROC-AUC"].to_numpy()
              for k in ["impute: median", "impute: median+flags"]}
    for k, d in paired.items():
        print(f"  {k} vs iterative+flags, per fold: {np.round(d, 4)}")
    draw(summary)
    return 0


def draw(summary: pd.DataFrame) -> None:
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1, 1.3, 1]})
    imp = summary.loc[["impute: median", "impute: median+flags", "impute: iterative+flags"]]
    bars = a1.bar(["median", "median\n+ flags", "iterative\n+ flags"], imp["ROC-AUC"], color=[style.GREY, style.BLUE, style.BLUE], width=0.6)
    a1.bar_label(bars, fmt="%.4f", padding=2, fontsize=9)
    a1.set_ylim(imp["ROC-AUC"].min() - 0.006, imp["ROC-AUC"].max() + 0.004)
    a1.set_title("Missing values: flags matter")
    names = ["balance: none", "balance: class weights", "balance: SMOTE inside the folds", "balance: SMOTE before the split (wrong)"]
    bal = summary.loc[names]
    bars = a2.bar(["none", "class\nweights", "SMOTE\ninside folds", "SMOTE before\nsplit (WRONG)"], bal["ROC-AUC"],
                  color=[style.BLUE, style.BLUE, style.BLUE, style.ORANGE], width=0.6)
    a2.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
    a2.set_ylim(0.72, bal["ROC-AUC"].max() + 0.01)
    a2.set_title("Imbalance: leakage flatters")
    honest = bal.iloc[:3]
    bars = a3.bar(["none", "class\nweights", "SMOTE"], honest["mean predicted"], color=style.BLUE, width=0.6)
    a3.bar_label(bars, fmt=lambda v: f"{v:.0%}", padding=2, fontsize=9)
    a3.axhline(0.0807, color=style.INK_SECONDARY, linestyle="--", linewidth=1, label="true rate 8.1%")
    a3.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    a3.set_ylim(0, 0.55)
    a3.set_title("...and balancing inflates risk")
    a3.legend(frameon=False, fontsize=9, loc="upper left")
    for a in (a1, a2, a3):
        a.tick_params(axis="x", labelsize=9)
        style.clean_axes(a)
    fig.tight_layout()
    style.save(fig, "prep_experiments")


if __name__ == "__main__":
    sys.exit(main())
