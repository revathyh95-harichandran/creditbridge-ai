"""Choose the cut-off, fix the final model, then open the test set exactly once.

Order matters:
1. The cut-off for every candidate is chosen on OUT-OF-FOLD training predictions (each one
   made by a model that never saw that applicant), at 5 : 1, and again at 3 : 1 and 10 : 1.
2. The final model is chosen by a rule written down in advance:
   - the small scorecard is preferred if its CV ROC-AUC is within one fold sd of the full
     logistic regression; otherwise the full logistic regression;
   - a black box (gradient boosting) replaces it only if it is >= 0.01 ROC-AUC better.
3. Only then are the 61,503 test applicants scored, once. Other candidates are scored in the
   same run for context only; none can be chosen after seeing the test.

Writes reports/decision.json, reports/test_results.csv, charts, and the fitted final model +
scorecard to data/processed/models/ (git-ignored: they were learned from Kaggle data).

    python src/evaluate.py
"""

from __future__ import annotations

import json
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

import style
from compare_models import CHOICES_JSON, OOF_CSV, gradient_boosting
from config import PROCESSED_DIR, REPORTS_DIR
from costs import (
    MISSED_DEFAULT,
    RATIO_RANGE,
    approve_everyone_cost,
    best_cutoff,
    confusion_per_1000,
    cost_per_1000,
    theoretical_cutoff,
)
from cv import CV
from data import SEED, load_train_test, split_xy
from pipeline import build_pipeline

MODELS_DIR = PROCESSED_DIR / "models"


def candidates(choices: dict) -> dict:
    sc = f"scorecard ({len(choices['scorecard_fields'])} fields)"
    return {
        "logistic regression": build_pipeline(LogisticRegression(max_iter=2000)),
        sc: build_pipeline(LogisticRegression(max_iter=2000), columns=choices["scorecard_fields"]),
        f"tree depth {choices['tree_depth']}": build_pipeline(
            DecisionTreeClassifier(max_depth=choices["tree_depth"], random_state=SEED)),
        "LDA": build_pipeline(LinearDiscriminantAnalysis()),
        "linear SVM": build_pipeline(CalibratedClassifierCV(LinearSVC(), method="sigmoid", cv=3)),
        "gradient boosting (reference)": build_pipeline(gradient_boosting()),
    }


def main() -> int:
    choices = json.loads(CHOICES_JSON.read_text())
    cands = candidates(choices)
    train, test = load_train_test()
    X, y = split_xy(train)
    oof = pd.read_csv(OOF_CSV)
    assert (oof["TARGET"].to_numpy() == y).all(), "out-of-fold file does not match the training rows"

    # CV results (mean and fold sd of ROC-AUC) for the decision rule
    cv = pd.concat([pd.read_csv(REPORTS_DIR / "model_comparison.csv", index_col=0),
                    pd.read_csv(REPORTS_DIR / "scorecard_cv.csv", index_col=0)])

    # 1. Cut-offs from out-of-fold training predictions
    cut, curves = {}, {}
    for name in cands:
        t, c, curve = best_cutoff(y, oof[name].to_numpy())
        cut[name], curves[name] = {"cut-off": t, "cost": c}, curve
    cutoffs = pd.DataFrame(cut).T

    # 2. The final model, by the rule fixed in advance
    sc = [n for n in cands if n.startswith("scorecard")][0]
    lr_auc, lr_sd = cv.loc["logistic regression", "ROC-AUC"], cv.loc["logistic regression", "ROC-AUC sd"]
    explainable = sc if cv.loc[sc, "ROC-AUC"] >= lr_auc - lr_sd else "logistic regression"
    gap = cv.loc["gradient boosting (reference)", "ROC-AUC"] - cv.loc[explainable, "ROC-AUC"]
    final = "gradient boosting (reference)" if gap >= 0.01 else explainable
    print(f"FINAL MODEL (fixed before the test): {final}; gradient boosting gap {gap:+.4f}")

    fold_of = np.zeros(len(y), dtype=int)
    for k, (_, val_idx) in enumerate(CV.split(X, y)):
        fold_of[val_idx] = k
    p = oof[final].to_numpy()
    decision = {"final_model": final, "explainable_choice": explainable, "black_box_gap": round(float(gap), 4),
                "decided_before_test": True, "cutoffs": {}, "cv_folds_with_saving": {}}
    for r in RATIO_RANGE:
        t, _, _ = best_cutoff(y, p, r)
        decision["cutoffs"][f"{r}:1"] = t
        decision["cv_folds_with_saving"][f"{r}:1"] = int(sum(
            cost_per_1000(y[fold_of == k], p[fold_of == k], t, r) < approve_everyone_cost(y[fold_of == k], r)
            for k in range(5)))
    final_cut = decision["cutoffs"][f"{MISSED_DEFAULT}:1"]

    # 3. The test set, used once
    X_test, y_test = split_xy(test)
    rows, test_prob, fitted = [], {}, {}
    for name, pipe in cands.items():
        fitted[name] = pipe.fit(X, y)
        test_prob[name] = fitted[name].predict_proba(X_test)[:, 1]
        rows.append({"model": name, "CV ROC-AUC": cv.loc[name, "ROC-AUC"],
                     "test ROC-AUC": roc_auc_score(y_test, test_prob[name]),
                     "test PR-AUC": average_precision_score(y_test, test_prob[name]),
                     "cut-off (training)": cutoffs.loc[name, "cut-off"],
                     "test cost / 1,000": cost_per_1000(y_test, test_prob[name], cutoffs.loc[name, "cut-off"])})
        print(f"{name:<32} test ROC-AUC {rows[-1]['test ROC-AUC']:.4f}")
    test_table = pd.DataFrame(rows).set_index("model")
    base = approve_everyone_cost(y_test)
    test_table["saving vs approve-all"] = 1 - test_table["test cost / 1,000"] / base
    test_table.round(5).to_csv(REPORTS_DIR / "test_results.csv")

    pf = test_prob[final]
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(1000):
        i = rng.integers(0, len(y_test), len(y_test))
        boot.append(approve_everyone_cost(y_test[i]) - cost_per_1000(y_test[i], pf[i], final_cut))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    sens = {f"{r}:1": {"cut-off": decision["cutoffs"][f"{r}:1"],
                       "model": cost_per_1000(y_test, pf, decision["cutoffs"][f"{r}:1"], r),
                       "approve everyone": approve_everyone_cost(y_test, r)} for r in RATIO_RANGE}

    g = test["CODE_GENDER"].to_numpy()
    approved = pf < final_cut
    fairness = {lab: {"applicants": int((g == c).sum()), "default rate": float(y_test[g == c].mean()),
                      "approval rate": float(approved[g == c].mean()),
                      "default rate among approved": float(y_test[(g == c) & approved].mean())}
                for c, lab in [("F", "women"), ("M", "men")]}

    cm_test = confusion_per_1000(y_test, pf, final_cut)
    decision.update({
        "test_roc_auc": round(float(roc_auc_score(y_test, pf)), 4),
        "cv_roc_auc": round(float(cv.loc[final, "ROC-AUC"]), 4),
        "test_cost_per_1000": round(cost_per_1000(y_test, pf, final_cut), 1),
        "approve_everyone_cost": round(base, 1),
        "saving_per_1000": round(base - cost_per_1000(y_test, pf, final_cut), 1),
        "saving_95_range": [round(float(lo), 1), round(float(hi), 1)],
        "test_by_ratio": {k: {kk: round(float(vv), 3) for kk, vv in v.items()} for k, v in sens.items()},
        "test_confusion_per_1000": np.round(cm_test, 1).tolist(),
        "fairness": {k: {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in fairness.items()},
        "theoretical_cutoff": round(theoretical_cutoff(), 3),
    })
    (REPORTS_DIR / "decision.json").write_text(json.dumps(decision, indent=2))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[final], MODELS_DIR / "final_model.joblib")
    joblib.dump(fitted[sc], MODELS_DIR / "scorecard.joblib")
    print(json.dumps({k: decision[k] for k in ["final_model", "cutoffs", "cv_folds_with_saving", "test_roc_auc",
                                                "test_cost_per_1000", "approve_everyone_cost", "saving_95_range"]}, indent=1))

    draw_cost_curve(curves, cutoffs, y, [n for n in cands if n in ("logistic regression", sc, "gradient boosting (reference)")
                                         or n.startswith("tree")])
    draw_confusions(y, p, final_cut)
    draw_test(cm_test, sens, final, final_cut)
    draw_cv_vs_test(test_table, final)
    draw_fairness(fairness)
    return 0


def _confusion(ax, cm, title, ratio=MISSED_DEFAULT):
    cost = np.array([[0, 1], [ratio, 0]])
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1000)
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            col = "white" if v > 500 else style.INK
            ax.text(j, i - 0.1, f"{v:.0f}", ha="center", va="center", fontsize=15, fontweight="bold", color=col)
            ax.text(j, i + 0.22, "cost 0" if cost[i, j] == 0 else f"x{cost[i, j]} = {v * cost[i, j]:.0f}",
                    ha="center", va="center", fontsize=9, color=col)
    ax.set_xticks([0, 1], ["approved", "declined"])
    ax.set_yticks([0, 1], ["repaid", "defaulted"])
    total = ratio * cm[1, 0] + cm[0, 1]
    ax.set_title(f"{title}\ncost {total:.0f} per 1,000", fontsize=11)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def draw_confusions(y, p, cut):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    _confusion(axes[0], confusion_per_1000(y, np.zeros(len(y)), 0.5), "Approve everyone")
    _confusion(axes[1], confusion_per_1000(y, p, 0.5), "Model, cut-off 0.5")
    _confusion(axes[2], confusion_per_1000(y, p, cut), f"Model, cost-based cut-off {cut}")
    fig.suptitle("Per 1,000 applicants (training, out-of-fold): accuracy points the wrong way",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    style.save(fig, "confusion_matrices")


def draw_cost_curve(curves, cutoffs, y, show):
    from costs import CUTOFF_GRID
    colors = [style.BLUE, style.GREEN, style.GREY, style.ORANGE]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for name, col in zip(show, colors):
        ax.plot(CUTOFF_GRID, curves[name], color=col, linewidth=2, label=name)
        ax.scatter([cutoffs.loc[name, "cut-off"]], [cutoffs.loc[name, "cost"]], color=col, s=40, zorder=4)
    ax.axhline(approve_everyone_cost(y), color=style.INK_SECONDARY, linestyle=":", linewidth=1, label="approve everyone")
    for x, text in [(theoretical_cutoff(), "theory 1/6"), (0.5, "default 0.5")]:
        ax.axvline(x, color=style.INK_SECONDARY, linestyle="--", linewidth=0.8)
        ax.text(x + 0.006, 312, text, fontsize=9, color=style.INK_SECONDARY)
    ax.set_xlim(0, 0.6)
    ax.set_ylim(300, 650)
    ax.set_xlabel("Cut-off: decline if predicted probability of default is at least this")
    ax.set_ylabel("Cost per 1,000 applicants (5 : 1)")
    ax.set_title("The cheapest cut-off is far below 0.5")
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=2)
    style.clean_axes(ax)
    style.save(fig, "cost_curve")


def draw_test(cm, sens, final, cut):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    _confusion(a1, cm, f"Test set: {final}, cut-off {cut}")
    xx = np.arange(len(sens))
    base = [v["approve everyone"] for v in sens.values()]
    model = [v["model"] for v in sens.values()]
    b1 = a2.bar(xx - 0.2, base, 0.38, color=style.GREY, label="approve everyone")
    b2 = a2.bar(xx + 0.2, model, 0.38, color=style.BLUE, label="final model")
    a2.bar_label(b1, fmt="%.0f", padding=2, fontsize=9)
    a2.bar_label(b2, labels=[f"{m:.0f}\n(-{1 - m / b:.0%})" for m, b in zip(model, base)], padding=2, fontsize=9)
    a2.set_xticks(xx, list(sens))
    a2.set_xlabel("Cost ratio (missed defaulter : rejected good customer)")
    a2.set_ylabel("Cost per 1,000 applicants")
    a2.set_ylim(0, max(base) * 1.3)
    a2.set_title("Test set: cost against approving everyone", fontsize=11)
    a2.legend(frameon=False, loc="upper left")
    style.clean_axes(a2)
    fig.tight_layout()
    style.save(fig, "test_result")


def draw_cv_vs_test(table, final):
    t = table.sort_values("test ROC-AUC")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    yy = np.arange(len(t))
    ax.scatter(t["CV ROC-AUC"], yy, color=style.GREY, s=50, label="cross-validation (training)", zorder=3)
    ax.scatter(t["test ROC-AUC"], yy, color=style.BLUE, s=50, marker="D", label="test set (once)", zorder=3)
    ax.set_yticks(yy, [f"{n}  <- final" if n == final else n for n in t.index])
    ax.set_xlabel("ROC-AUC")
    ax.set_title("Test scores land where cross-validation said they would")
    ax.legend(frameon=False, loc="lower right")
    style.clean_axes(ax, grid_axis="x")
    style.save(fig, "cv_vs_test")


def draw_fairness(fair):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    keys = ["default rate", "approval rate", "default rate among approved"]
    xx = np.arange(3)
    for k, (grp, col) in enumerate([("women", style.BLUE), ("men", style.ORANGE)]):
        b = ax.bar(xx + (k - 0.5) * 0.4, [fair[grp][m] for m in keys], 0.38, color=col, label=grp)
        ax.bar_label(b, fmt=lambda v: f"{v:.0%}", padding=2, fontsize=9)
    ax.set_xticks(xx, ["default rate\n(reality)", "approval rate\n(model)", "default rate\namong approved"])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylim(0, 1.15)
    ax.set_title("Fairness check on the test set (gender is never a model input)")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    style.clean_axes(ax)
    style.save(fig, "fairness")


if __name__ == "__main__":
    sys.exit(main())
