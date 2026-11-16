"""Fair model comparison on the full population: same preparation, same 5 folds, same measures.

Models: logistic regression, a decision tree (depth chosen by cross-validation), LDA, a linear
SVM (Platt-scaled to give probabilities), and gradient boosting with the settings used for the
original thin-file model, as a "black box" reference.

Tie rule used throughout: if two options differ by less than one fold standard deviation of
ROC-AUC, the simpler one wins. A black box must beat the explainable model by >= 0.01 AUC.

Writes reports/model_comparison.csv (+ per fold), reports/model_choices.json,
data/processed/oof_predictions.csv (git-ignored) and two charts.

    python src/compare_models.py
"""

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

import style
from config import PROCESSED_DIR, REPORTS_DIR
from cv import evaluate_on_folds, prepare_folds, summarise
from data import SEED, load_train_test, split_xy
from pipeline import build_preparation

DEPTHS = [2, 3, 4, 5, 6, 8, 10, 12, 15]
OOF_CSV = PROCESSED_DIR / "oof_predictions.csv"
CHOICES_JSON = REPORTS_DIR / "model_choices.json"


def gradient_boosting():
    """Same settings as the original thin-file model (src/model_thin_file.py)."""
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
                                          min_samples_leaf=100, l2_regularization=1.0, random_state=SEED)


def main() -> int:
    train, _ = load_train_test()
    X, y = split_xy(train)
    folds = prepare_folds(build_preparation(), X, y)
    print(f"Prepared 5 folds; models see {len(folds[0]['names'])} columns")
    results, oof = {}, {}

    def run(name, model, **kw):
        table, prob, _ = evaluate_on_folds(model, folds, **kw)
        results[name], oof[name] = table, prob
        print(f"{name:<34} ROC-AUC {table['ROC-AUC'].mean():.4f} (sd {table['ROC-AUC'].std():.4f})  "
              f"lowest cost {table['lowest cost'].mean():.1f}")

    run("logistic regression", LogisticRegression(max_iter=2000))
    for d in DEPTHS:
        run(f"tree depth {d}", DecisionTreeClassifier(max_depth=d, random_state=SEED), train_scores=True)
    tree = pd.DataFrame({"depth": DEPTHS,
                         "train ROC-AUC": [results[f"tree depth {d}"]["train ROC-AUC"].mean() for d in DEPTHS],
                         "validation ROC-AUC": [results[f"tree depth {d}"]["ROC-AUC"].mean() for d in DEPTHS]})
    top = tree["validation ROC-AUC"].idxmax()
    tie_line = tree.loc[top, "validation ROC-AUC"] - results[f"tree depth {int(tree.loc[top, 'depth'])}"]["ROC-AUC"].std()
    depth = int(tree[tree["validation ROC-AUC"] >= tie_line]["depth"].min())   # shallowest within the tie
    print(f"Tree depth chosen: {depth}")
    run("LDA", LinearDiscriminantAnalysis())
    run("linear SVM", CalibratedClassifierCV(LinearSVC(), method="sigmoid", cv=3))
    run("gradient boosting (reference)", gradient_boosting())

    main_models = ["logistic regression", f"tree depth {depth}", "LDA", "linear SVM", "gradient boosting (reference)"]
    summary = summarise(results)
    summary.round(5).to_csv(REPORTS_DIR / "model_comparison.csv")
    pd.concat(results, names=["model", "fold"]).round(5).to_csv(REPORTS_DIR / "model_comparison_folds.csv")
    tree.round(5).to_csv(REPORTS_DIR / "tree_depth.csv", index=False)
    pd.DataFrame({**{m: oof[m] for m in main_models}, "TARGET": y}).to_csv(OOF_CSV, index=False)
    CHOICES_JSON.write_text(json.dumps({"tree_depth": depth}, indent=2))

    draw_folds(results, main_models)
    draw_tree(tree, depth)
    print(summary.loc[main_models, ["ROC-AUC", "ROC-AUC sd", "PR-AUC", "lowest cost", "best cut-off"]].round(4).to_string())
    return 0


def draw_folds(results, models):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    order = sorted(models, key=lambda m: results[m]["ROC-AUC"].mean())
    for i, name in enumerate(order):
        f = results[name]["ROC-AUC"]
        col = style.ORANGE if "reference" in name else style.BLUE
        ax.plot([f.min(), f.max()], [i, i], color=style.GRID, linewidth=2, zorder=2)
        ax.scatter(f, [i] * len(f), color=col, s=35, alpha=0.85, zorder=3)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("ROC-AUC on each of the 5 validation folds")
    ax.set_title("Same folds, same preparation: one dot per fold")
    style.clean_axes(ax, grid_axis="x")
    style.save(fig, "model_comparison")


def draw_tree(tree, depth):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(tree["depth"], tree["train ROC-AUC"], color=style.GREY, marker="o", linewidth=2, label="training folds")
    ax.plot(tree["depth"], tree["validation ROC-AUC"], color=style.BLUE, marker="o", linewidth=2, label="validation folds")
    ax.axvline(depth, color=style.INK_SECONDARY, linestyle=":", linewidth=1)
    ax.text(depth + 0.2, tree["validation ROC-AUC"].min(), f"chosen: depth {depth}", fontsize=9, color=style.INK_SECONDARY)
    ax.set_xlabel("Tree depth")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Deeper trees memorise the training data")
    ax.legend(frameon=False, loc="upper left")
    style.clean_axes(ax)
    style.save(fig, "tree_depth")


if __name__ == "__main__":
    sys.exit(main())
