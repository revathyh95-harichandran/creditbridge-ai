"""One cross-validation design for every model in the project.

5-fold stratified CV on the training rows only (seed 42). Each fold refits the whole
preparation pipeline on its training part, so caps, medians and dropped columns are never
learned from the rows they're scored on.

To save time, `prepare_folds` fits the preparation once per fold and keeps the prepared
matrices; `evaluate_on_folds` then fits any number of models on top. This gives exactly the
same result as refitting the full pipeline per model.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from costs import approve_everyone_cost, best_cutoff, cost_per_1000, theoretical_cutoff
from data import SEED

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def fold_metrics(y_true, prob) -> dict:
    """Everything reported for one validation fold."""
    cut, low, _ = best_cutoff(y_true, prob)
    return {
        "ROC-AUC": roc_auc_score(y_true, prob),
        "PR-AUC": average_precision_score(y_true, prob),
        "Brier": brier_score_loss(y_true, prob),
        "mean predicted": float(np.mean(prob)),
        "cost @0.5": cost_per_1000(y_true, prob, 0.5),
        "cost @theory": cost_per_1000(y_true, prob, theoretical_cutoff()),
        "lowest cost": low,          # best cut-off on this fold (slightly optimistic)
        "best cut-off": cut,
        "approve-all cost": approve_everyone_cost(y_true),
    }


def _prepare_one(prep, X, y, train_idx, val_idx):
    fitted = clone(prep).fit(X.iloc[train_idx], y[train_idx])
    return {
        "X_train": np.asarray(fitted.transform(X.iloc[train_idx]), dtype=np.float32),
        "y_train": y[train_idx],
        "X_val": np.asarray(fitted.transform(X.iloc[val_idx]), dtype=np.float32),
        "y_val": y[val_idx],
        "val_idx": val_idx,
        "names": list(fitted[-1].get_feature_names_out()),
        "prep": fitted,
    }


def prepare_folds(prep, X, y, n_jobs: int = 3) -> list[dict]:
    """Fit the preparation on each fold's training part; keep the prepared data."""
    return Parallel(n_jobs=n_jobs)(
        delayed(_prepare_one)(prep, X, y, tr, va) for tr, va in CV.split(X, y)
    )


def _fit_one(model, fold, max_train_rows, train_scores):
    Xt, yt = fold["X_train"], fold["y_train"]
    if max_train_rows and len(yt) > max_train_rows:
        keep, _ = train_test_split(np.arange(len(yt)), train_size=max_train_rows,
                                   stratify=yt, random_state=SEED)
        Xt, yt = Xt[keep], yt[keep]
    start = time.time()
    fitted = clone(model).fit(Xt, yt)
    seconds = time.time() - start
    prob = fitted.predict_proba(fold["X_val"])[:, 1]
    row = fold_metrics(fold["y_val"], prob)
    row["fit seconds"] = seconds
    if train_scores:
        row["train ROC-AUC"] = roc_auc_score(yt, fitted.predict_proba(Xt)[:, 1])
    return fold["val_idx"], prob, row, fitted


def evaluate_on_folds(model, folds, n_jobs: int = 3, max_train_rows=None, train_scores=False):
    """Fit `model` on each prepared fold. Returns (per-fold table, out-of-fold probs, fitted models)."""
    out = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one)(model, f, max_train_rows, train_scores) for f in folds
    )
    oof = np.zeros(sum(len(f["val_idx"]) for f in folds))
    for val_idx, prob, _, _ in out:
        oof[val_idx] = prob
    return pd.DataFrame([r for _, _, r, _ in out]), oof, [m for _, _, _, m in out]


def summarise(results: dict) -> pd.DataFrame:
    """Fold means (plus the ROC-AUC spread), one row per model."""
    rows = {}
    for name, folds in results.items():
        row = folds.mean(numeric_only=True).to_dict()
        row["ROC-AUC sd"] = folds["ROC-AUC"].std()
        rows[name] = row
    cols = ["ROC-AUC", "ROC-AUC sd", "PR-AUC", "Brier", "mean predicted", "cost @0.5",
            "cost @theory", "lowest cost", "best cut-off", "approve-all cost", "fit seconds"]
    table = pd.DataFrame(rows).T
    extra = [c for c in table.columns if c not in cols]
    return table[[c for c in cols if c in table.columns] + extra]
