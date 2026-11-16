"""Applicant segments learned with k-means, compared with the rule-based thin/thick split.

The segments are built from 13 profile fields (life stage, money, assets, product), WITHOUT
the default label, the external credit scores (already risk scores) or gender. Only afterwards
are default rates compared. Training rows only.

    python src/clusters.py
"""

from __future__ import annotations

import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

import style
from compare_models import OOF_CSV
from config import PROCESSED_DIR, REPORTS_DIR
from data import SEED, load_train_test, split_xy
from pipeline import fix_known_issues

K_RANGE = range(2, 9)
LOGGED = ["income", "credit amount", "annuity / income", "years employed"]
NAMES = {}   # filled in after fitting, from each group's profile


def build_profile(raw: pd.DataFrame) -> pd.DataFrame:
    f = fix_known_issues(raw)
    return pd.DataFrame({
        "age (years)": f["AGE_YEARS"], "years employed": f["EMPLOYED_YEARS"], "not employed": f["NOT_EMPLOYED"],
        "children": f["CNT_CHILDREN"], "family size": f["CNT_FAM_MEMBERS"], "income": f["AMT_INCOME_TOTAL"],
        "credit amount": f["AMT_CREDIT"], "annuity / income": f["ANNUITY_TO_INCOME"],
        "credit / goods": f["CREDIT_TO_GOODS"], "owns car": (f["FLAG_OWN_CAR"] == "Y").astype(int),
        "owns property": (f["FLAG_OWN_REALTY"] == "Y").astype(int),
        "revolving loan": (f["NAME_CONTRACT_TYPE"] == "Revolving loans").astype(int),
        "higher education": f["NAME_EDUCATION_TYPE"].isin(["Higher education", "Academic degree"]).astype(int),
    }, index=raw.index)


def to_matrix(profile: pd.DataFrame, stats: dict | None = None):
    """Cap, log, fill, standardise; `stats` are learned once on training rows and reused."""
    d = profile.copy()
    stats = dict(stats) if stats else {"low": d.quantile(0.001), "high": d.quantile(0.999)}
    d = d.clip(stats["low"], stats["high"], axis=1)
    d[LOGGED] = np.log1p(d[LOGGED])
    stats.setdefault("median", d.median())
    d = d.fillna(stats["median"])
    if "mean" not in stats:
        stats["mean"], stats["sd"] = d.mean(), d.std()
    return ((d - stats["mean"]) / stats["sd"]).to_numpy(), stats


def name_group(row: pd.Series) -> str:
    """A plain name from a group's profile (medians / shares)."""
    if row["revolving loan"] > 0.5:
        return "Revolving-loan customers"
    if row["not employed"] > 0.5:
        return "Pensioners"
    if row["children"] >= 1:
        return "Young families"
    return "Employees without children"


def main() -> int:
    train, _ = load_train_test()
    X, y = split_xy(train)
    profile = build_profile(X)
    Z, stats = to_matrix(profile)
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(Z), 10000, replace=False)
    scan = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
        scan.append({"k": k, "silhouette": silhouette_score(Z[sample], km.labels_[sample])})
    scan = pd.DataFrame(scan)
    usable = scan[(scan["k"] >= 3) & (scan["k"] <= 6)]          # 2 says little; > 6 is hard to act on
    K = int(usable.loc[usable["silhouette"].idxmax(), "k"])
    km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(Z)
    labels = km.labels_

    seeds_ari = [adjusted_rand_score(labels, KMeans(n_clusters=K, n_init=10, random_state=s).fit_predict(Z))
                 for s in (1, 2, 3)]
    pred = pd.read_csv(OOF_CSV)["logistic regression"].to_numpy()
    prof = profile.assign(group=labels).groupby("group")
    table = pd.DataFrame({
        "share": prof.size() / len(labels),
        "median age": prof["age (years)"].median(), "children": prof["children"].mean(),
        "not employed": prof["not employed"].mean(), "median income": prof["income"].median(),
        "median credit": prof["credit amount"].median(), "revolving loan": prof["revolving loan"].mean(),
        "default rate": pd.Series(y).groupby(labels).mean(),
        "model's average prediction": pd.Series(pred).groupby(labels).mean(),
    })
    table["name"] = [name_group(r) for _, r in table.rename(columns={"median age": "age"}).iterrows()]
    table.round(4).to_csv(REPORTS_DIR / "segments.csv")
    scan.round(4).to_csv(REPORTS_DIR / "segments_choose_k.csv", index=False)
    joblib.dump({"kmeans": km, "stats": stats, "names": table["name"].to_dict(),
                 "default_rate": table["default rate"].to_dict()}, PROCESSED_DIR / "models" / "segments.joblib")
    print(f"k = {K} (silhouette {usable['silhouette'].max():.3f}); stability across seeds: ARI {min(seeds_ari):.3f}+")
    print(table.round(3).to_string())

    t = table.sort_values("default rate")
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    yy = np.arange(len(t))
    ax.barh(yy - 0.2, t["default rate"], 0.38, color=style.BLUE, label="actual default rate")
    ax.barh(yy + 0.2, t["model's average prediction"], 0.38, color=style.GREY, label="model's average prediction")
    for i, (_, r) in enumerate(t.iterrows()):
        ax.text(r["default rate"] + 0.002, i - 0.2, f"{r['default rate']:.1%}", va="center", fontsize=9)
    ax.axvline(y.mean(), color=style.INK_SECONDARY, linestyle="--", linewidth=1, label=f"all applicants {y.mean():.1%}")
    ax.set_yticks(yy, [f"{r['name']} ({r['share']:.0%})" for _, r in t.iterrows()])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlim(0, t["default rate"].max() * 1.3)
    ax.set_title("Segments built without the default label still differ in risk")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style.clean_axes(ax, grid_axis="x")
    style.save(fig, "segments")
    return 0


if __name__ == "__main__":
    sys.exit(main())
