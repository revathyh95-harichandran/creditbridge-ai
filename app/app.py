"""CreditBridge decision app.

Run from the repository root, after the modelling scripts (see README):

    streamlit run app/app.py

Three tabs:
  1. Cut-off and costs  - slide the cut-off, watch the confusion matrix and cost per 1,000 update
  2. Score an applicant - the small scorecard: probability, approve/decline, and the main reasons
  3. Segments           - the k-means applicant segments and their default rates

Everything shown comes from training data (out-of-fold predictions), so nobody can tune the
cut-off on the test set; the one-time test result is shown as a fixed reference.
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import clusters  # noqa: E402
import style  # noqa: E402
from compare_models import CHOICES_JSON, OOF_CSV  # noqa: E402
from config import REPORTS_DIR  # noqa: E402
from costs import approve_everyone_cost, best_cutoff, confusion_per_1000, cost_per_1000  # noqa: E402
from data import load_train_test, split_xy  # noqa: E402
from evaluate import MODELS_DIR  # noqa: E402
from scorecard import reasons  # noqa: E402

st.set_page_config(page_title="CreditBridge decision app", layout="wide")

NEEDED = [OOF_CSV, CHOICES_JSON, REPORTS_DIR / "decision.json", MODELS_DIR / "scorecard.joblib",
          MODELS_DIR / "segments.joblib"]
missing = [p for p in NEEDED if not p.exists()]
if missing:
    st.error("Run the modelling scripts first (see the README): missing "
             + ", ".join(str(p.relative_to(ROOT)) for p in missing))
    st.stop()


@st.cache_resource(show_spinner="Loading models and training data...")
def load():
    oof = pd.read_csv(OOF_CSV)
    choices = json.loads(CHOICES_JSON.read_text())
    decision = json.loads((REPORTS_DIR / "decision.json").read_text())
    scorecard = joblib.load(MODELS_DIR / "scorecard.joblib")
    seg = joblib.load(MODELS_DIR / "segments.joblib")
    segments = pd.read_csv(REPORTS_DIR / "segments.csv", index_col=0)
    train, _ = load_train_test()
    X, _ = split_xy(train)
    # A "typical applicant": median / most common value of every column. The form only
    # changes the fields it asks about.
    template = {c: (float(X[c].median()) if pd.api.types.is_numeric_dtype(X[c]) else X[c].mode().iloc[0])
                for c in X.columns}
    return oof, choices, decision, scorecard, seg, segments, template


oof, choices, decision, scorecard, seg, segments, template = load()
final = decision["final_model"]
y_all, p_all = oof["TARGET"].to_numpy(), oof[final].to_numpy()
sc_name = f"scorecard ({len(choices['scorecard_fields'])} fields)"
SC_CUTOFF, _, _ = best_cutoff(y_all, oof[sc_name].to_numpy())

st.title("CreditBridge: should we approve this loan?")
st.caption("Home Credit Default Risk data · training rows only · costs per 1,000 applicants")
tab1, tab2, tab3 = st.tabs(["1 · Cut-off and costs", "2 · Score an applicant", "3 · Segments"])

# --- Tab 1 ---------------------------------------------------------------------------------
with tab1:
    st.subheader("Where should the lender draw the line?")
    st.write(f"Each applicant gets a probability of default from the final model ({final}). "
             "Applicants **at or above the cut-off are declined**. The numbers are out-of-fold "
             "predictions for the 246,008 training applicants.")
    c1, c2 = st.columns(2)
    with c1:
        ratio = st.radio("Cost of a missed defaulter vs a wrongly rejected good customer",
                         [3, 5, 10], index=1, horizontal=True, format_func=lambda r: f"{r} : 1")
    with c2:
        cutoff = st.slider("Cut-off (decline if probability ≥ cut-off)", 0.01, 0.60,
                           float(decision["cutoffs"][f"{ratio}:1"]), 0.01)
    cm = confusion_per_1000(y_all, p_all, cutoff)
    cost = ratio * cm[1, 0] + cm[0, 1]
    base = approve_everyone_cost(y_all, ratio)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cost per 1,000 applicants", f"{cost:.0f}", f"{cost - base:+.0f} vs approve everyone",
              delta_color="inverse")
    m2.metric("Approved", f"{(cm[0, 0] + cm[1, 0]) / 10:.1f}%")
    m3.metric("Defaulters caught", f"{cm[1, 1] / cm[1].sum():.1%}")
    m4.metric("Accuracy", f"{(cm[0, 0] + cm[1, 1]) / 1000:.1%}")
    left, right = st.columns([1, 1.4])
    with left:
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1000)
        weights = np.array([[0, 1], [ratio, 0]])
        for i in range(2):
            for j in range(2):
                v, col = cm[i, j], ("white" if cm[i, j] > 500 else style.INK)
                ax.text(j, i - 0.1, f"{v:.0f}", ha="center", va="center", fontsize=16, fontweight="bold", color=col)
                ax.text(j, i + 0.22, "cost 0" if weights[i, j] == 0 else f"×{weights[i, j]} = {v * weights[i, j]:.0f}",
                        ha="center", va="center", fontsize=9, color=col)
        ax.set_xticks([0, 1], ["approved", "declined"])
        ax.set_yticks([0, 1], ["repaid", "defaulted"])
        ax.set_title("Per 1,000 applicants", fontsize=11)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        st.pyplot(fig)
    with right:
        grid = np.round(np.arange(0.01, 0.601, 0.01), 2)
        curve = [cost_per_1000(y_all, p_all, t, ratio) for t in grid]
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ax.plot(grid, curve, color=style.BLUE, linewidth=2)
        ax.axhline(base, color=style.INK_SECONDARY, linestyle=":", linewidth=1, label="approve everyone")
        ax.axvline(cutoff, color=style.ORANGE, linewidth=1.5, label=f"your cut-off {cutoff:.2f}")
        ax.scatter([cutoff], [cost], color=style.ORANGE, zorder=4)
        ax.set_xlabel("Cut-off")
        ax.set_ylabel("Cost per 1,000 applicants")
        ax.set_ylim(min(curve) * 0.9, max(base * 1.6, cost * 1.05))
        ax.set_title("Cost for every possible cut-off", fontsize=11)
        ax.legend(frameon=False, fontsize=9)
        style.clean_axes(ax)
        st.pyplot(fig)
    st.info(f"Untouched test set (scored once): cut-off {decision['cutoffs']['5:1']} cost "
            f"{decision['test_cost_per_1000']:.0f} per 1,000 vs {decision['approve_everyone_cost']:.0f} for "
            f"approving everyone (ROC-AUC {decision['test_roc_auc']:.3f}).")

# --- Tab 2 ---------------------------------------------------------------------------------
with tab2:
    st.subheader(f"Score one applicant with the {len(choices['scorecard_fields'])}-field scorecard")
    st.write("A small logistic regression on the fields a loan officer can type in. "
             f"It declines at a probability of **{SC_CUTOFF:.2f}** or more (5 : 1 costs). "
             "The profile fields only place the applicant in a segment.")
    with st.form("applicant"):
        a, b, c = st.columns(3)
        with a:
            st.markdown("**External credit scores** (0 = worst, 1 = best)")
            ext = {}
            for k, default in [(1, 0.50), (2, 0.57), (3, 0.53)]:
                known = st.checkbox(f"Score {k} available", value=True, key=f"has{k}")
                ext[k] = st.slider(f"External score {k}", 0.0, 1.0, default, 0.01, key=f"ext{k}") if known else np.nan
        with b:
            st.markdown("**Loan and work**")
            credit = st.number_input("Credit amount", 10_000, 4_000_000, 520_000, 10_000)
            goods = st.number_input("Price of the goods", 10_000, 4_000_000, 450_000, 10_000)
            employed = st.checkbox("Currently employed", value=True)
            years = st.slider("Years employed", 0.0, 45.0, 4.5, 0.5) if employed else None
            revolving = st.checkbox("Revolving loan (credit-card style)")
        with c:
            st.markdown("**Profile** (for the segment only)")
            age = st.slider("Age", 21, 69, 43)
            children = st.number_input("Children", 0, 10, 0)
            income = st.number_input("Yearly income", 25_000, 2_000_000, 147_600, 5_000)
            annuity = st.number_input("Yearly repayment (annuity)", 2_000, 300_000, 24_900, 500)
            car = st.checkbox("Owns a car")
            realty = st.checkbox("Owns property", value=True)
            higher = st.checkbox("Higher education")
        st.form_submit_button("Score this applicant", type="primary")

    row = dict(template)
    row.update({
        "EXT_SOURCE_1": ext[1], "EXT_SOURCE_2": ext[2], "EXT_SOURCE_3": ext[3],
        "AMT_CREDIT": credit, "AMT_GOODS_PRICE": goods,
        "DAYS_EMPLOYED": -years * 365.25 if employed else 365243,
        "NAME_CONTRACT_TYPE": "Revolving loans" if revolving else "Cash loans",
        "DAYS_BIRTH": -age * 365.25, "CNT_CHILDREN": children, "CNT_FAM_MEMBERS": children + 1,
        "AMT_INCOME_TOTAL": income, "AMT_ANNUITY": annuity,
        "FLAG_OWN_CAR": "Y" if car else "N", "FLAG_OWN_REALTY": "Y" if realty else "N",
        "NAME_EDUCATION_TYPE": "Higher education" if higher else "Secondary / secondary special",
    })
    applicant = pd.DataFrame([row])
    prob = float(scorecard.predict_proba(applicant)[0, 1])
    r1, r2 = st.columns([1, 1.6])
    with r1:
        st.metric("Predicted probability of default", f"{prob:.1%}",
                  f"{prob - y_all.mean():+.1%} vs average applicant", delta_color="inverse")
        if prob >= SC_CUTOFF:
            st.error(f"**Decline**: probability ≥ cut-off {SC_CUTOFF:.2f}")
        else:
            st.success(f"**Approve**: probability below cut-off {SC_CUTOFF:.2f}")
    with r2:
        push = reasons(scorecard, applicant)
        push = push[push.abs() > 0.01]
        fig, ax = plt.subplots(figsize=(6.4, 0.42 * max(len(push), 3) + 0.8))
        ax.barh(push.index, push.values, color=[style.BLUE if v < 0 else style.ORANGE for v in push.values], height=0.6)
        ax.axvline(0, color=style.INK_SECONDARY, linewidth=1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_xlabel("← lowers the risk      raises the risk →   (log-odds vs average applicant)")
        ax.set_title("Main reasons for this score", fontsize=11)
        style.clean_axes(ax, grid_axis="x")
        st.pyplot(fig)
    Z, _ = clusters.to_matrix(clusters.build_profile(applicant), seg["stats"])
    group = int(seg["kmeans"].predict(Z)[0])
    st.info(f"Segment (tab 3): **{seg['names'][group]}** — {seg['default_rate'][group]:.1%} of this "
            "segment defaulted in the training data.")

# --- Tab 3 ---------------------------------------------------------------------------------
with tab3:
    st.subheader("Applicant segments, found without the default label")
    st.write("k-means on 13 profile fields (age, employment, family, income, loan, assets, product). "
             "The default label, the external scores and gender are not used to build the groups.")
    show = pd.DataFrame({
        "segment": segments["name"],
        "share of applicants": segments["share"].map("{:.0%}".format),
        "median age": segments["median age"].round().astype(int),
        "children (avg)": segments["children"].round(1),
        "not employed": segments["not employed"].map("{:.0%}".format),
        "median income": segments["median income"].map("{:,.0f}".format),
        "median credit": segments["median credit"].map("{:,.0f}".format),
        "revolving loans": segments["revolving loan"].map("{:.0%}".format),
        "default rate": segments["default rate"].map("{:.1%}".format),
        "model's average prediction": segments["model's average prediction"].map("{:.1%}".format),
    })
    st.dataframe(show, hide_index=True, width="stretch")
    t = segments.sort_values("default rate")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bars = ax.barh(t["name"], t["default rate"], color=style.BLUE, height=0.55)
    bars[list(t.index).index(group)].set_color(style.ORANGE)
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in t["default rate"]], padding=3, fontsize=9)
    ax.axvline(y_all.mean(), color=style.INK_SECONDARY, linestyle="--", linewidth=1, label=f"all {y_all.mean():.1%}")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlim(0, t["default rate"].max() * 1.25)
    ax.set_title("Default rate by segment (orange = the applicant from tab 2)", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style.clean_axes(ax, grid_axis="x")
    st.pyplot(fig)
    st.caption("Segments help with products and monitoring; approve/decline stays with the scorecard.")
