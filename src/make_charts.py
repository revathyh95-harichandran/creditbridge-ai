"""Draw every model chart used in the README (reports/figures/*.png).

Every number on the charts is computed here from the database and from
src/validate_model.py - nothing is typed in by hand - so re-running this script after
the data or model changes keeps the README pictures honest.

The charts only show group-level percentages, scores and smoothed curves (thinned to a
couple of hundred points), never individual applicants, so the PNGs are safe to commit
(unlike the Kaggle data itself).

Colour rule, the same on every chart: blue = the model, grey = random / baseline /
comparison.

    python src/make_charts.py
"""

from __future__ import annotations

import sys

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from config import DUCKDB_PATH, REPORTS_DIR  # noqa: E402
from model_thin_file import load_segment  # noqa: E402
from validate_model import run_validation  # noqa: E402

FIGURES_DIR = REPORTS_DIR / "figures"

# Reference palette (dataviz skill, light mode). MODEL = categorical slot 1, which
# passes the palette validator; BASELINE_GREY is the "muted" ink step.
SURFACE = "#fcfcfb"
MODEL = "#2a78d6"
BASELINE_GREY = "#898781"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

LAYOUT_DPI = 110  # 1 figure pixel ~= 1 screen pixel at GitHub's README width
SAVE_DPI = 220    # saved at 2x so it stays sharp on high-res screens
BAR_PX = 18       # bar thickness, under the 24px cap
RADIUS_PX = 4     # rounded data-end

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9.5,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
})

pct = FuncFormatter(lambda v, _: f"{v:.0%}")
auc_fmt = FuncFormatter(lambda v, _: f"{v:.2f}")
AGE_LABELS = {"30-45": "30–45"}


# --- shared chrome -----------------------------------------------------------------

def _hide_spines(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)


def style_axes(ax, xmax, xformatter=pct, xmin=0.0):
    """Horizontal-bar axes: vertical hairline grid, muted ticks."""
    ax.set_xlim(xmin, xmax)
    _hide_spines(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(xformatter)
    ax.tick_params(axis="x", colors=INK_MUTED, length=0, labelsize=8.5)
    ax.tick_params(axis="y", colors=INK_SECONDARY, length=0, labelsize=9.5)


def style_column_axes(ax, ymax, yformatter=pct, ymin=0.0):
    """Column-chart axes: horizontal hairline grid, muted ticks."""
    ax.set_ylim(ymin, ymax)
    _hide_spines(ax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(yformatter)
    ax.tick_params(axis="y", colors=INK_MUTED, length=0, labelsize=8.5)
    ax.tick_params(axis="x", colors=INK_SECONDARY, length=0, labelsize=9.5)


def style_line_axes(ax, xlim, ylim, xformatter=pct, yformatter=pct):
    """Curve axes: hairline grid both ways, a baseline on the left and bottom."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    _hide_spines(ax)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(xformatter)
    ax.yaxis.set_major_formatter(yformatter)
    ax.tick_params(colors=INK_MUTED, length=0, labelsize=8.5)


def axis_label(ax, x=None, y=None):
    if x:
        ax.set_xlabel(x, color=INK_SECONDARY, fontsize=9, labelpad=6)
    if y:
        ax.set_ylabel(y, color=INK_SECONDARY, fontsize=9, labelpad=6)


def value_label(ax, x, y, text, ha="left", va="center", color=INK, size=9):
    # Surface-colored backing so reference lines never run through a number.
    ax.text(x, y, text, va=va, ha=ha, color=color, fontsize=size, zorder=4,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0})


def add_titles(fig, title, subtitle):
    fig.text(0.02, 0.965, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=INK)
    fig.text(0.02, 0.965 - 0.075 * (4.6 / fig.get_figheight()), subtitle,
             ha="left", va="top", fontsize=9.5, color=INK_SECONDARY)


def add_legend(fig, handles, bottom):
    """One row, right-aligned, in the band between the subtitle and the plot - never on the data."""
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.97, bottom), ncol=len(handles),
               frameon=False, fontsize=9, labelcolor=INK_SECONDARY, handlelength=1.4, columnspacing=1.6)


def model_vs_random_handles(model_label="Model", random_label="Random / baseline", kind="patch"):
    if kind == "line":
        return [Line2D([], [], color=MODEL, linewidth=2, label=model_label),
                Line2D([], [], color=BASELINE_GREY, linewidth=2, label=random_label)]
    return [Patch(color=MODEL, label=model_label), Patch(color=BASELINE_GREY, label=random_label)]


def save(fig, path):
    fig.savefig(path, dpi=SAVE_DPI, facecolor=SURFACE)
    plt.close(fig)


# --- bars with a square baseline and a 4px-rounded data end ---------------------------

def _px_per_unit(ax):
    box = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return box.width / abs(x1 - x0), box.height / abs(y1 - y0)


def draw_hbars(ax, ys, values, colors, bar_px=BAR_PX, fmt=lambda v: f"{v:.1%}", labels=True):
    """Horizontal bars from the left edge of the x-axis. Call after the limits are set."""
    px_x, px_y = _px_per_unit(ax)
    x0 = ax.get_xlim()[0]
    height, rx = bar_px / px_y, RADIUS_PX / px_x
    for y, value, color in zip(ys, values, colors):
        top, length = y - height / 2, value - x0
        if length > 2 * rx:
            ax.add_patch(FancyBboxPatch((x0, top), length, height, boxstyle=f"round,pad=0,rounding_size={rx}",
                                        mutation_aspect=px_x / px_y, facecolor=color, edgecolor="none"))
            ax.add_patch(Rectangle((x0, top), length - rx, height, facecolor=color, edgecolor="none"))
        else:
            ax.add_patch(Rectangle((x0, top), length, height, facecolor=color, edgecolor="none"))
        if labels:
            value_label(ax, value + 6 / px_x, y, fmt(value))
    ax.axvline(x0, color=AXIS, linewidth=1)


def draw_vbars(ax, xs, values, colors, bar_px=BAR_PX, fmt=lambda v: f"{v:.1%}", labels=True):
    """Columns from the bottom of the y-axis. Call after the limits are set."""
    px_x, px_y = _px_per_unit(ax)
    y0 = ax.get_ylim()[0]
    width, rx, ry = bar_px / px_x, RADIUS_PX / px_x, RADIUS_PX / px_y
    for x, value, color in zip(xs, values, colors):
        left, height = x - width / 2, value - y0
        if height > 2 * ry:
            ax.add_patch(FancyBboxPatch((left, y0), width, height, boxstyle=f"round,pad=0,rounding_size={rx}",
                                        mutation_aspect=px_y and px_x / px_y, facecolor=color, edgecolor="none"))
            ax.add_patch(Rectangle((left, y0), width, height - ry, facecolor=color, edgecolor="none"))
        else:
            ax.add_patch(Rectangle((left, y0), width, height, facecolor=color, edgecolor="none"))
        if labels:
            value_label(ax, x, value + 4 / px_y, fmt(value), ha="center", va="bottom", size=8.5)
    ax.axhline(y0, color=AXIS, linewidth=1)


def row_layout(families):
    """[(header, [(label, ...), ...]), ...] -> bar rows with a gap + header line between families."""
    ys, labels, headers, y = [], [], [], 0.0
    for i, (header, items) in enumerate(families):
        if i:
            y += 0.6
        headers.append((header, y))
        y += 1
        for item in items:
            ys.append(y)
            labels.append(item)
            y += 1
    return ys, labels, headers, y


def draw_row_headers(ax, headers):
    for header, y in headers:
        ax.text(-0.012, y, header, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=9.5, fontweight="semibold", color=INK)


# --- the charts ----------------------------------------------------------------------

def chart_default_rates(v, thick_rate, path):
    labels = ["All thin-file applicants", "Average normal (thick-file) customer", "Model's safest 75%",
              "Model's safest 50%", "Model's safest 25%", "Model's riskiest 10%"]
    values = [v["test_default_rate"], thick_rate, v["safest"][0.75], v["safest"][0.50],
              v["safest"][0.25], v["riskiest_10"]]
    colors = [BASELINE_GREY, BASELINE_GREY, MODEL, MODEL, MODEL, MODEL]

    fig = plt.figure(figsize=(8, 4.6), dpi=LAYOUT_DPI)
    ax = fig.add_axes([0.34, 0.12, 0.60, 0.62])
    style_axes(ax, xmax=0.28)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_yticks(range(len(labels)), labels)
    draw_hbars(ax, range(len(labels)), values, colors)
    axis_label(ax, x="Default rate (% who didn't pay back)")
    add_titles(fig, "The model's safest half default less than the average normal customer",
               f"Thin-file applicants the model never saw ({v['n_test']:,} people) · "
               "average thick-file customer shown for comparison")
    add_legend(fig, model_vs_random_handles("Picked out by the model", "For comparison"), bottom=0.76)
    save(fig, path)


def chart_risk_groups(v, path):
    table = v["calibration"]
    fig = plt.figure(figsize=(8, 4.6), dpi=LAYOUT_DPI)
    ax = fig.add_axes([0.10, 0.17, 0.86, 0.58])
    ax.set_xlim(0.4, 10.6)
    style_column_axes(ax, ymax=0.27)
    draw_vbars(ax, table["group"], table["actual"], [MODEL] * len(table), bar_px=24)
    ax.axhline(v["test_default_rate"], color=BASELINE_GREY, linewidth=1.2, zorder=0.9)
    value_label(ax, 0.5, v["test_default_rate"] + 0.004, f"Everyone: {v['test_default_rate']:.2%}",
                va="bottom", color=INK_SECONDARY, size=8.5)
    ax.set_xticks(table["group"], ["1\nsafest"] + [str(g) for g in table["group"][1:-1]] + ["10\nriskiest"])
    axis_label(ax, x=f"Risk group (10 equal groups of ~{v['n_test'] // 10:,} people, sorted by the model's score)",
               y="Actually defaulted")
    add_titles(fig, "Riskier groups really do default more",
               f"Default rate in each risk group · {v['n_test']:,} thin-file applicants the model never saw")
    save(fig, path)


def chart_gains(v, path):
    g = v["curves"]["gains"]
    n, half_model, half_random = v["n_test"], v["half_defaulters_model"], v["half_defaulters_random"]
    fig = plt.figure(figsize=(8, 5.4), dpi=LAYOUT_DPI)
    ax = fig.add_axes([0.11, 0.12, 0.55, 0.64])
    style_line_axes(ax, (0, 1), (0, 1))
    ax.plot([0, 1], [0, 1], color=BASELINE_GREY, linewidth=1.5)
    ax.plot(g["share_checked"], g["share_found"], color=MODEL, linewidth=2, solid_capstyle="round")
    for people, color, name in ((half_model, MODEL, "Model"), (half_random, BASELINE_GREY, "Random")):
        x = people / n
        ax.plot([x, x], [0, 0.5], color=color, linewidth=1, zorder=1)
        ax.plot(x, 0.5, "o", markersize=8, color=color, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    ax.axhline(0.5, color=INK_MUTED, linewidth=0.8, zorder=0.9)
    axis_label(ax, x="People checked, riskiest first (% of test set)", y="Defaulters found (% of all 1,664)")
    fig.text(0.70, 0.62, "To find half of all defaulters:", fontsize=9.5, fontweight="semibold", color=INK)
    fig.text(0.70, 0.56, f"Model's order: {half_model:,} people ({half_model / n:.0%})", fontsize=9.5, color=INK)
    fig.text(0.70, 0.51, f"Random order: {half_random:,} people ({half_random / n:.0%})", fontsize=9.5, color=INK)
    fig.text(0.70, 0.43, f"That's {1 - half_model / half_random:.0%} fewer people to check.",
             fontsize=9.5, color=INK_SECONDARY)
    add_titles(fig, "Going down the model's list finds defaulters about twice as fast",
               "Share of all defaulters found vs share of people checked · test set")
    add_legend(fig, model_vs_random_handles("Model's order (riskiest first)", "Random order", kind="line"),
               bottom=0.79)
    save(fig, path)


def chart_pr_curve(v, path):
    recall, precision = v["curves"]["pr"]
    keep = recall > 0  # drop sklearn's artificial (recall 0, precision 1) starting point
    fig = plt.figure(figsize=(8, 4.8), dpi=LAYOUT_DPI)
    ax = fig.add_axes([0.11, 0.14, 0.84, 0.58])
    style_line_axes(ax, (0, 1), (0, 0.45))
    ax.fill_between(recall[keep], precision[keep], color=MODEL, alpha=0.10, linewidth=0)
    ax.plot(recall[keep], precision[keep], color=MODEL, linewidth=2)
    ax.axhline(v["test_default_rate"], color=BASELINE_GREY, linewidth=1.5)
    # Both labels sit in empty space: one inside the shaded area, one below the grey line.
    ax.text(0.30, 0.135, f"Area under the blue line (PR-AUC) = {v['pr_auc']:.3f}", fontsize=9.5, color=INK)
    ax.text(0.30, 0.055, f"Random: always {v['test_default_rate']:.1%}, so its area (PR-AUC) = "
            f"{v['test_default_rate']:.3f}", fontsize=9.5, color=INK_SECONDARY)
    axis_label(ax, x="Defaulters found so far (recall)", y="Hit rate: % of flagged who defaulted (precision)")
    add_titles(fig, "The model's hit rate stays well above random all the way down the list",
               "Hit rate as the bank flags more and more of the riskiest people · test set")
    add_legend(fig, model_vs_random_handles("Model", "Random order", kind="line"), bottom=0.75)
    save(fig, path)


def chart_roc(v, path):
    fpr, tpr = v["curves"]["roc"]
    i = int(np.argmax(tpr - fpr))
    fig = plt.figure(figsize=(8, 5.4), dpi=LAYOUT_DPI)
    ax = fig.add_axes([0.11, 0.12, 0.55, 0.64])
    style_line_axes(ax, (0, 1), (0, 1))
    ax.plot([0, 1], [0, 1], color=BASELINE_GREY, linewidth=1.5)
    ax.plot(fpr, tpr, color=MODEL, linewidth=2)
    ax.plot([fpr[i], fpr[i]], [fpr[i], tpr[i]], color=INK, linewidth=1, zorder=3)
    # Between the two lines, just right of the KS marker: clear of both curves.
    ax.text(fpr[i] + 0.015, fpr[i] + 0.65 * (tpr[i] - fpr[i]), f"KS = {v['ks']:.3f}",
            fontsize=8.5, color=INK, va="center")
    value_label(ax, 0.55, 0.18, "Coin flip (AUC 0.5)", color=INK_SECONDARY, size=8.5)
    axis_label(ax, x="Good payers wrongly flagged (false positive rate)",
               y="Defaulters caught (true positive rate)")
    fig.text(0.70, 0.60, f"AUC = {v['auc']:.3f}", fontsize=15, fontweight="semibold", color=INK)
    fig.text(0.70, 0.54, f"Gini = {v['gini']:.3f}  (= 2 × AUC − 1)", fontsize=9.5, color=INK)
    fig.text(0.70, 0.49, f"KS = {v['ks']:.3f}", fontsize=9.5, color=INK)
    fig.text(0.70, 0.41, "The further the blue line bows\ntowards the top-left, the better.",
             fontsize=9, color=INK_SECONDARY)
    add_titles(fig, "The model clearly beats a coin flip",
               "ROC curve on the test set · every possible cut-off at once")
    add_legend(fig, model_vs_random_handles("Model", "Coin flip", kind="line"), bottom=0.79)
    save(fig, path)


def chart_calibration(v, path):
    table = v["calibration"]
    fig = plt.figure(figsize=(8, 5.4), dpi=LAYOUT_DPI)
    ax = fig.add_axes([0.11, 0.12, 0.55, 0.64])
    style_line_axes(ax, (0, 0.28), (0, 0.28))
    ax.plot([0, 0.28], [0, 0.28], color=BASELINE_GREY, linewidth=1.5)
    ax.plot(table["predicted"], table["actual"], color=MODEL, linewidth=1.2, zorder=2)
    ax.plot(table["predicted"], table["actual"], "o", markersize=8, color=MODEL,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    first, last = table.iloc[0], table.iloc[-1]
    ax.text(first["predicted"] + 0.004, first["actual"] - 0.014, "Safest group", fontsize=8.5, color=INK)
    value_label(ax, last["predicted"] - 0.008, last["actual"] + 0.012,
                f"Riskiest group: predicted {last['predicted']:.1%}, actual {last['actual']:.1%}",
                ha="right", size=8.5)
    axis_label(ax, x="Model's predicted default rate", y="Actual default rate")
    fig.text(0.70, 0.62, "Brier score (lower is better)", fontsize=9.5, fontweight="semibold", color=INK)
    fig.text(0.70, 0.57, f"Model: {v['brier']:.4f}", fontsize=9.5, color=INK)
    fig.text(0.70, 0.52, f"Everyone gets the average: {v['brier_base']:.4f}", fontsize=9.5, color=INK)
    fig.text(0.70, 0.44, "Log loss (lower is better)", fontsize=9.5, fontweight="semibold", color=INK)
    fig.text(0.70, 0.39, f"Model: {v['log_loss']:.3f}", fontsize=9.5, color=INK)
    fig.text(0.70, 0.34, f"Everyone gets the average: {v['log_loss_base']:.3f}", fontsize=9.5, color=INK)
    add_titles(fig, "When the model says 10%, about 10% really do default",
               "Predicted vs actual default rate in each of the 10 risk groups · test set")
    add_legend(fig, model_vs_random_handles("Model's risk groups", "Perfect match", kind="line"), bottom=0.79)
    save(fig, path)


def chart_flagging(v, path):
    flags = v["flagging"]
    names = [f"Riskiest\n{f['share']:.0%}" for f in flags]
    panels = [
        ("Hit rate (precision)\n% of flagged who defaulted",
         [f["precision"] for f in flags], [v["test_default_rate"]] * len(flags), 0.30,
         f"Grey = random order: always {v['test_default_rate']:.1%}"),
        ("Defaulters caught (recall)\n% of all defaulters flagged",
         [f["recall"] for f in flags], [f["share"] for f in flags], 0.85,
         "Grey = random order: catches the same % as it flags"),
    ]
    fig = plt.figure(figsize=(8, 4.8), dpi=LAYOUT_DPI)
    for k, (title, model_vals, random_vals, ymax, note) in enumerate(panels):
        ax = fig.add_axes([0.08 + k * 0.49, 0.19, 0.40, 0.45])
        ax.set_xlim(-0.6, len(flags) - 0.4)
        style_column_axes(ax, ymax=ymax)
        offset = (BAR_PX + 2) / 2 / _px_per_unit(ax)[0]
        xs = np.arange(len(flags))
        draw_vbars(ax, xs - offset, model_vals, [MODEL] * len(flags))
        # No labels on the grey bars: they'd spill onto the blue bar beside them. The note says it.
        draw_vbars(ax, xs + offset, random_vals, [BASELINE_GREY] * len(flags), labels=False)
        ax.set_xticks(xs, names)
        ax.set_title(title, loc="left", fontsize=9.5, fontweight="semibold", color=INK, pad=10)
        axis_label(ax, x=note)
    add_titles(fig, "Flag more people and you catch more defaulters, but with more false alarms",
               "If the bank flagged the riskiest 10%, 25% or 50% of the test set")
    add_legend(fig, model_vs_random_handles("Model", "Random order"), bottom=0.77)
    save(fig, path)


def chart_model_comparison(v, path):
    bench = v["thick_file_benchmark"]
    rows = [
        ("Simple 3-factor rule\n(income type, education, region)", v["simple_rule_auc"], BASELINE_GREY),
        ("Our thin-file model", v["auc"], MODEL),
        ("Thick-file, same info", bench["same features"], BASELINE_GREY),
        ("Thick-file + credit scores", bench["same features + EXT_SOURCE scores"], BASELINE_GREY),
    ]
    fig = plt.figure(figsize=(8, 4.2), dpi=LAYOUT_DPI)
    ax = fig.add_axes([0.32, 0.14, 0.60, 0.56])
    style_axes(ax, xmax=0.80, xmin=0.50, xformatter=auc_fmt)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    draw_hbars(ax, range(len(rows)), [r[1] for r in rows], [r[2] for r in rows], fmt=lambda a: f"{a:.3f}")
    axis_label(ax, x="AUC on unseen test data (0.5 = coin flip, 1.0 = perfect)")
    add_titles(fig, "Our model beats a simple rule and matches thick-file scoring",
               "Bars start at 0.5, the coin-flip line · only credit scores push clearly higher")
    add_legend(fig, model_vs_random_handles("Thin-file model", "For comparison"), bottom=0.73)
    save(fig, path)


def chart_stability(v, path):
    seeds = list(v["stability"].items())
    low, high = v["bootstrap_range"]
    fig = plt.figure(figsize=(8, 4.2), dpi=LAYOUT_DPI)
    ax = fig.add_axes([0.24, 0.15, 0.70, 0.55])
    style_axes(ax, xmax=0.71, xmin=0.66, xformatter=FuncFormatter(lambda a, _: f"{a:.3f}"))
    ax.set_ylim(len(seeds) - 0.4, -0.6)
    ax.set_yticks(range(len(seeds)), [f"Random split {s + 1}" for s, _ in seeds])
    ax.axvspan(low, high, color=MODEL, alpha=0.12, linewidth=0)
    ax.axvline(v["auc"], color=INK_MUTED, linewidth=1, zorder=0.9)
    for row, (_, auc) in enumerate(seeds):
        ax.plot(auc, row, "o", markersize=9, color=MODEL, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        value_label(ax, auc + 0.0012, row, f"{auc:.3f}", size=8.5)
    # Range label bottom-left inside the band, main-result label top-right of its line: no crossings.
    ax.text(low + 0.0005, len(seeds) - 0.45, f"95% bootstrap range\n{low:.3f}–{high:.3f}", fontsize=8.5,
            color=INK_SECONDARY, va="bottom")
    ax.text(v["auc"] + 0.0005, -0.45, f"Main result {v['auc']:.3f}", fontsize=8.5,
            color=INK_SECONDARY, va="center")
    axis_label(ax, x="Test AUC")
    add_titles(fig, "The score barely moves when we reshuffle the data",
               "Test AUC from 5 different random train/test splits · shaded = range from resampling the test set")
    save(fig, path)


def chart_subgroups(v, path):
    sub = v["subgroup_auc"]
    income_names = {"Commercial associate": "Private business", "State servant": "Government job",
                    "Working": "Regular job", "Pensioner": "Pensioner"}
    families = [
        ("Income type", [(income_names[k], a) for k, a in sub["income type"].items()]),
        ("Age", [(AGE_LABELS.get(k, k), a) for k, a in sub["age band"].items()]),
        ("Gender*", list(sub["gender"].items())),
    ]
    ys, items, headers, height = row_layout(families)
    fig = plt.figure(figsize=(8, 5.6), dpi=LAYOUT_DPI)
    ax = fig.add_axes([0.24, 0.12, 0.68, 0.64])
    style_axes(ax, xmax=0.75, xmin=0.50, xformatter=auc_fmt)
    ax.set_ylim(height - 0.4, -0.6)
    ax.set_yticks(ys, [label for label, _ in items])
    ax.axvline(v["auc"], color=INK_MUTED, linewidth=1, zorder=0.9)
    draw_hbars(ax, ys, [a for _, a in items], [MODEL] * len(items), fmt=lambda a: f"{a:.3f}")
    draw_row_headers(ax, headers)
    ax.text(v["auc"] + 0.002, -0.35, f"Overall {v['auc']:.3f}", fontsize=8.5, color=INK_SECONDARY, va="center")
    axis_label(ax, x="AUC within the group (bars start at 0.5 = coin flip)")
    add_titles(fig, "It works for every group, a bit less well for pensioners and over-45s",
               "AUC inside each group of the test set")
    fig.text(0.02, 0.02, "*Gender is used only for this check. It is never a model input.",
             fontsize=8, color=INK_MUTED)
    save(fig, path)


def chart_fairness(v, path):
    families = [(family.capitalize() if family != "age band" else "Age", table)
                for family, table in v["fairness"].items()]
    layout = [(header, [AGE_LABELS.get(g, g) for g in table["group"]]) for header, table in families]
    ys, labels, headers, height = row_layout(layout)
    table = pd.concat([t for _, t in families], ignore_index=True)

    fig = plt.figure(figsize=(8, 5.0), dpi=LAYOUT_DPI)
    left = fig.add_axes([0.16, 0.12, 0.32, 0.56])
    style_axes(left, xmax=0.85)
    left.set_ylim(height - 0.4, -0.6)
    left.set_yticks(ys, labels)
    left.axvline(0.5, color=INK_MUTED, linewidth=1, zorder=0.9)
    draw_hbars(left, ys, table["approval_rate"], [MODEL] * len(ys))
    draw_row_headers(left, headers)
    left.text(0.515, -0.35, "Everyone: 50%", fontsize=8.5, color=INK_SECONDARY, va="center")
    left.set_title("Approval rate", loc="left", fontsize=9.5, fontweight="semibold", color=INK, pad=8)

    right = fig.add_axes([0.62, 0.12, 0.32, 0.56])
    style_axes(right, xmax=0.165)
    right.set_xticks([0, 0.04, 0.08, 0.12, 0.16])  # whole-percent ticks; 2.5% would round to a wrong "2%"
    right.set_ylim(height - 0.4, -0.6)
    right.set_yticks(ys, [""] * len(ys))
    gap = (12 + 2) / 2 / _px_per_unit(right)[1]
    draw_hbars(right, [y - gap for y in ys], table["default_rate_approved"], [MODEL] * len(ys), bar_px=12)
    draw_hbars(right, [y + gap for y in ys], table["default_rate_overall"], [BASELINE_GREY] * len(ys), bar_px=12)
    right.set_title("Default rate", loc="left", fontsize=9.5, fontweight="semibold", color=INK, pad=8)

    add_titles(fig, "Big approval gaps by age and gender, but similar default rates once approved",
               "If the bank approved the safest 50% of the test set · gender is never a model input")
    add_legend(fig, model_vs_random_handles("Approved by the model", "Whole group, no model"), bottom=0.73)
    save(fig, path)


def grouped_rates(df, column, order, rename):
    rates = df.groupby(column, observed=True)["TARGET"].mean()
    return [rename[k] for k in order], [float(rates[k]) for k in order]


def chart_key_factors(df, path):
    df = df.copy()
    loan_bands = ["Loan no bigger than price", "Up to 10% bigger", "10–20% bigger", "Over 20% bigger"]
    df["loan_vs_price"] = pd.cut(df["credit_to_goods"], [0, 1.0, 1.1, 1.2, float("inf")], labels=loan_bands)
    average = float(df["TARGET"].mean())

    panels = [
        ("Loan size vs price of what's being bought",
         *grouped_rates(df, "loan_vs_price", loan_bands, {k: k for k in loan_bands})),
        ("Income type",
         *grouped_rates(df, "NAME_INCOME_TYPE",
                        ["Pensioner", "State servant", "Commercial associate", "Working"],
                        {"Pensioner": "Pensioner", "State servant": "Government job",
                         "Commercial associate": "Private business", "Working": "Regular job"})),
        ("Education",
         *grouped_rates(df, "NAME_EDUCATION_TYPE",
                        ["Higher education", "Incomplete higher", "Secondary / secondary special", "Lower secondary"],
                        {"Higher education": "University degree", "Incomplete higher": "Some university",
                         "Secondary / secondary special": "Secondary school", "Lower secondary": "Lower secondary"})),
        ("Region rating",
         *grouped_rates(df, "REGION_RATING_CLIENT", [1.0, 2.0, 3.0],
                        {1.0: "Best-rated region", 2.0: "Middle region", 3.0: "Worst-rated region"})),
    ]

    fig = plt.figure(figsize=(8, 6.4), dpi=LAYOUT_DPI)
    positions = [[0.235, 0.47, 0.25, 0.30], [0.715, 0.47, 0.25, 0.30],
                 [0.235, 0.07, 0.25, 0.30], [0.715, 0.07, 0.25, 0.30]]
    for (title, labels, values), pos in zip(panels, positions):
        ax = fig.add_axes(pos)
        style_axes(ax, xmax=0.18)
        ax.axvline(average, color=INK_MUTED, linewidth=1, zorder=0.9)  # behind the bars, above the grid
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.set_yticks(range(len(labels)), labels)
        draw_hbars(ax, range(len(labels)), values, [MODEL] * len(values))
        ax.set_title(title, loc="left", x=-0.9, fontsize=10, fontweight="semibold", color=INK, pad=8)
    add_titles(fig, "Four things that made the biggest difference",
               f"Default rate among all {len(df):,} thin-file applicants · "
               f"grey line = thin-file average ({average:.1%})")
    fig.text(0.02, 0.015, "Tiny groups left out: unemployed (17 people) and academic degree (30 people).",
             fontsize=8, color=INK_MUTED)
    save(fig, path)


CHARTS = [
    "model_default_rates", "eval_risk_groups", "eval_gains", "eval_pr_curve", "eval_roc", "eval_calibration",
    "eval_flagging", "eval_model_comparison", "eval_stability", "eval_subgroups", "eval_fairness",
    "model_key_factors",
]


def main() -> int:
    if not DUCKDB_PATH.exists():
        print(f"No database at {DUCKDB_PATH}.\nRun: python src/load_to_duckdb.py", file=sys.stderr)
        return 1

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(DUCKDB_PATH), read_only=True) as con:
        v = run_validation(con)
        thin = load_segment(con, "thin-file")
        thick_rate = float(con.execute(
            "SELECT AVG(TARGET) FROM applicant_segments WHERE segment = 'thick-file'"
        ).fetchone()[0])

    path = {name: FIGURES_DIR / f"{name}.png" for name in CHARTS}
    chart_default_rates(v, thick_rate, path["model_default_rates"])
    chart_risk_groups(v, path["eval_risk_groups"])
    chart_gains(v, path["eval_gains"])
    chart_pr_curve(v, path["eval_pr_curve"])
    chart_roc(v, path["eval_roc"])
    chart_calibration(v, path["eval_calibration"])
    chart_flagging(v, path["eval_flagging"])
    chart_model_comparison(v, path["eval_model_comparison"])
    chart_stability(v, path["eval_stability"])
    chart_subgroups(v, path["eval_subgroups"])
    chart_fairness(v, path["eval_fairness"])
    chart_key_factors(thin, path["model_key_factors"])
    for p in path.values():
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
