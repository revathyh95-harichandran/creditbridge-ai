"""Shared chart style for the decision-model charts (same palette as make_charts.py)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import REPORTS_DIR  # noqa: E402

FIGURES_DIR = REPORTS_DIR / "figures"

BLUE = "#2a78d6"     # the model / main series
ORANGE = "#eb6834"   # second series, warning, "wrong way"
GREEN = "#1baf7a"    # third series
GREY = "#898781"     # baseline / comparison
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "font.size": 10,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def clean_axes(ax, grid_axis="y"):
    """No box, light grid lines on one axis."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path
