"""The business side of the model: what each mistake costs, and the best cut-off.

The lender declines every applicant whose predicted probability of default is at or
above a cut-off. Two mistakes are possible:

  * approving someone who then defaults   -> cost MISSED_DEFAULT (5 by assumption)
  * declining someone who would have repaid -> cost REJECTED_GOOD (1)

5 : 1 is an assumption (the textbook German Credit cost matrix uses the same ratio), so
every result is also checked at 3 : 1 and 10 : 1. Costs are reported per 1,000 applicants.
"""

from __future__ import annotations

import numpy as np

MISSED_DEFAULT = 5
REJECTED_GOOD = 1
RATIO_RANGE = (3, 5, 10)
CUTOFF_GRID = np.round(np.arange(0.005, 0.9951, 0.005), 3)


def cost_per_1000(y_true, prob, cutoff: float, ratio: float = MISSED_DEFAULT) -> float:
    """Total cost per 1,000 applicants when everyone with prob >= cutoff is declined."""
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    declined = prob >= cutoff
    missed = np.sum(~declined & (y_true == 1))
    rejected = np.sum(declined & (y_true == 0))
    return 1000 * (ratio * missed + REJECTED_GOOD * rejected) / len(y_true)


def approve_everyone_cost(y_true, ratio: float = MISSED_DEFAULT) -> float:
    """The no-model baseline: every defaulter gets through."""
    return 1000 * ratio * float(np.mean(y_true))


def theoretical_cutoff(ratio: float = MISSED_DEFAULT) -> float:
    """With honest probabilities, declining pays off once p * ratio > (1 - p): p > 1 / (1 + ratio)."""
    return REJECTED_GOOD / (REJECTED_GOOD + ratio)


def best_cutoff(y_true, prob, ratio: float = MISSED_DEFAULT, grid=CUTOFF_GRID):
    """(cut-off, cost) with the lowest cost on the given predictions, plus the whole cost curve."""
    curve = np.array([cost_per_1000(y_true, prob, t, ratio) for t in grid])
    i = int(np.argmin(curve))
    return float(grid[i]), float(curve[i]), curve


def confusion_per_1000(y_true, prob, cutoff: float) -> np.ndarray:
    """2x2 counts per 1,000 applicants: rows = repaid/defaulted, columns = approved/declined."""
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    d = prob >= cutoff
    n = len(y_true) / 1000
    return np.array([[np.sum(~d & (y_true == 0)), np.sum(d & (y_true == 0))],
                     [np.sum(~d & (y_true == 1)), np.sum(d & (y_true == 1))]]) / n
