"""Tests for src/costs.py: the money side of the decision, checked against hand calculations."""

import numpy as np
import pytest

from costs import (
    approve_everyone_cost,
    best_cutoff,
    confusion_per_1000,
    cost_per_1000,
    theoretical_cutoff,
)

Y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])                  # 2 defaulters in 10
P = np.array([0.05, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.6, 0.4, 0.9])


def test_cost_matches_a_hand_calculation():
    # cut-off 0.35: declined = {0.6 (good), 0.4 (bad), 0.9 (bad)} -> 1 rejected good, 0 missed
    assert cost_per_1000(Y, P, 0.35) == pytest.approx(1000 * (5 * 0 + 1 * 1) / 10)
    # cut-off 0.5: the defaulter at 0.4 is approved -> 1 missed (x5) + 1 rejected good (x1)
    assert cost_per_1000(Y, P, 0.5) == pytest.approx(1000 * (5 * 1 + 1 * 1) / 10)


def test_approve_everyone_is_every_defaulter_times_the_ratio():
    assert approve_everyone_cost(Y) == pytest.approx(1000 * 5 * 0.2)
    assert approve_everyone_cost(Y, ratio=10) == pytest.approx(1000 * 10 * 0.2)


def test_theoretical_cutoff_is_one_over_one_plus_ratio():
    assert theoretical_cutoff(5) == pytest.approx(1 / 6)
    assert theoretical_cutoff(10) < theoretical_cutoff(5) < theoretical_cutoff(3)


def test_best_cutoff_finds_the_cheapest_point_on_the_curve():
    cut, cost, curve = best_cutoff(Y, P)
    assert cost == pytest.approx(min(curve))
    assert cost_per_1000(Y, P, cut) == pytest.approx(cost)
    assert cost == pytest.approx(100.0)                       # declining 0.4+ costs 1 rejected good


def test_confusion_counts_add_up_to_1000():
    cm = confusion_per_1000(Y, P, 0.35)
    assert cm.sum() == pytest.approx(1000)
    assert cm[1].sum() == pytest.approx(200)                  # 20% defaulters
    assert cm[1, 1] == pytest.approx(200)                     # both defaulters declined at 0.35
