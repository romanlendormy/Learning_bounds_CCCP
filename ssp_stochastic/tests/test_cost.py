"""
Tests Module B — coûts salariaux.
"""

import numpy as np
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from ssp_stochastic.cost import generate_cost, shift_cost, WORK_INDICATOR


def test_work_indicators():
    """W=1, P=R=O=0."""
    assert WORK_INDICATOR['W'] == 1
    for sym in ('P', 'R', 'O'):
        assert WORK_INDICATOR[sym] == 0


def test_cost_vector_shape():
    rng = np.random.default_rng(0)
    s, _ = generate_cost(24, rng)
    assert s.shape == (24,)
    assert (s > 0).all()


def test_determinism():
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    s1, p1 = generate_cost(24, rng1)
    s2, p2 = generate_cost(24, rng2)
    np.testing.assert_array_equal(s1, s2)
    assert p1 == p2


def test_known_shift():
    """Quart connu : WWWOPWWOOOO... — coût = somme de s[t] pour t avec W."""
    T = 8
    rng = np.random.default_rng(1)
    s, _ = generate_cost(T, rng)
    word = "WWWOPWOO"
    expected = s[0] + s[1] + s[2] + s[5]  # positions 0,1,2,5 sont W
    assert abs(shift_cost(word, s) - expected) < 1e-9


def test_all_O_zero_cost():
    rng = np.random.default_rng(2)
    s, _ = generate_cost(12, rng)
    assert shift_cost("O" * 12, s) == 0.0


def test_all_W_full_cost():
    rng = np.random.default_rng(3)
    s, _ = generate_cost(12, rng)
    assert abs(shift_cost("W" * 12, s) - s.sum()) < 1e-9


def test_soir_premium():
    """La fenêtre soir a un coût strictement plus élevé que b0 seul."""
    rng = np.random.default_rng(4)
    # Force b_soir et b_nuit positifs avec b0 > 0
    s, params = generate_cost(24, rng,
                               b0_range=(1.0, 1.0),
                               b_soir_range=(0.5, 0.5),
                               b_nuit_range=(0.0, 0.0))
    lo, hi = params["soir_window_lo"], params["soir_window_hi"]
    if lo < hi:
        assert s[lo] > 1.0  # b0 + b_soir = 1.5


def test_params_in_range():
    rng = np.random.default_rng(5)
    b0_range = (2.0, 3.0)
    _, p = generate_cost(24, rng, b0_range=b0_range)
    assert b0_range[0] <= p["b0"] <= b0_range[1]
